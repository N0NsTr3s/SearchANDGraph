"""
Canonical Data Pipeline Orchestrator
Enforces strict sequential processing: Crawl -> Translate -> Process -> Graph

This module ensures all NLP and graph operations work with clean, translated, canonical text.
"""
import asyncio
import re
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from utils.logger import setup_logger
from urllib.parse import urlparse

logger = setup_logger(__name__)


@dataclass
class ContentItem:
    """Represents a single piece of content in the pipeline."""
    url: str
    raw_content: str
    content_type: str  # 'webpage', 'pdf', 'image'
    source_url: Optional[str] = None
    translated_content: Optional[str] = None
    entities: Optional[List[Any]] = None
    relations: Optional[Dict[Tuple[str, str], List[Any]]] = None
    entity_metadata: Optional[Dict[str, Any]] = None
    is_relevant: bool = False
    relevance_score: float = 0.0


class CanonicalPipeline:
    """
    Orchestrates the canonical data pipeline with strict stage separation.
    
    Pipeline Stages:
    1. Crawl & Enqueue: Fetch raw content, place in translation queue
    2. Translate: Normalize all text to target language (English)
    3. NLP Processing: Extract entities, link to QIDs, classify relations
    4. Graph Aggregation: Add canonical, linked entities to knowledge graph
    5. Analyze & Re-prompt: Identify high-value entities for next iteration
    """
    
    def __init__(
        self,
        nlp_processor: Any,
        knowledge_graph: Any,
        query: str,
        target_language: str = "en",
        min_relevance_score: float = 0.1,
        max_workers: int = 4
    ):
        """
        Initialize the canonical pipeline.
        
        Args:
            nlp_processor: NLP processor instance
            knowledge_graph: Knowledge graph builder
            query: Search query
            target_language: Target language for translation (default: English)
            min_relevance_score: Minimum relevance score for content filtering
            max_workers: Number of parallel workers for NLP processing
        """
        self.nlp_processor = nlp_processor
        self.knowledge_graph = knowledge_graph
        self.query = query
        self.target_language = target_language
        # Minimum relevance score used by the pipeline; can be relaxed if needed
        self.min_relevance_score = min_relevance_score
        self.max_workers = max_workers
        
        # Thread pool for parallel NLP processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized pipeline with {max_workers} parallel workers")
        self._executor_shutdown = False
        
        # Pipeline queues
        self.translation_queue: List[ContentItem] = []
        self.processing_queue: List[ContentItem] = []
        self.graph_queue: List[ContentItem] = []
        
        # Statistics
        self.stats = {
            'total_crawled': 0,
            'total_translated': 0,
            'total_processed': 0,
            'total_added_to_graph': 0,
            'skipped_irrelevant': 0,
            'skipped_low_quality': 0
        }

    
    def _clean_text(self, text: str) -> str:
        """
        Aggressively clean text before processing.
        Remove malformed prefixes, trailing punctuation, and artifacts.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove leading numbers + entity patterns like "10George Simion"
        text = re.sub(r'^\d+(?=[A-Z])', '', text)
        
        # Remove malformed prefixes with brackets/symbols like "source]George", "[tag]Entity"
        text = re.sub(r'^[^\w\s]+', '', text)
        
        # Remove trailing punctuation and whitespace
        text = text.strip('.,;:!?-_\'"\\/@#$%^&*()[]{}|<> \t\n\r')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    async def enqueue_content(
        self,
        url: str,
        raw_content: str,
        content_type: str,
        source_url: Optional[str] = None
    ):
        """
        Stage 1: Crawl & Enqueue
        Add raw content to translation queue with aggressive cleaning.
        
        Args:
            url: Content URL
            raw_content: Raw text content
            content_type: Type of content (webpage, pdf, image)
            source_url: Optional source URL
        """
        # Clean the raw content before adding to queue
        cleaned_content = self._clean_text(raw_content)
        
        item = ContentItem(
            url=url,
            raw_content=cleaned_content,
            content_type=content_type,
            source_url=source_url or url
        )
        self.translation_queue.append(item)
        self.stats['total_crawled'] += 1
        logger.debug(f"Enqueued for translation: {url} ({content_type})")
    
    async def translate_batch(self, batch_size: int = 10) -> int:
        """
        Stage 2: Translate (The Great Normalizer)
        Process translation queue in batches, producing canonical translated text.
        
        Args:
            batch_size: Number of items to process in parallel
            
        Returns:
            Number of items translated
        """
        if not self.translation_queue:
            return 0
        
        batch = self.translation_queue[:batch_size]
        self.translation_queue = self.translation_queue[batch_size:]
        
        logger.info(f"Translating batch of {len(batch)} items...")
        
        translated_count = 0
        for item in batch:
            try:
                # Translate if needed (translator handles language detection internally)
                if self.nlp_processor.translator:
                    logger.debug(f"Translating content from {item.url}")
                    # Pass site (URL) so cached translations for this site use consistent IDs
                    translated = self.nlp_processor.translator.translate_if_needed(item.raw_content, site=item.url)
                    item.translated_content = translated
                else:
                    # Translation disabled, use raw content
                    logger.debug("Translation disabled, using raw content")
                    item.translated_content = item.raw_content
                
                # Move to processing queue
                self.processing_queue.append(item)
                translated_count += 1
                self.stats['total_translated'] += 1
                
            except Exception as e:
                logger.warning(f"Translation failed for {item.url}: {e}")
                # On error, use raw content
                item.translated_content = item.raw_content
                self.processing_queue.append(item)
                continue
        
        logger.info(f"Translated {translated_count}/{len(batch)} items")
        return translated_count
    
    def _process_single_item(self, item: ContentItem) -> Tuple[bool, Optional[Exception]]:
        """
        Process a single content item with NLP (relevance check, entity extraction, relation extraction).
        This method is designed to be called in parallel by ThreadPoolExecutor.
        
        Args:
            item: ContentItem to process
            
        Returns:
            Tuple of (success, exception) - success is True if item was processed successfully
        """
        try:
            # Ensure we have translated content
            if not item.translated_content:
                logger.warning(f"No translated content for {item.url}, skipping")
                return False, None
            
            # Step 1: Relevance Check (run after translation)
            # Use pipeline-level min_relevance_score so we can relax it dynamically
            is_relevant, relevance_score, processed_text = self.nlp_processor.is_content_relevant(
                item.translated_content,
                self.query,
                min_relevance_score=self.min_relevance_score,
                site=item.url
            )

            # If the relevance check translated the text (or otherwise processed it),
            # make sure we use that canonical/translated text for entity extraction.
            if processed_text and processed_text != item.translated_content:
                logger.debug(f"Adopting translated text from relevance check for {item.url}")
                item.translated_content = processed_text
            
            item.is_relevant = is_relevant
            item.relevance_score = relevance_score
            
            if not is_relevant:
                logger.debug(f"Content not relevant (score: {relevance_score:.2f}): {item.url}")
                return False, None
            
            logger.debug(f"Content relevant (score: {relevance_score:.2f}): {item.url}")
            
            # Step 2-4: Extract entities, link to QIDs, extract & classify relations
            # This all happens in extract_entities_and_relations with the canonical text
            logger.debug(f"Extracting entities and relations from canonical text...")
            entities, relations, entity_metadata = self.nlp_processor.extract_entities_and_relations(
                item.translated_content,
                item.url,
                skip_translation=True  # Already translated in Stage 2
            )
            
            # Verbose logging for debugging
            logger.info(f"  Extracted {len(entities)} entities and {len(relations)} relations from {item.url}")
            if entity_metadata:
                # Be defensive: entity metadata may contain non-dict values in some paths
                linked_count = 0
                for meta in entity_metadata.values():
                    qid = None
                    if isinstance(meta, dict):
                        qid = meta.get('id')
                    elif isinstance(meta, (list, tuple)) and len(meta) > 0:
                        # Some legacy formats might return a list where first element is an id
                        candidate = meta[0]
                        if isinstance(candidate, str):
                            qid = candidate
                    elif isinstance(meta, str):
                        qid = meta

                    if isinstance(qid, str) and qid.startswith('Q'):
                        linked_count += 1

                logger.info(f"  Linked {linked_count}/{len(entities)} entities to Wikidata QIDs")
            else:
                logger.debug("  No entity metadata available (entity linking may be disabled)")
            
            # Track high-value entities (on translated, canonical text)
            try:
                for entity in entities.keys():
                    # Ensure entity is a valid string
                    if isinstance(entity, str) and entity.strip():
                        self.nlp_processor.track_entity_mention(entity)
            except Exception:
                logger.debug("Failed to track entity mentions for debugging")
            
            item.entities = entities
            item.relations = relations
            item.entity_metadata = entity_metadata
            
            if entities or relations:
                logger.debug(f"Processed: {len(entities)} entities, {len(relations)} relations from {item.url}")
                return True, None
            else:
                logger.debug(f"No entities/relations extracted from {item.url}")
                return False, None
            
        except Exception as e:
            logger.warning(f"NLP processing failed for {item.url}: {e}")
            return False, e
    
    async def process_nlp_batch(self, batch_size: int = 5) -> int:
        """
        Stage 3: NLP Processing (PARALLEL)
        Run full NLP pipeline on translated, canonical text using ThreadPoolExecutor.
        
        Pipeline Steps:
        - Content Relevance Check
        - Entity Extraction
        - Entity Linking (Wikidata QIDs)
        - Relation Extraction & Classification
        
        Args:
            batch_size: Number of items to process in parallel
            
        Returns:
            Number of items processed
        """
        if not self.processing_queue:
            return 0
        
        batch = self.processing_queue[:batch_size]
        self.processing_queue = self.processing_queue[batch_size:]
        
        logger.info(f"Processing NLP batch of {len(batch)} items in parallel with {self.max_workers} workers...")
        
        # Submit all items to thread pool for parallel processing
        loop = asyncio.get_event_loop()
        futures = []
        
        for item in batch:
            future = loop.run_in_executor(
                self.executor,
                self._process_single_item,
                item
            )
            futures.append((item, future))
        
        # Wait for all futures to complete
        processed_count = 0
        for item, future in futures:
            try:
                success, exception = await future
                
                if success:
                    # Move to graph queue
                    self.graph_queue.append(item)
                    processed_count += 1
                    self.stats['total_processed'] += 1
                elif exception:
                    # Processing failed with exception (already logged)
                    pass
                elif not item.is_relevant:
                    # Item was not relevant
                    self.stats['skipped_irrelevant'] += 1
                elif not item.translated_content:
                    # No content
                    self.stats['skipped_low_quality'] += 1
                else:
                    # No entities/relations extracted
                    self.stats['skipped_low_quality'] += 1
                    
            except Exception as e:
                logger.error(f"Error waiting for NLP processing of {item.url}: {e}")
                continue
        
        logger.info(f"NLP processed {processed_count}/{len(batch)} items in parallel")
        return processed_count

    def shutdown(self, wait: bool = True):
        """Gracefully shutdown the internal ThreadPoolExecutor."""
        if getattr(self, 'executor', None) and not getattr(self, '_executor_shutdown', False):
            try:
                logger.info("Shutting down pipeline executor...")
                self.executor.shutdown(wait=wait)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
            finally:
                self._executor_shutdown = True

    async def _process_and_aggregate_site(self, site_items: List[ContentItem]) -> int:
        """
        Translate, process and aggregate all items for a single website (site_items).

        This method ensures we fully translate and process every page belonging to
        a site and then immediately merge the results into the knowledge graph
        before switching context to another site. It avoids leaking partial
        site-state across sites and supports consistent per-site translation
        caching (we pass each item's URL as the site argument to the translator).

        Returns the number of items successfully aggregated.
        """
        if not site_items:
            return 0

        logger.info(f"Processing site batch of {len(site_items)} items: {site_items[0].url}")

        # Stage A: Translate all items for this site
        translated_count = 0
        for item in site_items:
            try:
                if self.nlp_processor.translator:
                    item.translated_content = self.nlp_processor.translator.translate_if_needed(
                        item.raw_content,
                        site=item.url
                    )
                else:
                    item.translated_content = item.raw_content

                translated_count += 1
                self.stats['total_translated'] += 1
            except Exception as e:
                logger.warning(f"Translation failed for {item.url} during site batch: {e}")
                item.translated_content = item.raw_content

        logger.info(f"Translated {translated_count}/{len(site_items)} items for site")

        # Stage B: Run NLP processing in parallel on translated content
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.executor, self._process_single_item, item) for item in site_items]

        processed_count = 0
        for item, fut in zip(site_items, futures):
            try:
                success, exception = await fut
                if success:
                    # Immediately aggregate successful items into the knowledge graph
                    try:
                        if item.entities and item.relations:
                            self.knowledge_graph.merge_entities_and_relations(
                                item.entities,
                                item.relations,
                                item.entity_metadata or {}
                            )
                            processed_count += 1
                            self.stats['total_processed'] += 1
                            self.stats['total_added_to_graph'] += 1
                            logger.info(f"  Aggregated: {len(item.entities)} entities from {item.url}")
                        else:
                            self.stats['skipped_low_quality'] += 1
                    except Exception as e:
                        logger.warning(f"Graph aggregation failed for {item.url} in site batch: {e}")
                elif exception:
                    logger.debug(f"Processing failed for {item.url} in site batch: {exception}")
                elif not item.is_relevant:
                    self.stats['skipped_irrelevant'] += 1
                else:
                    self.stats['skipped_low_quality'] += 1
            except Exception as e:
                logger.error(f"Error waiting for NLP processing of {item.url} in site batch: {e}")

        logger.info(f"Site batch processing complete: aggregated {processed_count}/{len(site_items)} items")

        # Stage C: After adding all items for this site, run a unification/merge pass
        try:
            if hasattr(self.knowledge_graph, 'unify_duplicate_nodes'):
                logger.info("Running per-site graph unification to merge duplicates before switching context")
                self.knowledge_graph.unify_duplicate_nodes()
            else:
                logger.debug("Knowledge graph does not support unify_duplicate_nodes()")
        except Exception as e:
            logger.warning(f"Graph unification failed after site batch: {e}")

        return processed_count

    
    async def aggregate_to_graph_batch(self, batch_size: int = 10) -> int:
        """
        Stage 4: Graph Aggregation
        Add canonical, translated, linked entities to knowledge graph.
        
        Args:
            batch_size: Number of items to aggregate
            
        Returns:
            Number of items added to graph
        """
        if not self.graph_queue:
            return 0
        
        batch = self.graph_queue[:batch_size]
        self.graph_queue = self.graph_queue[batch_size:]
        
        logger.info(f"Aggregating {len(batch)} items to graph...")
        
        aggregated_count = 0
        total_entities_added = 0
        total_relations_added = 0
        
        for item in batch:
            try:
                if item.entities and item.relations:
                    entity_count = len(item.entities)
                    relation_count = len(item.relations)
                    
                    # Add to knowledge graph (entities are now in English and linked to QIDs)
                    self.knowledge_graph.merge_entities_and_relations(
                        item.entities,
                        item.relations,
                        item.entity_metadata or {}
                    )
                    
                    aggregated_count += 1
                    total_entities_added += entity_count
                    total_relations_added += relation_count
                    self.stats['total_added_to_graph'] += 1
                    
                    logger.info(f"  Added to graph: {entity_count} entities, {relation_count} relations from {item.url}")
            
            except Exception as e:
                logger.warning(f"Graph aggregation failed for {item.url}: {e}")
                continue
        
        logger.info(f"Aggregated {aggregated_count}/{len(batch)} items to graph")
        logger.info(f"  Total: {total_entities_added} entities, {total_relations_added} relations added")
        return aggregated_count
    
    async def run_full_pipeline(self):
        """
        Run the complete pipeline: Translate -> Process -> Aggregate
        Processes all queued items through all stages.
        """
        logger.info("=" * 70)
        logger.info("Running Canonical Pipeline")
        logger.info(f"Queue sizes: Translation={len(self.translation_queue)}, "
                   f"Processing={len(self.processing_queue)}, "
                   f"Graph={len(self.graph_queue)}")
        logger.info("=" * 70)
        # NEW BEHAVIOR: Process items grouped by site (domain). For each site we:
        # 1) Translate all pages from that site (site-aware cache keys)
        # 2) Run NLP processing on all translated pages for the site
        # 3) Aggregate results into the graph
        # 4) Run a unification pass so the site's data is merged before switching
        # This prevents partial cross-site state from confusing entity merging or
        # hub selection and ensures per-site translation caching is consistent.

        if not self.translation_queue:
            logger.info("No items to process in translation queue")
        else:
            # Group queued items by site (netloc)
            site_map: Dict[str, List[ContentItem]] = {}
            for item in list(self.translation_queue):
                try:
                    parsed = urlparse(item.url)
                    site_key = parsed.netloc or parsed.path or item.url
                except Exception:
                    site_key = item.url

                site_map.setdefault(site_key, []).append(item)

            # Clear the global translation queue (we'll process per-site)
            self.translation_queue.clear()

            # Process each site sequentially (to keep per-site context isolated)
            for site_key, items in site_map.items():
                logger.info(f"Starting site-level processing for: {site_key} ({len(items)} pages)")
                try:
                    # Enforce a timeout for a site's translation & processing to avoid hangs
                    await asyncio.wait_for(self._process_and_aggregate_site(items), timeout=120.0)
                except asyncio.TimeoutError:
                    logger.error(f"Site-level processing for {site_key} timed out. Skipping remaining pages for site.")
                    continue
                except Exception as e:
                    logger.warning(f"Site-level processing failed for {site_key}: {e}")
                    continue

        # After processing all sites, there should be no leftover processing/graph queues
        # but drain them just in case (fallback for other code paths).
        while self.processing_queue:
            await self.process_nlp_batch(batch_size=5)

        while self.graph_queue:
            await self.aggregate_to_graph_batch(batch_size=10)
        
        # Report statistics
        logger.info("=" * 70)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Total crawled: {self.stats['total_crawled']}")
        logger.info(f"  Total translated: {self.stats['total_translated']}")
        logger.info(f"  Total NLP processed: {self.stats['total_processed']}")
        logger.info(f"  Total added to graph: {self.stats['total_added_to_graph']}")
        logger.info(f"  Skipped (irrelevant): {self.stats['skipped_irrelevant']}")
        logger.info(f"  Skipped (low quality): {self.stats['skipped_low_quality']}")
        logger.info("=" * 70)
    
    def analyze_and_reprompt(self, max_entities: int = 5) -> List[Tuple[str, str, int]]:
        """
        Stage 5: Analyze & Re-prompt
        Analyze aggregated, translated entities to find high-value targets.
        
        Args:
            max_entities: Maximum number of entities to return
            
        Returns:
            List of (entity_name, entity_type, degree) tuples
        """
        if not self.knowledge_graph.graph or self.knowledge_graph.graph.number_of_nodes() == 0:
            logger.warning("No entities in graph for analysis")
            return []
        
        logger.info("Analyzing graph for high-value entities...")
        
        # Get entities by degree (most connected)
        node_degrees = dict(self.knowledge_graph.graph.degree())
        sorted_by_degree = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Filter and extract entity info
        high_value_entities = []
        query_lower = self.query.lower()
        
        # === NEW: Define the only entity types allowed for deep-dive ===
        ALLOWED_TYPES_FOR_DEEP_DIVE = {'PERSON', 'ORG'}

        for entity, degree in sorted_by_degree:
            if entity.lower() == query_lower:
                continue  # Skip the query itself
            
            if degree < 2:
                continue  # Skip weakly connected entities
            
            # Get entity type from graph
            entity_type = "UNKNOWN"
            if self.knowledge_graph.graph.has_node(entity):
                node_data = self.knowledge_graph.graph.nodes[entity]
                entity_type = node_data.get('label', 'UNKNOWN')

            # === NEW: Apply the type filter ===
            if entity_type not in ALLOWED_TYPES_FOR_DEEP_DIVE:
                logger.debug(f"Skipping entity '{entity}' for deep-dive (type: {entity_type} is not in allowed list).")
                continue

            high_value_entities.append((entity, entity_type, degree))

            if len(high_value_entities) >= max_entities:
                break
        
        logger.info(f"Identified {len(high_value_entities)} high-value entities:")
        for entity, entity_type, degree in high_value_entities:
            logger.info(f"  - {entity} ({entity_type}, degree={degree})")
        
        return high_value_entities
    
    def get_statistics(self) -> Dict[str, int]:
        """Get pipeline statistics."""
        return self.stats.copy()
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            'translation_queue': len(self.translation_queue),
            'processing_queue': len(self.processing_queue),
            'graph_queue': len(self.graph_queue)
        }
    
    def shutdown(self):
        """Shutdown the thread pool executor gracefully."""
        if self.executor:
            logger.info("Shutting down parallel processing executor...")
            self.executor.shutdown(wait=True)
            logger.info("Executor shutdown complete")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass
