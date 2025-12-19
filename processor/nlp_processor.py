"""
NLP processing module for entity and relation extraction.
"""
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import re
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Mapping, Sequence, Any
try:
    from ..utils.config import NLPConfig, CacheConfig
    from ..utils.logger import setup_logger
    from ..utils.translator import TextTranslator
except:
    from utils.config import NLPConfig, CacheConfig
    from utils.logger import setup_logger
    from utils.translator import TextTranslator
from processor.provenance import Provenance, migrate_legacy_reasons
import diskcache

logger = setup_logger(__name__)

# Import advanced NLP enhancements
try:
    from processor.nlp_enhancements import (
        CoreferenceResolver,
        EnhancedRelationExtractor,
        EntityLinker,
        RelationConfidenceScorer
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NLP enhancements not available: {e}")
    ENHANCEMENTS_AVAILABLE = False


    @Language.component("refine_ner_boundaries")
    def refine_ner_boundaries(doc: Doc) -> Doc:
        """
        Custom spaCy pipeline component to refine and filter NER entities.
        This component runs after the standard 'ner' component.
        """
        new_ents = []
        for ent in doc.ents:
            # --- Rule 1: Trim leading/trailing punctuation and stop words ---
            start = ent.start
            end = ent.end

            # Trim leading tokens
            while start < end and (doc[start].is_punct or doc[start].is_stop or doc[start].pos_ == 'ADP'):
                start += 1

            # Trim trailing tokens
            while end > start and (doc[end - 1].is_punct or doc[end - 1].is_stop or doc[end - 1].pos_ == 'PRON'):
                end -= 1

            # If the entity has changed, create a new span
            if start != ent.start or end != ent.end:
                # Create a char-span for the trimmed span
                try:
                    start_char = doc[start].idx
                    end_char = doc[end - 1].idx + len(doc[end - 1])
                    new_ent = doc.char_span(start_char, end_char, label=ent.label_)
                except Exception:
                    new_ent = None

                if new_ent:
                    # Further validation on the new, trimmed entity
                    if len(new_ent.text.split()) > 0:  # Ensure not empty
                        new_ents.append(new_ent)
            else:
                # Original entity is fine, keep it
                new_ents.append(ent)

        # --- Rule 2: Filter out any remaining invalid entities ---
        final_ents = []
        for ent in new_ents:
            # Discard if it contains a verb (strong indicator of a bad entity)
            if any(token.pos_ == 'VERB' for token in ent):
                continue

            # Discard if it's just a single stopword
            if len(ent) == 1 and ent[0].is_stop:
                continue

            final_ents.append(ent)

        try:
            doc.ents = final_ents
        except ValueError:
            # spaCy can raise a ValueError if spans overlap after trimming.
            # In this case, fall back to original entities.
            pass

        return doc

# Import relation classifier
try:
    from processor.relation_classifier import RelationClassifier
    RELATION_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Relation classifier not available: {e}")
    RELATION_CLASSIFIER_AVAILABLE = False


class NLPProcessor:
    """Handles NLP processing for entity and relation extraction."""
    
    def __init__(self, config: NLPConfig, cache_config: Optional[CacheConfig] = None):
        """
        Initialize the NLP processor.
        
        Args:
            config: NLP configuration settings
            cache_config: Cache configuration settings
        """
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        
        logger.info(f"Loading spaCy model: {config.spacy_model}")
        try:
            self.nlp = spacy.load(config.spacy_model)
            # Add our custom component to the pipeline after the 'ner' component.
            if "refine_ner_boundaries" not in self.nlp.pipe_names:
                try:
                    self.nlp.add_pipe("refine_ner_boundaries", after="ner")
                    logger.info("Added custom 'refine_ner_boundaries' component to spaCy pipeline.")
                except Exception as e:
                    logger.warning(f"Failed to add 'refine_ner_boundaries' to spaCy pipeline: {e}")
        except OSError:
            logger.error(f"Model '{config.spacy_model}' not found. Please install it using: "
                        f"python -m spacy download {config.spacy_model}")
            raise
        
        # Initialize translator if enabled
        self.translator: Optional[TextTranslator] = None
        if config.enable_translation:
            self.translator = TextTranslator(
                target_language=config.target_language,
                provider=config.translation_provider,
                cache_config=self.cache_config  # Pass cache config to translator
            )
            logger.info("Translation enabled for better entity detection")
        
        # Initialize cache
        if self.cache_config.enabled:
            self.cache = diskcache.Cache(
                self.cache_config.cache_dir,
                size_limit=self.cache_config.max_cache_size
            )
            logger.info(f"NLP cache enabled at {self.cache_config.cache_dir}")
        else:
            self.cache = None
            logger.info("NLP cache disabled")
        
        # Initialize thread pool for parallel processing
        self.executor = None
        if config.parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            logger.info(f"Parallel NLP processing enabled with {config.max_workers} workers")
        
        # Initialize advanced NLP enhancements
        self.coref_resolver = None
        self.enhanced_extractor = None
        self.entity_linker = None
        self.confidence_scorer = None
        self.relation_classifier = None
        
        if ENHANCEMENTS_AVAILABLE:
            if config.enable_coreference:
                self.coref_resolver = CoreferenceResolver()
                logger.info("Coreference resolution enabled")
            
            if config.enable_enhanced_relations:
                self.enhanced_extractor = EnhancedRelationExtractor()
                logger.info("Enhanced relation extraction enabled")
            
            if config.enable_entity_linking:
                self.entity_linker = EntityLinker(
                    threshold=config.entity_linking_threshold,
                    cache_dir=self.cache_config.cache_dir
                )
                logger.info(f"Entity linking enabled (threshold: {config.entity_linking_threshold})")
            
            if config.enable_relation_confidence:
                self.confidence_scorer = RelationConfidenceScorer()
                logger.info(f"Relation confidence scoring enabled (min: {config.min_relation_confidence})")
        
        # Initialize relation classifier
        if RELATION_CLASSIFIER_AVAILABLE and getattr(config, 'enable_relation_classification', True):
            self.relation_classifier = RelationClassifier()
            logger.info("Relation classification enabled")
        
        # High-value entity tracking (for canonical, translated entities only)
        self.entity_mention_count: Dict[str, int] = {}
        self.high_value_entities: set = set()
        logger.info("High-value entity tracking initialized (post-translation)")
    
    def track_entity_mention(self, entity: str, count: int = 1):
        """
        Track mentions of a canonical (translated) entity to identify high-value entities.
        This should ONLY be called on translated text in the target language.
        
        Args:
            entity: Entity name (must be in target language)
            count: Number of mentions to add (default 1)
        """
        if not entity or len(entity) < 3:
            return
        
        if entity not in self.entity_mention_count:
            self.entity_mention_count[entity] = 0
        self.entity_mention_count[entity] += count
        
        # Mark as high-value if mentioned frequently
        if self.entity_mention_count[entity] >= 3:  # Threshold
            if entity not in self.high_value_entities:
                self.high_value_entities.add(entity)
                logger.info(f"Identified high-value entity: '{entity}' ({self.entity_mention_count[entity]} mentions)")
    
    def get_top_entities(self, top_n: int = 10, exclude_query: bool = True, query: str = "") -> List[Tuple[str, int]]:
        """
        Get the most frequently mentioned canonical entities.
        
        Args:
            top_n: Number of top entities to return
            exclude_query: Exclude query entities from results
            query: Query string to exclude
            
        Returns:
            List of (entity, mention_count) tuples, sorted by frequency
        """
        sorted_entities = sorted(
            self.entity_mention_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if exclude_query and query:
            # Filter out query entities
            query_terms = set(query.lower().split())
            sorted_entities = [
                (entity, count) for entity, count in sorted_entities
                if entity.lower() not in query_terms
            ]
        
        return sorted_entities[:top_n]
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the NLP processor and wait for all background tasks to complete.
        
        Args:
            wait: If True, wait for all pending tasks to complete
            timeout: Maximum time to wait for tasks to complete (seconds)
        """
        if self.executor:
            logger.info("Shutting down NLP processor thread pool...")
            if wait:
                logger.info(f"Waiting up to {timeout}s for background tasks to complete...")
                self.executor.shutdown(wait=True, cancel_futures=False)
                logger.info("✓ All NLP background tasks completed")
            else:
                self.executor.shutdown(wait=False, cancel_futures=True)
                logger.info("✓ NLP processor shutdown (tasks cancelled)")
        
        if self.cache:
            logger.debug("Closing NLP cache")
            self.cache.close()
    
    def _normalize_romanian_text(self, text: str) -> str:
        """
        Normalize Romanian text by removing diacritics and converting to lowercase.
        This helps match Romanian names with and without diacritics.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        import unicodedata
        
        # Remove diacritics (ă→a, â→a, î→i, ș→s, ț→t, etc.)
        normalized = unicodedata.normalize('NFKD', text)
        # Remove combining characters (diacritics)
        without_diacritics = ''.join([c for c in normalized if not unicodedata.combining(c)])
        
        return without_diacritics.lower()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"nlp:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _get_cached_extraction(self, text: str) -> Optional[Tuple[Dict[str, str], Dict[Tuple[str, str], List[str]]]]:
        """Get cached NLP extraction result if available."""
        if not self.cache:
            return None
        
        try:
            key = self._get_cache_key(text)
            cached = self.cache.get(key)
            if cached and isinstance(cached, bytes):
                result = pickle.loads(cached)
                logger.debug("NLP cache HIT")
                return result
            logger.debug("NLP cache MISS")
        except Exception as e:
            logger.warning(f"NLP cache read error: {e}")
        
        return None
    
    def _cache_extraction(self, text: str, entities: Dict[str, str], relations: Dict[Tuple[str, str], List[str]]):
        """Cache NLP extraction result."""
        if not self.cache:
            return
        
        try:
            key = self._get_cache_key(text)
            data = pickle.dumps((entities, relations))
            self.cache.set(key, data, expire=self.cache_config.nlp_cache_ttl)
            logger.debug("Cached NLP extraction result")
        except Exception as e:
            logger.warning(f"NLP cache write error: {e}")
    
    def extract_entities_and_relations(
        self, 
        text: str,
        source_url: str | None = None,
        skip_translation: bool = False
    ) -> Tuple[Dict[str, str], Dict[Tuple[str, str], List[str]], Dict[str, Dict]]:
        """
        Extract named entities and their relationships from text.
        Uses cache to avoid reprocessing.
        
        Args:
            text: Input text to process (if already translated, set skip_translation=True)
            source_url: Optional URL where the text came from
            skip_translation: If True, assumes text is already translated
            
        Returns:
            Tuple of (entities dict, relations dict, entity_metadata dict)
            - entities: {entity_text: entity_label}
            - relations: {(entity1, entity2): [reasons]}
            - entity_metadata: {entity_text: {id, label, aliases, ...}}
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return {}, {}, {}
        
        # Check cache first
        cached = self._get_cached_extraction(text)
        if cached:
            # Cache currently returns (entities, relations), add empty metadata
            return cached[0], cached[1], {}
        
        # Translate text if translation is enabled and not already translated
        processed_text = text
        if not skip_translation and self.translator and self.config.enable_translation:
            try:
                translated = self.translator.translate_if_needed(text)
                if translated and translated != text:
                    logger.debug("Text translated for better entity detection")
                    processed_text = translated
            except Exception as e:
                logger.warning(f"Translation failed, using original text: {e}")
                processed_text = text
        else:
            if skip_translation:
                logger.debug("Skipping translation (text already translated)")
        
        doc = self.nlp(processed_text)
        
        # Extract entities (this will deduplicate and merge reordered names)
        entities = self._extract_entities(doc)
        
        # Track high-value entities on the canonical, translated text
        for entity_text in entities.keys():
            self.track_entity_mention(entity_text)
        
        # Dictionary to store entity metadata (QID, label, aliases)
        entity_metadata = {}
        
        # 🔥 ENHANCEMENT: Link entities to Wikidata/DBpedia in parallel
        if self.entity_linker and self.config.enable_entity_linking:
            # Use the new batch method for parallel execution
            try:
                batch_metadata = self.entity_linker.link_entities_batch(entities)
                if batch_metadata:
                    # Merge batch results into entity_metadata
                    entity_metadata.update(batch_metadata)

                    # Process the results to add aliases
                    linked_entities = {}
                    for entity_text, linked_info in list(batch_metadata.items()):
                        entity_label = entities.get(entity_text, 'UNKNOWN')
                        for alias in linked_info.get('aliases', [])[:3]:
                            if alias not in entities:
                                linked_entities[alias] = entity_label
                                if alias not in entity_metadata:
                                    entity_metadata[alias] = linked_info
                    entities.update(linked_entities)
            except Exception as e:
                logger.warning(f"Batch entity linking failed: {e}")
        
        # Extract relations (using original doc entities)
        relations_raw = self._extract_relations(doc, source_url)
        
        # 🔥 ENHANCEMENT: Add enhanced pattern-based relations
        if self.enhanced_extractor and self.config.enable_enhanced_relations:
            enhanced_relations = self.enhanced_extractor.extract_enhanced_relations(doc)
            # Merge enhanced relations with standard relations, converting string reasons
            # into Provenance objects so relations_raw consistently stores Provenance instances.
            
            # Validate source URL
            validated_url = None
            if source_url and source_url.strip():
                url_lower = source_url.strip().lower()
                if url_lower.startswith('http://') or url_lower.startswith('https://'):
                    validated_url = source_url.strip()
            
            if validated_url:
                for key, reasons in enhanced_relations.items():
                    prov_reasons = []
                    for r in reasons:
                        if isinstance(r, Provenance):
                            prov_reasons.append(r)
                        else:
                            # Wrap plain reason strings (or other types) into a Provenance object.
                            prov_reasons.append(
                                Provenance(
                                    text=str(r),
                                    source_url=validated_url,
                                    confidence=None,
                                    dates=None,
                                    sentence_offset=0
                                )
                            )
                    if key in relations_raw:
                        relations_raw[key].extend(prov_reasons)
                    else:
                        relations_raw[key] = prov_reasons
                logger.debug(f"Added {len(enhanced_relations)} enhanced relations")
            else:
                logger.warning("Skipping enhanced relations due to missing valid source URL")
        
        # 🔥 ENHANCEMENT: Resolve coreferences and expand relations
        if self.coref_resolver and self.config.enable_coreference:
            coref_map = self.coref_resolver.resolve_coreferences(doc)
            if coref_map:
                # The coref expand method expects relation reasons as List[str].
                # Convert existing Provenance objects (or other types) to strings,
                # call the coref expansion, then wrap returned strings back into Provenance.
                try:
                    relations_for_coref = {}
                    for (s, t), reasons in relations_raw.items():
                        str_reasons = []
                        for r in reasons:
                            if isinstance(r, Provenance):
                                str_reasons.append(r.text)
                            else:
                                str_reasons.append(str(r))
                        relations_for_coref[(s, t)] = str_reasons

                    expanded = self.coref_resolver.expand_relations_with_coreferences(
                        relations_for_coref, coref_map, doc
                    )

                    # Validate source URL for coreference-expanded relations
                    validated_url = None
                    if source_url and source_url.strip():
                        url_lower = source_url.strip().lower()
                        if url_lower.startswith('http://') or url_lower.startswith('https://'):
                            validated_url = source_url.strip()
                    
                    if not validated_url:
                        logger.warning("Skipping coreference-expanded relations due to missing valid source URL")
                    else:
                        # Convert expanded string reasons back into Provenance objects
                        relations_raw = {}
                        for (s, t), reasons in expanded.items():
                            prov_reasons = []
                            for r in reasons:
                                if isinstance(r, Provenance):
                                    prov_reasons.append(r)
                                else:
                                    prov_reasons.append(
                                        Provenance(
                                            text=str(r),
                                            source_url=validated_url,
                                            confidence=None,
                                            dates=None,
                                            sentence_offset=0
                                        )
                                    )
                            relations_raw[(s, t)] = prov_reasons
                except Exception as e:
                    logger.warning(f"Coreference expansion failed: {e}")
                logger.debug(f"Resolved {len(coref_map)} coreferences")
        
        # Remap relations to use deduplicated entity names
        relations = self._remap_relations(relations_raw, entities)
        
        # 🔥 ENHANCEMENT: Apply confidence scoring and filtering
        if self.confidence_scorer and self.config.enable_relation_confidence:
            filtered_relations = {}
            min_conf = self.config.min_relation_confidence
            for (ent1, ent2), reasons in relations.items():
                kept_reasons = []
                for reason in reasons:
                    if isinstance(reason, Provenance):
                        conf = reason.confidence
                        if conf is None or conf >= min_conf:
                            kept_reasons.append(reason)
                        else:
                            logger.debug(
                                "Dropping low-confidence provenance (%s): %.2f",
                                reason.source_url,
                                conf
                            )
                    elif isinstance(reason, str):
                        if 'confidence:' not in reason:
                            kept_reasons.append(reason)
                        else:
                            match = re.search(r'confidence:([\d.]+)', reason)
                            if match:
                                conf = float(match.group(1))
                                if conf >= min_conf:
                                    kept_reasons.append(reason)
                    else:
                        # Unknown reason type – keep it rather than risk data loss
                        kept_reasons.append(reason)

                if kept_reasons:
                    filtered_relations[(ent1, ent2)] = kept_reasons

            relations = filtered_relations
            logger.debug(f"Filtered to {len(relations)} high-confidence relations")
        
        # Cache the result
        self._cache_extraction(text, entities, relations)
        
        logger.debug(f"Extracted {len(entities)} entities and {len(relations)} relations")
        return entities, relations, entity_metadata

    def ingest(self, text: str, source_url: Optional[str] = None, skip_translation: bool = False) -> Dict[str, Any]:
        """
        Convenience ingest method for external callers (crawler, document processor).

        Runs the full extraction pipeline and returns a dict with extracted
        entities (list) and relations (mapping). This method is synchronous
        and may be executed inside a thread via `asyncio.to_thread` by callers.

        Args:
            text: Text to process
            source_url: Optional URL where the text came from
            skip_translation: If True, skip translation step

        Returns:
            Dict with keys: 'entities' -> list[str], 'relations' -> dict, 'metadata' -> dict
        """
        try:
            entities, relations, metadata = self.extract_entities_and_relations(text, source_url, skip_translation)
            return {
                'entities': list(entities.keys()),
                'relations': relations,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"NLP ingest failed for {source_url or '<in-memory>'}: {e}")
            return {'entities': [], 'relations': {}, 'metadata': {}}
    
    def process_pages_parallel(
        self,
        pages: List[Tuple[str, str, Optional[str]]]  # [(text, url, source_url), ...]
    ) -> Tuple[Dict[str, str], Dict[Tuple[str, str], List[str]]]:
        """
        Process multiple pages in parallel using ThreadPoolExecutor.
        
        Args:
            pages: List of (text, url, source_url) tuples
            
        Returns:
            Merged entities and relations from all pages
        """
        if not self.executor or not pages:
            # Fallback to sequential processing
            all_entities = {}
            all_relations = defaultdict(list)
            for text, url, source_url in pages:
                entities, relations, _ = self.extract_entities_and_relations(text, source_url)
                all_entities.update(entities)
                for key, reasons in relations.items():
                    all_relations[key].extend(reasons)
            return all_entities, dict(all_relations)
        
        logger.info(f"Processing {len(pages)} pages in parallel...")
        
        # Submit all tasks
        futures = []
        for text, url, source_url in pages:
            future = self.executor.submit(
                self.extract_entities_and_relations,
                text,
                source_url
            )
            futures.append((future, url))
        
        # Collect results
        all_entities = {}
        all_relations = defaultdict(list)
        completed = 0
        
        for future, url in futures:
            try:
                entities, relations, _ = future.result(timeout=30)
                all_entities.update(entities)
                for key, reasons in relations.items():
                    all_relations[key].extend(reasons)
                completed += 1
            except Exception as e:
                logger.error(f"Parallel processing error for {url}: {e}")
        
        logger.info(f"Parallel processing completed: {completed}/{len(pages)} pages")
        return all_entities, dict(all_relations)

    def build_context_keywords(self, text: str, query: str | None = None, max_keywords: int = 20) -> set:
        """
        Build a small set of context keywords from the provided text and optional query.
        Uses extracted entities and common noun lemmas to create a focused context.

        Args:
            text: Source text to extract keywords from
            query: Optional query to include terms from
            max_keywords: Maximum number of keywords to return

        Returns:
            Set of lowercase keywords
        """
        if not text:
            return set()

        processed_text = text
        if self.translator and self.config.enable_translation:
            try:
                translated = self.translator.translate_if_needed(text)
                if translated and translated != text:
                    processed_text = translated
            except Exception:
                processed_text = text

        doc = self.nlp(processed_text)

        keywords = []
        # include query words first
        if query:
            keywords.extend([w.lower() for w in query.split() if len(w) > 2])

        # add named entities (PERSON, ORG, GPE, NORP, EVENT)
        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'NORP', 'EVENT'}:
                token = ent.text.strip().lower()
                if token and len(token) > 2:
                    keywords.append(token.replace(' ', '_'))

        # add frequent noun lemmas
        noun_lemmas = []
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop:
                noun_lemmas.append(token.lemma_.lower())

        # preserve order and uniqueness
        for kw in noun_lemmas:
            if kw not in keywords and len(kw) > 2:
                keywords.append(kw)

        # return top N keywords as a set
        result = set(keywords[:max_keywords])
        return result
    
    def expand_query_with_synonyms(self, query: str, max_expansions: int = 5) -> set[str]:
        """
        Expand query with related terms using spaCy word vectors.
        
        Args:
            query: Original query string
            max_expansions: Maximum number of expansion terms
            
        Returns:
            Set of query terms including original and expansions
        """
        if not self.config.enable_query_expansion:
            return {query.lower()}
        
        expanded = {query.lower()}
        
        try:
            doc = self.nlp(query)
            
            # Get main tokens (non-stop words, nouns/proper nouns)
            main_tokens = [
                token for token in doc 
                if not token.is_stop and token.pos_ in {'NOUN', 'PROPN'} and token.has_vector
            ]
            
            if not main_tokens:
                return expanded
            
            # Find similar words for each main token
            for token in main_tokens[:3]:  # Limit to first 3 tokens
                # Use spaCy's similarity to find related words
                most_similar = []
                
                # Get vocabulary and find most similar words
                for word in self.nlp.vocab:
                    if word.has_vector and word.is_lower and not word.is_stop:
                        similarity = token.similarity(word)
                        if similarity > 0.7:  # High similarity threshold
                            most_similar.append((word.text, similarity))
                
                # Sort by similarity and take top matches
                most_similar.sort(key=lambda x: x[1], reverse=True)
                for word, sim in most_similar[:max_expansions]:
                    if len(word) > 3:  # Filter short words
                        expanded.add(word.lower())
                        logger.debug(f"Query expansion: '{token.text}' -> '{word}' (similarity: {sim:.2f})")
            
            logger.info(f"Expanded query from '{query}' to {len(expanded)} terms")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return expanded
    
    def disambiguate_entity(self, entity_text: str, context: str) -> str:
        """
        Disambiguate an entity using context.
        Returns a more specific entity name or the original if disambiguation fails.
        
        Args:
            entity_text: Entity to disambiguate
            context: Surrounding context text
            
        Returns:
            Disambiguated entity name (or original if can't disambiguate)
        """
        if not self.config.enable_disambiguation:
            return entity_text
        
        try:
            doc = self.nlp(context)
            
            # Find the entity in the context
            entity_mentions = [
                ent for ent in doc.ents 
                if entity_text.lower() in ent.text.lower()
            ]
            
            if not entity_mentions:
                return entity_text
            
            # Get context around entity
            for ent in entity_mentions:
                # Look for descriptive phrases nearby
                sent = ent.sent
                
                # Check for appositions (e.g., "Jensen Huang, CEO of Nvidia")
                for token in sent:
                    if token.dep_ == 'appos' and ent.start <= token.head.i < ent.end:
                        # Found an apposition - this adds context
                        subtree_tokens = list(token.subtree)
                        appos_start = min(t.i for t in subtree_tokens)
                        appos_end = max(t.i for t in subtree_tokens) + 1
                        appos_text = sent[appos_start:appos_end].text
                        disambiguated = f"{entity_text} ({appos_text})"
                        logger.debug(f"Disambiguated: '{entity_text}' -> '{disambiguated}'")
                        return disambiguated
                
                # Check for descriptive compounds (e.g., "CEO Jensen Huang")
                for i in range(max(0, ent.start - 3), ent.start):
                    if sent[i].pos_ in {'NOUN', 'PROPN'}:
                        descriptor = sent[i].text
                        if len(descriptor) > 2:
                            disambiguated = f"{descriptor} {entity_text}"
                            logger.debug(f"Disambiguated: '{entity_text}' -> '{disambiguated}'")
                            return disambiguated
        except Exception as e:
            logger.warning(f"Entity disambiguation failed for '{entity_text}': {e}")
        
        return entity_text
    
    def extract_temporal_info(self, text: str) -> List[Dict[str, str]]:
        """
        Extract temporal information from text (dates, time expressions).
        
        Args:
            text: Text to extract temporal info from
            
        Returns:
            List of temporal expressions with type and value
        """
        if not self.config.enable_temporal_extraction:
            return []
        
        temporal_info = []
        
        try:
            doc = self.nlp(text)
            
            # Extract DATE entities
            for ent in doc.ents:
                if ent.label_ == 'DATE':
                    temporal_info.append({
                        'type': 'date',
                        'value': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Look for temporal keywords
            temporal_keywords = {
                'before': 'before',
                'after': 'after',
                'during': 'during',
                'since': 'since',
                'until': 'until',
                'while': 'during',
                'when': 'at',
                'founded': 'founded',
                'established': 'established',
                'created': 'created',
                'born': 'born',
                'died': 'died'
            }
            
            for token in doc:
                if token.text.lower() in temporal_keywords:
                    temporal_info.append({
                        'type': 'temporal_keyword',
                        'value': temporal_keywords[token.text.lower()],
                        'start': token.idx,
                        'end': token.idx + len(token.text)
                    })
            
            logger.debug(f"Extracted {len(temporal_info)} temporal expressions")
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
        
        return temporal_info

    def is_content_relevant(self, text: str, query: str, min_relevance_score: float = 0.1, site: str | None = None) -> tuple[bool, float, str]:
        """
        Determine relevance using a weighted model including semantic similarity.

        This implementation uses spaCy vectors (when available) as the primary
        signal (60% weight) and exact full-query presence as a strong secondary
        signal (40% weight). The method keeps the `site` parameter and will
        use the translator when enabled and pass the site for cache consistency.
        """
        if not text or not query:
            return False, 0.0, text

        try:
            # Step 1: Translate the text if needed (using existing logic)
            processed_text = text
            if self.translator and self.config.enable_translation:
                try:
                    translated = self.translator.translate_if_needed(text, site=site)
                    if translated and translated != text:
                        processed_text = translated
                except Exception as e:
                    logger.warning(f"Translation in relevance check failed: {e}")

            # Step 2: Process query and content with spaCy
            # Truncate large content for efficiency
            doc_query = self.nlp(query)
            doc_content = self.nlp(processed_text[:5000])

            # Score A: Semantic Similarity (60% weight)
            semantic_score = 0.0
            try:
                if getattr(doc_query, 'vector', None) is not None and getattr(doc_content, 'vector', None) is not None:
                    if doc_query.has_vector and doc_content.has_vector:
                        similarity = doc_query.similarity(doc_content)
                        semantic_score = max(0.0, float(similarity))
                else:
                    logger.debug("Word vectors not available for semantic similarity")
            except Exception as e:
                logger.debug(f"Semantic similarity calculation failed: {e}")

            # Score B: Exact Full Query Match (40% weight)
            exact_match_score = 1.0 if query.lower() in processed_text.lower() else 0.0

            # Final weighted score
            final_score = (semantic_score * 0.6) + (exact_match_score * 0.4)
            is_relevant = final_score >= min_relevance_score

            logger.debug(f"Relevance check for '{query}': Semantic={semantic_score:.2f}, ExactMatch={exact_match_score:.2f} -> FinalScore={final_score:.2f} -> Relevant={is_relevant}")

            return is_relevant, final_score, processed_text

        except Exception as e:
            logger.warning(f"Error checking content relevance: {e}")
            return False, 0.0, text
    
    def _extract_entities(self, doc) -> Dict[str, str]:
        """
        Extract named entities from a spaCy document with confidence filtering.
        Now includes normalization BEFORE entity linking to prevent duplicates.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary mapping normalized entity text to entity label
        """
        from processor.node_cleaner import clean_node_name, is_valid_entity_text
        
        entities = {}
        for ent in doc.ents:
            entity_text = ent.text.strip()
            # === NEW: Pre-Validation on Original Text Capitalization ===
            # If a multi-word entity has no capitalized letters, discard it immediately.
            # This is a language-agnostic heuristic that filters fragments like "of the company".
            if ' ' in entity_text and not any(c.isupper() for c in entity_text):
                logger.debug(f"Discarding non-capitalized multi-word fragment: '{entity_text}'")
                continue
            # ==========================================================
            # NEW: Validate entity text to filter out sentence-like phrases/questions
            try:
                # Use the centralized, language-agnostic validator from node_cleaner
                if not is_valid_entity_text(entity_text, self.nlp):
                    logger.debug(f"Discarding invalid entity-like text: '{entity_text}'")
                    continue
            except Exception as e:
                logger.debug(f"Entity validation error for '{entity_text}': {e}")
            if entity_text and len(entity_text) >= self.config.min_entity_length:
                # Step 1: Clean basic special characters and whitespace
                cleaned_entity = self._clean_entity_text(entity_text)
                
                # Step 2: NORMALIZE (removes parentheticals, possessives, "The", etc.)
                # This must happen BEFORE entity linking to ensure variants map to same entity
                normalized_entity = clean_node_name(cleaned_entity)
                
                if normalized_entity and len(normalized_entity) >= self.config.min_entity_length:
                    # Calculate confidence score for this entity
                    confidence = self._calculate_entity_confidence(ent, doc)
                    
                    # Filter out low-confidence entities (threshold: 0.5)
                    if confidence >= 0.5:
                        entities[normalized_entity] = ent.label_
                        
                        # Log if normalization changed the name
                        if normalized_entity != entity_text:
                            logger.debug(f"Normalized entity: '{entity_text}' → '{normalized_entity}' ({ent.label_})")
                        else:
                            logger.debug(f"Entity '{normalized_entity}' ({ent.label_}) - confidence: {confidence:.2f}")
                    else:
                        logger.debug(f"Filtered low-confidence entity '{cleaned_entity}' ({ent.label_}) - confidence: {confidence:.2f}")
        
        # Deduplicate entities: keep full names, remove partial names
        if self.config.deduplicate_entities:
            entities = self._deduplicate_entities(entities)
            # Also merge reordered names (e.g., "Name1 Name2" with "Name2 Name1")
            entities = self._merge_reordered_names(entities)
        
        return entities

    def _is_valid_entity_text(self, text: str) -> bool:
        """
        Performs sanity checks to filter out sentence-like phrases misidentified as entities.
        Returns False if the entity should be discarded, True otherwise.
        """
        """
        Performs multi-stage sanity checks to filter out sentence-like phrases
        misidentified as entities. This is the main gatekeeper for entity quality.
        """
        if not text or not text.strip():
            return False

        text_lower = text.lower()

        # Rule 1: Filter out phrases that start with common question or stop words.
        # This catches things like "Who is..." or "For the...".
        invalid_start_words = ('who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'for', 'the', 'an', 'a', 'of', 'in', 'on', 'at')
        if text_lower.startswith(invalid_start_words):
            logger.debug(f"Filtering entity starting with invalid word: '{text}'")
            return False

        # Rule 2: Filter out entities that are excessively long (likely full sentences).
        # A real entity name is rarely more than 8 words.
        if len(text.split()) > 8:
            logger.debug(f"Filtering overly long entity phrase: '{text}'")
            return False

        # Rule 3: Filter out entities containing obvious sentence markers or junk.
        if '→' in text or '...' in text or '|' in text:
            logger.debug(f"Filtering entity with sentence-like punctuation: '{text}'")
            return False

        # Rule 4: The most powerful check - filter out entities containing verbs.
        # Proper nouns should not contain verbs (e.g., "The company that *is* growing").
        try:
            doc = self.nlp(text)
            # Rule 5: Check the Part-of-Speech of the last token in the entity.
            # A proper name should not end in a conjunction, preposition, pronoun, auxiliary or punctuation.
            if doc and len(doc) > 0:
                last_token = doc[-1]
                if last_token.pos_ in ['PUNCT', 'ADP', 'CCONJ', 'SCONJ', 'PRON', 'AUX']:
                    logger.debug(f"Filtering entity ending with invalid part-of-speech ('{last_token.text}' is {last_token.pos_}): '{text}'")
                    return False

            # Also filter entities containing verbs anywhere - strong indicator of boundary error
            if any(token.pos_ == 'VERB' for token in doc):
                logger.debug(f"Filtering entity containing a verb: '{text}'")
                return False
        except Exception as e:
            logger.debug(f"POS-check failed for '{text}': {e}")

        return True
    
    def _calculate_entity_confidence(self, ent, doc) -> float:
        """
        Calculate confidence score for an entity based on multiple factors.
        
        Scoring factors:
        - Length: Longer entities (2+ words) are more reliable (0-0.4 points)
        - Context: Proper linguistic context (capitalized, in sentence) (0-0.3 points)
        - Entity type: Some types are more reliable than others (0-0.3 points)
        
        Args:
            ent: spaCy entity
            doc: spaCy document containing the entity
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # 1. Length score (0-0.4): Multi-word entities are more reliable
        words = ent.text.split()
        if len(words) >= 3:
            score += 0.4  # Very confident for 3+ words
        elif len(words) == 2:
            score += 0.3  # Confident for 2 words
        elif len(words) == 1:
            if len(ent.text) >= 4:
                score += 0.2  # Single word but reasonable length
            else:
                score += 0.1  # Very short single word (less confident)
        
        # 2. Context score (0-0.3): Check capitalization and sentence structure
        try:
            # Check if entity is properly capitalized (not all caps or all lowercase)
            if ent.text[0].isupper() and not ent.text.isupper():
                score += 0.15
            
            # Check if entity appears at sentence start (slightly less reliable)
            sent_start = False
            for token in ent:
                if token.is_sent_start:
                    sent_start = True
                    break
            
            if not sent_start:
                score += 0.15  # Not at sentence start = more reliable
            else:
                score += 0.05  # At sentence start = could be common noun
                
        except Exception as e:
            logger.debug(f"Context scoring error for '{ent.text}': {e}")
        
        # 3. Entity type score (0-0.3): Some types are more reliable
        reliable_types = {
            'PERSON': 0.3,      # Names are usually reliable
            'ORG': 0.3,         # Organizations are usually reliable
            'GPE': 0.25,        # Geo-political entities are reliable
            'LOC': 0.25,        # Locations are reliable
            'DATE': 0.2,        # Dates are somewhat reliable
            'EVENT': 0.2,       # Events are somewhat reliable
            'PRODUCT': 0.15,    # Products are less reliable
            'WORK_OF_ART': 0.15, # Works of art are less reliable
            'FAC': 0.2,         # Facilities are somewhat reliable
            'NORP': 0.15,       # Nationalities/religions are less reliable
            'MONEY': 0.1,       # Money is less important
            'QUANTITY': 0.1,    # Quantities are less important
            'ORDINAL': 0.05,    # Ordinals are least important
            'CARDINAL': 0.05,   # Cardinals are least important
        }
        
        score += reliable_types.get(ent.label_, 0.1)  # Default 0.1 for unknown types
        
        # Ensure score is in valid range
        return min(max(score, 0.0), 1.0)
    
    def _clean_entity_text(self, text: str) -> str:
        """
        Clean entity text by removing special characters and excessive whitespace.
        
        This method:
        - Removes special characters at the start and end
        - Removes excessive whitespace
        - Strips leading/trailing punctuation
        - Strips trailing bracketed artifacts like [edit]
        
        NOTE: Language-specific normalization (diacritics, etc.) is handled by
        the translation system, so we keep entities in their natural form here.
        
        Args:
            text: Original entity text
            
        Returns:
            Cleaned entity text
        """
        if not text:
            return text
        
        # Strip whitespace
        text = text.strip()
        
        # NEW: Remove trailing bracketed artifacts like [edit], [citation needed], etc.
        # This regex removes patterns like "[word" or "[multiple words" at the end
        text = re.sub(r'\[\w+.*?$', '', text).strip()
        
        # Remove special characters at the start (but keep letters with diacritics)
        # This preserves names in their original language
        text = re.sub(r'^[^\w\s]+', '', text, flags=re.UNICODE)
        
        # Remove special characters at the end
        text = re.sub(r'[^\w\s]+$', '', text, flags=re.UNICODE)
        
        # Normalize internal whitespace (replace multiple spaces with single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Final strip
        text = text.strip()
        
        return text
    
    def _deduplicate_entities(self, entities: Dict[str, str]) -> Dict[str, str]:
        """
        Remove partial entity names and merge variations when full names exist.
        
        Merging rules:
        - Removes partial names: "Robert" when "Robert Negoita" exists
        - Merges "the X" with "X": "the Parliament" → "Parliament"
        - Merges "X's" with "X": "Romania's" → "Romania"
        
        Args:
            entities: Dictionary of entities
            
        Returns:
            Deduplicated entities dictionary
        """
        # First pass: build canonical form mapping
        canonical_map = {}  # Maps variations to canonical form
        entity_list = list(entities.keys())
        
        for entity in entity_list:
            entity_lower = entity.lower().strip()
            
            # Check for "the" prefix
            if entity_lower.startswith('the '):
                # "the Parliament" → "Parliament"
                base_form = entity[4:].strip()  # Remove "the "
                
                # Check if base form exists in entities
                for other_entity in entity_list:
                    if other_entity.lower().strip() == base_form.lower():
                        canonical_map[entity] = other_entity
                        logger.debug(f"Merging '{entity}' → '{other_entity}' (removing 'the')")
                        break
                else:
                    # Base form doesn't exist, use it as canonical
                    canonical_map[entity] = base_form
                    logger.debug(f"Converting '{entity}' → '{base_form}' (removing 'the')")
            
            # Check for "'s" suffix (possessive)
            elif entity_lower.endswith("'s") or entity_lower.endswith("'s"):  # Regular and curly apostrophe
                # "Romania's" → "Romania"
                if entity.endswith("'s"):
                    base_form = entity[:-2].strip()
                else:  # Curly apostrophe
                    base_form = entity[:-2].strip()
                
                # Check if base form exists in entities
                for other_entity in entity_list:
                    if other_entity.lower().strip() == base_form.lower():
                        canonical_map[entity] = other_entity
                        logger.debug(f"Merging '{entity}' → '{other_entity}' (removing 's)")
                        break
                else:
                    # Base form doesn't exist, use it as canonical
                    canonical_map[entity] = base_form
                    logger.debug(f"Converting '{entity}' → '{base_form}' (removing 's)")
            
            # Check for "s'" suffix (possessive plural)
            elif entity_lower.endswith("s'") or entity_lower.endswith("s'"):
                # "Roberts'" → "Roberts"
                base_form = entity[:-1].strip()
                
                # Check if base form exists in entities
                for other_entity in entity_list:
                    if other_entity.lower().strip() == base_form.lower():
                        canonical_map[entity] = other_entity
                        logger.debug(f"Merging '{entity}' → '{other_entity}' (removing possessive)")
                        break
                else:
                    canonical_map[entity] = base_form
                    logger.debug(f"Converting '{entity}' → '{base_form}' (removing possessive)")
        
        # Second pass: remove partial entities
        deduplicated = {}
        
        for entity in entity_list:
            # Skip if this entity has a canonical form (it's a variation)
            if entity in canonical_map:
                # Use the canonical form instead
                canonical = canonical_map[entity]
                if canonical not in deduplicated:
                    # Use the label from the original entity
                    deduplicated[canonical] = entities[entity]
                continue
            
            # Check if this entity is a substring of any other entity
            is_partial = False
            entity_lower = entity.lower()
            
            for other_entity in entity_list:
                if entity == other_entity:
                    continue
                
                # Skip if other_entity is a variation that will be merged
                if other_entity in canonical_map:
                    continue
                    
                other_lower = other_entity.lower()
                
                # Check if entity is a word within other_entity
                # E.g., "Robert" is in "Robert Negoita"
                if entity_lower in other_lower.split():
                    # Entity is a partial name, skip it
                    is_partial = True
                    logger.debug(f"Removing partial entity '{entity}' (part of '{other_entity}')")
                    break
            
            if not is_partial:
                deduplicated[entity] = entities[entity]
        
        # Third pass: merge any remaining "the X" and "X's" variations
        # This handles cases where both forms were in the original list
        final_deduplicated = {}
        processed = set()
        
        for entity, label in deduplicated.items():
            if entity in processed:
                continue
            
            entity_lower = entity.lower().strip()
            
            # Check if there's a "the X" version of this entity
            the_version = f"the {entity}"
            if the_version in deduplicated and the_version not in processed:
                # Merge "the X" into "X"
                logger.debug(f"Merging 'the {entity}' into '{entity}'")
                processed.add(the_version)
            
            # Check if there's an "X's" version of this entity
            possessive_versions = [f"{entity}'s", f"{entity}'s", f"{entity}s'", f"{entity}s'"]
            for poss_version in possessive_versions:
                if poss_version in deduplicated and poss_version not in processed:
                    logger.debug(f"Merging '{poss_version}' into '{entity}'")
                    processed.add(poss_version)
            
            final_deduplicated[entity] = label
            processed.add(entity)
        
        return final_deduplicated
    
    def _merge_reordered_names(self, entities: Dict[str, str]) -> Dict[str, str]:
        """
        Merge entities that are the same name but with words in different order.
        Examples:
        - "Robert Negoita" and "Negoita Robert" → keep "Robert Negoita"
        - "John Paul Smith" and "Smith John Paul" → keep "John Paul Smith"
        
        Strategy:
        - Split each name into words
        - Sort the words to create a canonical key
        - Group entities with same sorted words
        - Keep the most common ordering (or first occurrence)
        
        Args:
            entities: Dictionary of entities
            
        Returns:
            Entities with reordered duplicates merged
        """
        from collections import Counter
        
        # Group entities by their sorted word components
        word_groups = {}  # sorted_words_tuple -> [entity_names]
        
        for entity in entities.keys():
            # Split into words and normalize
            words = entity.lower().strip().split()
            
            # Skip single-word entities (nothing to reorder)
            if len(words) < 2:
                continue
            
            # Create canonical key by sorting words
            sorted_words = tuple(sorted(words))
            
            if sorted_words not in word_groups:
                word_groups[sorted_words] = []
            word_groups[sorted_words].append(entity)
        
        # Build merge mapping
        merge_map = {}  # entity_to_remove -> canonical_entity
        
        for sorted_words, entity_group in word_groups.items():
            if len(entity_group) <= 1:
                # No duplicates for this word combination
                continue
            
            # We have multiple orderings of the same name
            logger.info(f"Found reordered name variants: {entity_group}")
            
            # Choose canonical form - prefer the one that appears first in natural order
            # (i.e., if one starts with a capital letter and follows name convention)
            canonical = None
            
            # Strategy 1: Prefer "FirstName LastName" over "LastName FirstName"
            # Look for the one where first word is capitalized and likely a first name
            for entity in entity_group:
                words = entity.split()
                # Prefer shorter first word (usually first names are shorter)
                # and natural ordering
                if canonical is None:
                    canonical = entity
                else:
                    # Keep the one that looks more like natural name order
                    # (this is a heuristic - could be improved)
                    if len(words[0]) < len(canonical.split()[0]):
                        canonical = entity
            
            # Map all other variants to the canonical form
            for entity in entity_group:
                if entity != canonical:
                    merge_map[entity] = canonical
                    logger.debug(f"Merging '{entity}' → '{canonical}' (reordered name)")
        
        # Apply the merge mapping
        merged_entities = {}
        
        for entity, label in entities.items():
            if entity in merge_map:
                # This entity should be merged into its canonical form
                canonical = merge_map[entity]
                if canonical not in merged_entities:
                    # Use the label from the first occurrence
                    merged_entities[canonical] = label
                # Skip this entity (it's been merged)
            else:
                # Keep this entity
                merged_entities[entity] = label
        
        return merged_entities
    
    def _remap_relations(
        self, 
        relations: Mapping[Tuple[str, str], Sequence[Any]], 
        deduplicated_entities: Dict[str, str]
    ) -> Dict[Tuple[str, str], List[Any]]:
        """
        Remap relations to use deduplicated entity names.
        
        This handles cases where relations reference entity names that were merged
        during deduplication or reordering. Accepts values that can be lists of
        strings or Provenance objects (or other reason types) by using covariant
        Mapping/Sequence[Any] typing.
        
        Args:
            relations: Original relations with potentially duplicate entity names
            deduplicated_entities: Final deduplicated entities
            
        Returns:
            Relations with remapped entity names (values are lists of reason objects)
        """
        # Build a mapping from all possible entity variations to canonical names
        entity_map = {}
        
        # Add exact matches
        for entity in deduplicated_entities.keys():
            entity_map[entity.lower().strip()] = entity
        
        # Also create mappings for reordered versions
        for entity in deduplicated_entities.keys():
            words = entity.lower().strip().split()
            if len(words) >= 2:
                # Create all possible orderings (for 2-word names, just reverse)
                if len(words) == 2:
                    reversed_name = f"{words[1]} {words[0]}"
                    entity_map[reversed_name] = entity
        
        # Remap relations
        remapped_relations = defaultdict(list)
        
        for (source, target), reasons in relations.items():
            # Clean and find canonical forms
            source_lower = self._clean_entity_text(source).lower().strip()
            target_lower = self._clean_entity_text(target).lower().strip()
            
            # Try to find canonical entity names
            canonical_source = None
            canonical_target = None
            
            # Try exact match first
            if source_lower in entity_map:
                canonical_source = entity_map[source_lower]
            else:
                # Try to find by checking sorted words
                source_words = tuple(sorted(source_lower.split()))
                for entity in deduplicated_entities.keys():
                    entity_words = tuple(sorted(entity.lower().split()))
                    if source_words == entity_words:
                        canonical_source = entity
                        break
            
            if target_lower in entity_map:
                canonical_target = entity_map[target_lower]
            else:
                # Try to find by checking sorted words
                target_words = tuple(sorted(target_lower.split()))
                for entity in deduplicated_entities.keys():
                    entity_words = tuple(sorted(entity.lower().split()))
                    if target_words == entity_words:
                        canonical_target = entity
                        break
            
            # Only keep relations where both entities are in deduplicated set
            if canonical_source and canonical_target:
                if canonical_source != canonical_target:  # Avoid self-loops
                    # Merge reasons (preserve original reason objects)
                    for reason in reasons:
                        if reason not in remapped_relations[(canonical_source, canonical_target)]:
                            remapped_relations[(canonical_source, canonical_target)].append(reason)
            else:
                # Log entities that couldn't be mapped
                if not canonical_source:
                    logger.debug(f"Could not remap source entity: {source}")
                if not canonical_target:
                    logger.debug(f"Could not remap target entity: {target}")
        
        return dict(remapped_relations)
    
    def _extract_relations(self, doc, source_url: str | None = None) -> Dict[Tuple[str, str], List[Provenance]]:
        """
        Extract relationships between entities in the document using improved logic.
        Uses dependency parsing and proximity checks to avoid false connections.
        Includes confidence scoring if enabled.
        
        Args:
            doc: spaCy document
            source_url: Optional URL where the text came from
            
        Returns:
            Dictionary mapping entity pairs to list of Provenance objects
        """
        relations = defaultdict(list)
        
        for sent in doc.sents:
            sent_ents = [
                ent for ent in sent.ents 
                if ent.text.strip() and len(ent.text.strip()) >= self.config.min_entity_length
            ]
            
            if len(sent_ents) < 2:
                continue
            
            # Only create relations between entities that are actually connected
            for i in range(len(sent_ents)):
                for j in range(i + 1, len(sent_ents)):
                    ent1, ent2 = sent_ents[i], sent_ents[j]
                    
                    ent1_text = ent1.text.strip()
                    ent2_text = ent2.text.strip()
                    
                    # Check if entities should be related
                    if not self._should_relate_entities(ent1, ent2, sent):
                        logger.debug(f"Skipping weak relation: '{ent1_text}' <-> '{ent2_text}'")
                        continue
                    
                    # Classify relation type if classifier is available
                    relation_type = None
                    if self.relation_classifier:
                        relation_type = self.relation_classifier.classify_relation(ent1, ent2, sent)
                        if relation_type:
                            logger.debug(f"Classified: '{ent1_text}' --[{relation_type}]--> '{ent2_text}'")
                    
                    # Calculate confidence if enabled
                    confidence = None
                    if self.config.enable_confidence_scoring:
                        confidence = self._calculate_relation_confidence(ent1, ent2, sent, source_url)
                        
                        # Filter by minimum confidence
                        if confidence < self.config.min_confidence:
                            logger.debug(f"Skipping low-confidence relation ({confidence:.2f}): '{ent1_text}' <-> '{ent2_text}'")
                            continue
                    
                    # Extract the full sentence as context
                    sentence_text = sent.text.strip()

                    # Clean the sentence to remove noisy citation markers
                    cleaned = self._clean_reason(sentence_text)

                    # Validate source URL - only use if it's a proper HTTP/HTTPS URL
                    validated_url = None
                    if source_url and source_url.strip():
                        url_lower = source_url.strip().lower()
                        if url_lower.startswith('http://') or url_lower.startswith('https://'):
                            validated_url = source_url.strip()
                        else:
                            logger.debug(f"Invalid source URL format (not HTTP/HTTPS): {source_url}")
                    
                    if not validated_url:
                        logger.warning(f"No valid source URL provided for relation: '{ent1_text}' <-> '{ent2_text}' - skipping this relation")
                        continue  # Skip relations without valid URLs
                    
                    # If the cleaner returned multiple fragments, add them separately
                    if isinstance(cleaned, list):
                        for frag in cleaned:
                            # Create structured Provenance object with validated URL
                            prov = Provenance(
                                text=frag,
                                source_url=validated_url,
                                confidence=confidence,
                                dates=None,  # Dates can be extracted later if needed
                                sentence_offset=sent.start_char,
                                relation_type=relation_type
                            )
                            relations[(ent1_text, ent2_text)].append(prov)
                    else:
                        # Create structured Provenance object with validated URL
                        prov = Provenance(
                            text=cleaned,
                            source_url=validated_url,
                            confidence=confidence,
                            dates=None,  # Dates can be extracted later if needed
                            sentence_offset=sent.start_char,
                            relation_type=relation_type
                        )
                        relations[(ent1_text, ent2_text)].append(prov)
        
        return dict(relations)
    
    def _should_relate_entities(self, ent1, ent2, sent) -> bool:
        """
        Determine if two entities should be related based on syntactic and semantic analysis.
        
        Args:
            ent1: First entity
            ent2: Second entity
            sent: Sentence containing both entities
            
        Returns:
            True if entities should be related, False otherwise
        """
        # Get token positions
        ent1_start, ent1_end = ent1.start, ent1.end
        ent2_start, ent2_end = ent2.start, ent2.end
        
        # 1. Proximity check: entities should be reasonably close (max 15 tokens apart)
        token_distance = abs(ent1_start - ent2_start)
        if token_distance > 15:
            logger.debug(f"Entities too far apart: {token_distance} tokens")
            return False
        
        # 2. Check if there's a verb or preposition between them (indicates relationship)
        tokens_between = list(sent[min(ent1_end, ent2_end):max(ent1_start, ent2_start)])
        has_connector = any(token.pos_ in {'VERB', 'AUX', 'ADP'} for token in tokens_between)
        
        # 3. Check dependency path between entities
        # Get root tokens of each entity
        ent1_tokens = [token for token in sent if ent1_start <= token.i < ent1_end]
        ent2_tokens = [token for token in sent if ent2_start <= token.i < ent2_end]
        
        if not ent1_tokens or not ent2_tokens:
            return False
        
        # Find head tokens (syntactic heads of the entities)
        ent1_head = ent1_tokens[0].head if ent1_tokens else ent1_tokens[0]
        ent2_head = ent2_tokens[0].head if ent2_tokens else ent2_tokens[0]
        
        # Check if entities share a common ancestor in dependency tree
        ent1_ancestors = set([ent1_head] + list(ent1_head.ancestors))
        ent2_ancestors = set([ent2_head] + list(ent2_head.ancestors))
        common_ancestors = ent1_ancestors & ent2_ancestors
        
        # 4. Check dependency relations that indicate meaningful connection
        meaningful_deps = {
            'nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj',  # Subject/object relations
            'compound', 'appos', 'amod',  # Modification relations
            'prep', 'agent', 'poss',  # Prepositional/possessive relations
            'conj', 'cc'  # Coordination
        }
        
        has_meaningful_dep = False
        for token in ent1_tokens + ent2_tokens:
            if token.dep_ in meaningful_deps:
                has_meaningful_dep = True
                break
        
        # 5. Type-based filtering: certain entity type combinations are more likely to be meaningful
        ent1_label = ent1.label_
        ent2_label = ent2.label_
        
        # Strong type combinations (always relate if close)
        strong_pairs = {
            ('PERSON', 'ORG'), ('ORG', 'PERSON'),
            ('PERSON', 'GPE'), ('GPE', 'PERSON'),
            ('PERSON', 'NORP'), ('NORP', 'PERSON'),
            ('ORG', 'GPE'), ('GPE', 'ORG'),
            ('PERSON', 'EVENT'), ('EVENT', 'PERSON'),
            ('ORG', 'EVENT'), ('EVENT', 'ORG'),
        }
        
        # Weak type combinations (need stronger evidence)
        weak_pairs = {
            ('ORG', 'ORG'),  # Two organizations mentioned together doesn't always mean relation
            ('GPE', 'GPE'),  # Two locations mentioned together
            ('NORP', 'NORP'),
        }
        
        is_strong_pair = (ent1_label, ent2_label) in strong_pairs
        is_weak_pair = (ent1_label, ent2_label) in weak_pairs
        
        # Decision logic
        if is_strong_pair:
            # Strong pairs need either connector or common ancestor
            if has_connector or common_ancestors:
                return True
        elif is_weak_pair:
            # Weak pairs need both connector AND meaningful dependency
            if has_connector and has_meaningful_dep and token_distance <= 8:
                return True
        else:
            # Other combinations need connector or meaningful dependency
            if has_connector or has_meaningful_dep:
                return True
        
        # 6. Additional check: if entities are in a list structure, they might not be related
        # (e.g., "John, Mary, and Bob went to the store" - John and Bob aren't directly related)
        tokens_between_text = ' '.join([t.text for t in tokens_between]).lower()
        if any(punct in tokens_between_text for punct in [',', ';', 'and', 'or']):
            # If they're in a list and far apart, probably not related
            if token_distance > 5:
                return False
        
        return False
    
    def _calculate_relation_confidence(self, ent1, ent2, sent, source_url: Optional[str] = None) -> float:
        """
        Calculate confidence score for a relationship (0.0 to 1.0).
        Higher score = more confident the relationship is meaningful.
        
        Args:
            ent1: First entity
            ent2: Second entity
            sent: Sentence containing both entities
            source_url: Optional source URL (more credible sources = higher confidence)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # 1. Distance factor (closer = more confident)
        token_distance = abs(ent1.start - ent2.start)
        if token_distance <= 3:
            confidence += 0.2
        elif token_distance <= 6:
            confidence += 0.1
        elif token_distance <= 10:
            confidence += 0.05
        else:
            confidence -= 0.1  # Penalize distant entities
        
        # 2. Entity type combination factor
        ent1_label = ent1.label_
        ent2_label = ent2.label_
        strong_pairs = {
            ('PERSON', 'ORG'), ('ORG', 'PERSON'),
            ('PERSON', 'GPE'), ('GPE', 'PERSON'),
            ('ORG', 'GPE'), ('GPE', 'ORG'),
        }
        if (ent1_label, ent2_label) in strong_pairs:
            confidence += 0.15
        
        # 3. Dependency pattern strength
        ent1_tokens = [token for token in sent if ent1.start <= token.i < ent1.end]
        ent2_tokens = [token for token in sent if ent2.start <= token.i < ent2.end]
        
        if ent1_tokens and ent2_tokens:
            # Check for strong dependency relations
            strong_deps = {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'appos', 'compound'}
            for token in ent1_tokens + ent2_tokens:
                if token.dep_ in strong_deps:
                    confidence += 0.1
                    break
        
        # 4. Connector presence (verb/preposition between entities)
        tokens_between = list(sent[min(ent1.end, ent2.end):max(ent1.start, ent2.start)])
        has_verb = any(token.pos_ == 'VERB' for token in tokens_between)
        has_prep = any(token.pos_ == 'ADP' for token in tokens_between)
        
        if has_verb:
            confidence += 0.15
        if has_prep:
            confidence += 0.1
        
        # 5. Source reliability factor
        if source_url:
            # Wikipedia and credible sources get a boost
            if 'wikipedia.org' in source_url:
                confidence += 0.1
            elif any(domain in source_url for domain in ['.edu', '.gov', '.org']):
                confidence += 0.05
        
        # 6. Entity prominence (longer entities often more specific = more confident)
        avg_entity_length = (len(ent1.text) + len(ent2.text)) / 2
        if avg_entity_length > 15:
            confidence += 0.05
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _clean_reason(self, text: str):
        """
        Clean extracted sentence text to make reasons more readable.

        Behavior:
        - Remove ALL bracketed content like "[text]", "[8]", "[citation needed]", etc.
        - Remove leading bracketed markers like "[change | source modification]".
        - Remove common standalone footnote markers like "- ^", "^ a b", and similar.
        - Split concatenated citation fragments separated by patterns like " - ^ " into multiple cleaned reasons.
        - Normalize whitespace and fix spacing around punctuation.
        - Clean up table formatting to be more readable.
        - Remove excessive special characters and pipes.

        Returns either a cleaned string or a list of cleaned fragments.
        """
        if not text:
            return text

        # Remove ALL bracketed content [anything] - citations, notes, etc.
        # This removes [8], [text], [citation needed], [change], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Also remove parenthetical citations like (2020) or (citation needed)
        text = re.sub(r'\(\s*\d{4}\s*\)', '', text)  # Remove year citations like (2020)
        text = re.sub(r'\(\s*citation\s+needed\s*\)', '', text, flags=re.IGNORECASE)

        # If there are obvious concatenated citation separators like " - ^ ", split on them.
        parts = re.split(r"\s*-\s*\^\s*", text)
        cleaned_parts = []

        for part in parts:
            p = part.strip()
            if not p:
                continue

            # Remove leading footnote markers more carefully
            # Pattern 1: Remove numbered footnotes like "11. " or "(2) " at the start
            # But only if followed by whitespace to avoid removing first letters
            p = re.sub(r"^[\[\(]*\d+[\]\)]*[\.:,;\s-]+", "", p)
            
            # Pattern 2: Remove short letter sequences at start like "a b " or "^ a " 
            # Only if they are isolated and followed by space (not part of a word)
            p = re.sub(r"^[\[\(]*[a-zA-Z]\s+([a-zA-Z]\s+)?", "", p)

            # Remove stray caret markers
            p = re.sub(r"\^+", "", p)
            
            # Remove isolated single letters only if they appear between spaces (not at word boundaries)
            # This prevents removing letters from actual names
            p = re.sub(r"^\s*\b([a-f])\b\s+", "", p, flags=re.IGNORECASE)

            # Clean up table formatting - detect and convert to readable format
            p = self._clean_table_formatting(p)

            # Remove excessive pipe characters (used in tables) but preserve single pipes
            p = re.sub(r'\|{2,}', ' | ', p)  # Replace multiple pipes with single pipe
            p = re.sub(r'\s*\|\s*\|\s*', ' | ', p)  # Clean up double pipes with spaces
            
            # Remove excessive dashes/underscores (table separators)
            p = re.sub(r'[-_]{3,}', '', p)
            
            # Remove standalone pipes at start/end
            p = re.sub(r'^\s*\|\s*', '', p)
            p = re.sub(r'\s*\|\s*$', '', p)

            # Normalize spacing before punctuation and collapse multiple spaces
            p = re.sub(r"\s+([,.;:!?])", r"\1", p)
            p = re.sub(r"\s{2,}", " ", p)

            p = p.strip()
            if p and len(p) > 3:  # Filter out very short fragments
                cleaned_parts.append(p)

        if not cleaned_parts:
            # final fallback: normalize whitespace and return
            return re.sub(r"\s{2,}", " ", text).strip()

        # If only one piece, return string; otherwise return list of fragments
        return cleaned_parts if len(cleaned_parts) > 1 else cleaned_parts[0]

    def _clean_table_formatting(self, text: str) -> str:
        """
        Clean up table formatting in text to make it more readable.
        Converts markdown-style tables to a more compact, readable format.
        
        Args:
            text: Text potentially containing table formatting
            
        Returns:
            Cleaned text with better table formatting
        """
        # Check if this looks like a table (has multiple pipes and potential header separators)
        pipe_count = text.count('|')
        
        if pipe_count < 3:
            # Not a table, return as-is
            return text
        
        # Split by newlines to process table rows
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip separator lines (like |---|---|)
            if re.match(r'^[\|\s\-:]+$', line):
                continue
            
            # If line has pipes, clean it up
            if '|' in line:
                # Remove leading/trailing pipes
                line = re.sub(r'^\||\|$', '', line)
                
                # Split by pipe and clean each cell
                cells = [cell.strip() for cell in line.split('|')]
                
                # Filter out empty cells
                cells = [cell for cell in cells if cell]
                
                # Rejoin with separator
                if cells:
                    cleaned_lines.append(' • '.join(cells))
            else:
                cleaned_lines.append(line)
        
        # Join cleaned lines
        result = ' '.join(cleaned_lines)
        
        # Final cleanup
        result = re.sub(r'\s{2,}', ' ', result).strip()
        
        return result if result else text
    
    def get_top_entities_for_discovery(
        self,
        entities_dict: dict[str, str],
        relations_dict: dict[tuple[str, str], list[str]],
        max_entities: int = 10,
        original_query: str | None = None,
        relevance_threshold: float = 0.15
    ) -> list[str]:
        """
        Extract the most important entities for further discovery.
        
        Ranks entities by:
        1. Semantic relevance to original query (if provided)
        2. Number of connections (degree centrality)
        3. Entity type importance (PERSON, ORG, GPE prioritized)
        4. Name length (longer names often more specific)
        
        Args:
            entities_dict: Dictionary of entities {entity_text: entity_label}
            relations_dict: Dictionary of relations {(entity1, entity2): [reasons]}
            max_entities: Maximum number of entities to return
            original_query: Original user query for relevance filtering
            relevance_threshold: Minimum relevance score to include entity (0.15 = 15%)
            
        Returns:
            List of top entity names for discovery
        """
        from collections import Counter
        
        # Count how many connections each entity has
        entity_connections = Counter()
        
        for (source, target), reasons in relations_dict.items():
            if reasons:  # Only count if there are actual reasons
                entity_connections[source] += 1
                entity_connections[target] += 1
        
        # Score entities
        entity_scores = {}
        
        # Entity type importance weights
        type_weights = {
            'PERSON': 3.0,      # Highest priority - people
            'ORG': 2.5,         # Organizations
            'GPE': 2.0,         # Geo-political entities
            'EVENT': 1.8,       # Events
            'NORP': 1.5,        # Nationalities, religious/political groups
            'FAC': 1.2,         # Facilities
            'LOC': 1.0,         # Locations
        }
        
        # Prepare query tokens for relevance checking (if query provided)
        query_tokens = set()
        query_normalized = ""
        if original_query:
            query_normalized = self._normalize_romanian_text(original_query)
            query_tokens = set(query_normalized.lower().split())
            logger.info(f"Query relevance filter enabled with tokens: {query_tokens}")
        
        for entity, entity_type in entities_dict.items():
            # Calculate query relevance score (0.0 to 1.0)
            relevance_score = 0.0
            
            if original_query:
                entity_normalized = self._normalize_romanian_text(entity)
                entity_tokens = set(entity_normalized.lower().split())
                
                # Check for exact match (full entity in query or vice versa)
                if entity_normalized in query_normalized or query_normalized in entity_normalized:
                    relevance_score = 1.0
                # Check for high token overlap
                elif entity_tokens and query_tokens:
                    overlap = len(entity_tokens.intersection(query_tokens))
                    max_tokens = max(len(entity_tokens), len(query_tokens))
                    relevance_score = overlap / max_tokens if max_tokens > 0 else 0.0
            else:
                # If no query provided, consider all entities relevant
                relevance_score = 1.0
            
            # Filter out irrelevant entities
            if relevance_score < relevance_threshold:
                logger.debug(f"Filtered out '{entity}' (relevance: {relevance_score:.2f} < {relevance_threshold})")
                continue
            
            # Base score from connections
            connection_score = entity_connections.get(entity, 0)
            
            # Type weight
            type_weight = type_weights.get(entity_type, 0.5)
            
            # Length bonus (longer names are often more specific)
            length_bonus = min(len(entity.split()), 3) * 0.5  # Cap at 3 words
            
            # Total score (relevance is CRITICAL - multiply by 10 to dominate other factors)
            total_score = (relevance_score * 10.0) + (connection_score * type_weight) + length_bonus
            
            entity_scores[entity] = {
                'total': total_score,
                'relevance': relevance_score,
                'connections': connection_score,
                'type_weight': type_weight
            }
        
        # Sort by total score and return top entities
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        top_entities = [entity for entity, scores in sorted_entities[:max_entities]]
        
        logger.info(f"Selected {len(top_entities)} top entities for discovery (filtered from {len(entities_dict)} total):")
        for i, entity in enumerate(top_entities[:5], 1):
            scores = entity_scores[entity]
            connections = entity_connections.get(entity, 0)
            entity_type = entities_dict[entity]
            logger.info(
                f"  {i}. {entity} "
                f"(type: {entity_type}, relevance: {scores['relevance']:.2f}, "
                f"connections: {connections}, total_score: {scores['total']:.2f})"
            )
        
        if len(top_entities) > 5:
            logger.info(f"  ... and {len(top_entities) - 5} more")
        
        return top_entities
