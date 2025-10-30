"""
Iterative Search Orchestrator for autonomous deep-dive research.
Continuously discovers new keywords and entities from search results and graph data.
"""
from typing import List, Dict, Set, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import asyncio
from collections import Counter

from advanced_search import SearchQuery, AdvancedSearchBuilder
from logger import setup_logger

if TYPE_CHECKING:
    from crawler import WebCrawler

logger = setup_logger(__name__)

# Try to import spaCy for NLP-driven keyword extraction
try:
    import spacy
    NLP_AVAILABLE = True
    try:
        nlp_model = spacy.load("en_core_web_lg")
        logger.info("Loaded spaCy model for keyword extraction")
    except OSError:
        NLP_AVAILABLE = False
        logger.warning("spaCy model not found. Using simple keyword extraction. Install with: python -m spacy download en_core_web_lg")
except ImportError:
    NLP_AVAILABLE = False
    nlp_model = None
    logger.warning("spaCy not installed. Using simple keyword extraction.")


@dataclass
class SearchIteration:
    """Represents one iteration of search."""
    iteration_number: int
    queries: List[SearchQuery]
    discovered_urls: List[Tuple[str, str, str, float]]  # (url, title, source, score)
    new_keywords: Set[str]
    new_entities: Set[str]


class IterativeSearchOrchestrator:
    """
    Orchestrates autonomous multi-iteration search with self-reprompting.
    Learns from each iteration to refine subsequent searches.
    """
    
    def __init__(
        self,
        crawler: 'WebCrawler',
        max_iterations: int = 3,
        max_results_per_query: int = 10,
        min_relevance_score: float = 0.3
    ):
        """
        Initialize orchestrator.
        
        Args:
            crawler: WebCrawler instance with browser for searches
            max_iterations: Maximum number of search iterations
            max_results_per_query: Max results per search query
            min_relevance_score: Minimum relevance score for URLs
        """
        self.crawler = crawler
        self.max_iterations = max_iterations
        self.max_results_per_query = max_results_per_query
        self.min_relevance_score = min_relevance_score
        self.builder = AdvancedSearchBuilder()
        
        # Tracking
        self.iterations: List[SearchIteration] = []
        self.all_discovered_urls: Set[str] = set()
        self.all_keywords: Set[str] = set()
        self.all_entities: Set[str] = set()
    
    def extract_keywords_from_titles(
        self,
        titles: List[str],
        min_length: int = 4,
        top_n: int = 10
    ) -> Set[str]:
        """
        Extract keywords from search result titles using NLP-driven analysis.
        Focuses on noun chunks and key terms rather than simple word frequency.
        
        Args:
            titles: List of result titles
            min_length: Minimum keyword length
            top_n: Number of top keywords to return
            
        Returns:
            Set of extracted keywords
        """
        if NLP_AVAILABLE and nlp_model:
            return self._extract_keywords_nlp(titles, min_length, top_n)
        else:
            return self._extract_keywords_simple(titles, min_length, top_n)
    
    def _extract_keywords_nlp(
        self,
        titles: List[str],
        min_length: int,
        top_n: int
    ) -> Set[str]:
        """
        NLP-driven keyword extraction using spaCy.
        Extracts noun chunks, named entities, and important noun phrases.
        """
        keyword_counts = Counter()
        
        # Process all titles together for better context
        combined_text = " | ".join(titles)
        doc = nlp_model(combined_text) # type: ignore
        
        # Extract noun chunks (e.g., "annual report", "financial data")
        for chunk in doc.noun_chunks:
            text = chunk.text.lower().strip()
            if len(text) >= min_length and not self._is_stopword_phrase(text):
                keyword_counts[text] += 2  # Weight noun chunks higher
        
        # Extract named entities (ORG, PERSON, GPE, etc.)
        for ent in doc.ents:
            text = ent.text.lower().strip()
            if len(text) >= min_length and ent.label_ in {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT'}:
                keyword_counts[text] += 3  # Weight entities highest
        
        # Extract important nouns
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop:
                text = token.text.lower().strip()
                if len(text) >= min_length:
                    keyword_counts[text] += 1
        
        # Get top N keywords
        top_keywords = {word for word, count in keyword_counts.most_common(top_n)}
        
        logger.debug(f"NLP-extracted keywords: {top_keywords}")
        return top_keywords
    
    def _extract_keywords_simple(
        self,
        titles: List[str],
        min_length: int,
        top_n: int
    ) -> Set[str]:
        """
        Simple keyword extraction (fallback when spaCy not available).
        Uses word frequency with stopword filtering.
        """
        # Simple keyword extraction (original logic)
        words = []
        for title in titles:
            # Split on common delimiters
            parts = title.replace('|', ' ').replace('-', ' ').replace(':', ' ')
            words.extend(parts.split())
        
        # Filter and count
        word_counts = Counter()
        stopwords = {'the', 'and', 'for', 'with', 'from', 'about', 'this', 'that', 'what', 'when', 'where', 'how'}
        
        for word in words:
            clean_word = word.strip('.,!?()[]{}').lower()
            if len(clean_word) >= min_length and clean_word not in stopwords:
                word_counts[clean_word] += 1
        
        # Get top N
        top_keywords = {word for word, count in word_counts.most_common(top_n)}
        logger.debug(f"Simple-extracted keywords: {top_keywords}")
        
        return top_keywords
    
    def _is_stopword_phrase(self, phrase: str) -> bool:
        """Check if phrase is a common stopword phrase."""
        stopword_phrases = {
            'the report', 'the company', 'the new', 'the best', 'the latest',
            'this year', 'last year', 'next year', 'the world', 'the first'
        }
        return phrase in stopword_phrases
    
    def _filter_urls_by_relevance(
        self,
        urls: List[Tuple[str, str, str]],
        query: str
    ) -> List[Tuple[str, str, str, float]]:
        """
        Filter and score URLs by relevance to the query.
        
        Args:
            urls: List of (url, title, source) tuples
            query: Search query string
            
        Returns:
            List of (url, title, source, score) tuples above min_relevance_score
        """
        scored = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for url, title, source in urls:
            title_lower = title.lower()
            
            # Calculate relevance score based on keyword matches
            matches = sum(1 for word in query_words if word in title_lower)
            score = matches / max(len(query_words), 1)
            
            if score >= self.min_relevance_score:
                scored.append((url, title, source, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored
    
    async def execute_iteration(
        self,
        queries: List[SearchQuery],
        iteration_number: int,
        context: Optional[Dict[str, Any]] = None
    ) -> SearchIteration:
        """
        Execute one search iteration.
        
        Args:
            queries: Search queries to execute
            iteration_number: Current iteration number
            context: Optional context from previous iterations
            
        Returns:
            SearchIteration with results
        """
        logger.info(f"=== Iteration {iteration_number} ===")
        logger.info(f"Executing {len(queries)} queries...")
        
        all_results = []
        seen_urls = set()
        
        # Ensure crawler/browser is started before executing queries
        try:
            if not getattr(self.crawler, 'browser', None):
                logger.info("Crawler browser not started; attempting to start before executing queries...")
                await self.crawler.start()
        except Exception as e:
            logger.warning(f"Failed to start crawler/browser before executing queries: {e}")

        # Execute all queries
        for i, query in enumerate(queries, 1):
            query_str = query.build()
            logger.info(f"Query {i}/{len(queries)}: {query_str}")
            
            try:
                # Use the crawler's browser-based search
                results = await self.crawler.search_google(
                    query_str,
                    max_results=self.max_results_per_query
                )
                
                # Score and filter results
                scored = self._filter_urls_by_relevance(
                    results,
                    query.query
                )
                
                logger.info(f"  Found {len(scored)} relevant results")
                
                for url, title, source, score in scored:
                    if url not in seen_urls and url not in self.all_discovered_urls:
                        seen_urls.add(url)
                        all_results.append((url, title, source, score))
                
            except Exception as e:
                logger.warning(f"  Query failed: {e}")
                continue
        
        # Extract new keywords from titles
        titles = [title for _, title, _, _ in all_results]
        new_keywords = self.extract_keywords_from_titles(titles)
        
        # Filter out keywords we've already seen
        new_keywords = new_keywords - self.all_keywords
        self.all_keywords.update(new_keywords)
        
        # Update discovered URLs
        for url, _, _, _ in all_results:
            self.all_discovered_urls.add(url)
        
        # Create iteration record
        iteration = SearchIteration(
            iteration_number=iteration_number,
            queries=queries,
            discovered_urls=all_results,
            new_keywords=new_keywords,
            new_entities=set()  # Will be populated after NLP processing
        )
        
        self.iterations.append(iteration)
        
        logger.info(f"Iteration {iteration_number} complete:")
        logger.info(f"  - Discovered {len(all_results)} new URLs")
        logger.info(f"  - Found {len(new_keywords)} new keywords")
        
        return iteration
    
    def generate_followup_queries(
        self,
        base_entity: str,
        entity_type: str,
        new_keywords: Set[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchQuery]:
        """
        Generate follow-up queries based on discovered keywords.
        
        Args:
            base_entity: Main entity being researched
            entity_type: Entity type (PERSON, ORG, etc.)
            new_keywords: Newly discovered keywords
            context: Optional context from graph
            
        Returns:
            List of follow-up search queries
        """
        queries = []
        
        # Combine base entity with new keywords
        for keyword in list(new_keywords)[:5]:  # Top 5 keywords
            # Create targeted searches
            queries.append(SearchQuery(
                query=f"{base_entity} {keyword}",
                filetype="pdf"
            ))
            
            queries.append(SearchQuery(
                query=f"{base_entity} {keyword}",
                intitle=keyword
            ))
        
        # If context available, use related entities
        if context and 'related_entities' in context:
            for related_entity in context['related_entities'][:3]:
                queries.append(SearchQuery(
                    query=f"{base_entity} {related_entity}",
                    exact_phrases=[f"{base_entity} {related_entity}"]
                ))
        
        logger.info(f"Generated {len(queries)} follow-up queries from {len(new_keywords)} keywords")
        
        return queries

    # === NEW: CONTEXT-AWARE QUERY GENERATION METHOD ===
    def _generate_contextual_initial_queries(self, entity_name: str, entity_type: str, context_entities: List[str]) -> List[SearchQuery]:
        """Generates specific, context-rich initial queries for the deep dive."""
        queries: List[SearchQuery] = []

        # Query 1: The entity's relationship with the primary context entity.
        if context_entities:
            primary_context = context_entities[0]
            queries.append(SearchQuery(
                query=f'"{entity_name}" "{primary_context}" relationship',
                exact_phrases=[entity_name, primary_context]
            ))

        # Query 2: Official reports / analysis (PDFs often contain high-value info)
        queries.append(SearchQuery(
            query=f'"{entity_name}" official reports OR analysis',
            exact_phrases=[entity_name],
            filetype="pdf"
        ))

        # Query 3: Relation/influence queries with next-most-important neighbors
        for neighbor in context_entities[1:3]:
            queries.append(SearchQuery(
                query=f'"{entity_name}" influence on "{neighbor}"',
                exact_phrases=[entity_name, neighbor]
            ))

        logger.info(f"Generated {len(queries)} context-aware initial queries for '{entity_name}'.")
        return queries
    
    async def run_autonomous_search(
        self,
        entity_name: str,
        entity_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchIteration]:
        """
        Run autonomous multi-iteration search.
        
        Args:
            entity_name: Entity to research
            entity_type: Entity type (PERSON, ORG, etc.)
            context: Optional context from knowledge graph
            
        Returns:
            List of all search iterations
        """
        logger.info("=" * 70)
        logger.info(f"Starting Autonomous Search for {entity_type}: {entity_name}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info("=" * 70)
        
        # Iteration 1: Initial contextual search when possible
        context_entities = context.get('related_entities', []) if context else []

        if context_entities:
            initial_queries = self._generate_contextual_initial_queries(entity_name, entity_type, context_entities)
        else:
            # Fallback: Build simple queries when we don't have graph context
            initial_queries = [
                SearchQuery(query=f'"{entity_name}"'),
                SearchQuery(query=f'"{entity_name}" profile OR biography'),
                SearchQuery(query=f'"{entity_name}"', filetype="pdf")
            ]
            logger.info(f"Generated {len(initial_queries)} basic initial queries for '{entity_name}'.")

        iteration_1 = await self.execute_iteration(
            initial_queries,
            iteration_number=1,
            context=context
        )
        
        # Subsequent iterations: Refine based on discoveries
        for i in range(2, self.max_iterations + 1):
            # Check if we found enough URLs
            total_urls = len(self.all_discovered_urls)
            if total_urls > 50:  # Threshold
                logger.info(f"Discovered sufficient URLs ({total_urls}), stopping iterations")
                break
            
            # Generate follow-up queries based on previous iteration
            previous = self.iterations[-1]
            
            if not previous.new_keywords:
                logger.info("No new keywords discovered, stopping iterations")
                break
            
            followup_queries = self.generate_followup_queries(
                entity_name,
                entity_type,
                previous.new_keywords,
                context
            )
            
            if not followup_queries:
                logger.info("No follow-up queries generated, stopping iterations")
                break
            
            # Execute next iteration
            iteration = await self.execute_iteration(
                followup_queries,
                iteration_number=i,
                context=context
            )
        
        # Summary
        total_urls = len(self.all_discovered_urls)
        total_keywords = len(self.all_keywords)
        
        logger.info("=" * 70)
        logger.info("Autonomous Search Complete")
        logger.info(f"  - Total iterations: {len(self.iterations)}")
        logger.info(f"  - Total URLs discovered: {total_urls}")
        logger.info(f"  - Total keywords discovered: {total_keywords}")
        logger.info("=" * 70)
        
        return self.iterations
    
    def get_all_discovered_urls(self) -> List[Tuple[str, str, str, float]]:
        """Get all discovered URLs from all iterations."""
        all_urls = []
        for iteration in self.iterations:
            all_urls.extend(iteration.discovered_urls)
        return all_urls
