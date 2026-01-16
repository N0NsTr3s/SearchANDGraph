"""
Iterative Search Orchestrator for autonomous deep-dive research.
Continuously discovers new keywords and entities from search results and graph data.
"""
from typing import List, Dict, Set, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import asyncio
import os
from collections import Counter
import re
import sys
from scraper.advanced_search import SearchQuery, AdvancedSearchBuilder
from utils.logger import setup_logger

if TYPE_CHECKING:
    from scraper.crawler import WebCrawler

logger = setup_logger(__name__)

def load_spacy_model():
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)

    model_path = os.path.join(base_path, "en_core_web_lg")
    return spacy.load(model_path)
# Try to import spaCy for NLP-driven keyword extraction
try:
    import spacy
    NLP_AVAILABLE = True
    try:
        nlp_model = load_spacy_model()
        logger.info("Loaded spaCy model for keyword extraction")
    except:
        logger.warning("Loading spaCy model from installed packages")

    try:
        import en_core_web_lg
        nlp_model = en_core_web_lg.load()
        logger.info("Loaded spaCy model for keyword extraction")
    except:
        logger.warning("Loading spaCy model using spacy.load()")

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
        min_relevance_score: float = 0.1  # Relaxed threshold for OSINT dork queries
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
        # Concurrency control for parallel searches
        self.max_concurrent_searches = int(os.environ.get('SAG_MAX_CONCURRENT_SEARCHES', '3'))
        self._search_sem = asyncio.Semaphore(self.max_concurrent_searches)
        
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
        scored: List[Tuple[str, str, str, float]] = []

        # Normalize query terms: SearchQuery.query often includes quotes/operators.
        # Using raw .split() produces tokens like '"csm"' which will never match titles.
        query_lower = (query or "").lower()
        raw_tokens = re.findall(r"[\w\-]+", query_lower)
        stop = {
            'the', 'and', 'or', 'for', 'with', 'from', 'about', 'this', 'that',
            'what', 'when', 'where', 'how', 'filetype', 'pdf', 'doc', 'docx',
            'official', 'reports', 'analysis', 'relationship', 'influence', 'on'
        }
        query_tokens = {t for t in raw_tokens if len(t) >= 3 and t not in stop}
        
        for url, title, source in urls:
            title_lower = (title or '').lower()
            url_lower = (url or '').lower()

            # If we couldn't extract meaningful tokens, treat results as relevant.
            if not query_tokens:
                scored.append((url, title, source, 1.0))
                continue

            # Count matches in title and URL
            title_matches = sum(1 for token in query_tokens if token in title_lower)
            url_matches = sum(1 for token in query_tokens if token in url_lower)
            
            # With OSINT dork queries, results are already highly targeted
            # Accept results even without title matches if URL matches
            # or if query has few tokens (highly specific)
            total_matches = title_matches + url_matches
            
            # Be very permissive: accept if ANY match found, or if query is very specific
            if total_matches == 0 and len(query_tokens) > 2:
                # Only skip if query has many tokens and zero matches
                continue

            # Weight title matches higher than URL matches.
            weighted = (2.0 * title_matches) + (1.0 * url_matches)
            # Use a more forgiving denominator
            score = max(0.15, weighted / max((1.5 * len(query_tokens)), 1.0))

            if score >= self.min_relevance_score:
                # Prefer official/preferred sources when crawler has a policy.
                try:
                    if hasattr(self.crawler, '_is_preferred_url') and self.crawler._is_preferred_url(url):
                        score = min(1.0, score * 1.25)
                except Exception:
                    pass
                scored.append((url, title, source, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored

    def _generate_osint_crossref_queries(
        self,
        entity_name: str,
        entity_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SearchQuery]:
        """Generate explicit OSINT-style cross-reference queries.

        This complements the keyword-driven follow-ups with deterministic
        high-signal queries using the full Google dork operator set.

        Context may include:
        - official_domain: str (e.g., company.com)
        - related_entities: list[str]
        - target_type: str (override for entity_type mapping)
        """
        et = (entity_type or '').upper().strip()
        name = (entity_name or '').strip()
        if not name:
            return []

        official_domain = None
        try:
            official_domain = (context or {}).get('official_domain')
            if official_domain:
                official_domain = str(official_domain).strip()
        except Exception:
            official_domain = None

        related = []
        try:
            related = list((context or {}).get('related_entities') or [])
        except Exception:
            related = []

        related_sites = []
        try:
            related_sites = list((context or {}).get('related_sites') or [])
        except Exception:
            related_sites = []

        # Check for target_type override in context
        target_type_override = None
        try:
            target_type_override = (context or {}).get('target_type')
            if target_type_override:
                target_type_override = str(target_type_override).lower().strip()
        except Exception:
            target_type_override = None

        # Map entity_type to target_type if no override
        if target_type_override and target_type_override != 'auto':
            target_type = target_type_override
        elif et == 'PERSON':
            target_type = 'person'
        elif et in {'ORG', 'COMPANY', 'ORGANIZATION'}:
            target_type = 'company'
        else:
            target_type = 'auto'

        # Use the comprehensive OSINT dork builder from AdvancedSearchBuilder
        queries = self.builder.create_osint_dorks_by_type(
            name=name,
            target_type=target_type,
            domain=official_domain,
            context=related[0] if related else None
        )

        # Add context-aware dorks using related entities and related sites
        try:
            queries.extend(
                self.builder.create_contextual_dorks(
                    name=name,
                    related_entities=related,
                    related_sites=related_sites
                )
            )
        except Exception:
            pass

        # Add role confirmation queries if we have related entities
        if related and et == 'PERSON':
            primary = str(related[0]).strip()
            if primary and primary.lower() != name.lower():
                queries.append(SearchQuery(query=f'"{name}" "{primary}" CEO OR CFO OR CTO OR director OR manager'))

        # De-dupe by built query string and limit to avoid overwhelming search APIs
        seen = set()
        deduped: List[SearchQuery] = []
        max_queries = None
        try:
            max_queries = getattr(self.crawler.config, 'osint_max_queries', None)
            if max_queries is not None:
                max_queries = int(max_queries)
        except Exception:
            max_queries = None
        for q in queries:
            try:
                s = q.build()
            except Exception:
                s = str(q)
            if s in seen:
                continue
            seen.add(s)
            deduped.append(q)
            if max_queries and max_queries > 0 and len(deduped) >= max_queries:
                break

        return deduped
    
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

        # Execute all queries in parallel (bounded by semaphore)
        async def _run_query(i: int, query: SearchQuery) -> List[Tuple[str, str, str, float]]:
            query_str = query.build()
            logger.info(f"Query {i}/{len(queries)}: {query_str}")

            # Jitter per task to reduce CAPTCHA triggers
            try:
                import random
                await asyncio.sleep(random.uniform(0.3, 1.2))
            except Exception:
                pass

            try:
                async with self._search_sem:
                    results = await self.crawler.search_google(
                        query_str,
                        max_results=self.max_results_per_query
                    )
            except Exception as e:
                logger.warning(f"  Query failed: {e}")
                return []

            scored = self._filter_urls_by_relevance(
                results,
                query.query
            )
            logger.info(f"  Found {len(scored)} relevant results")
            return scored

        tasks = [
            _run_query(i, query)
            for i, query in enumerate(queries, 1)
        ]

        for scored in await asyncio.gather(*tasks, return_exceptions=False):
            for url, title, source, score in scored:
                if url not in seen_urls and url not in self.all_discovered_urls:
                    seen_urls.add(url)
                    all_results.append((url, title, source, score))
        
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
        max_queries = None
        try:
            max_queries = getattr(self.crawler.config, 'osint_max_queries', None)
            if max_queries is not None:
                max_queries = int(max_queries)
        except Exception:
            max_queries = None

        for keyword in list(new_keywords):
            # Create targeted searches
            queries.append(SearchQuery(
                query=f"{base_entity} {keyword}",
                filetype="pdf"
            ))
            
            queries.append(SearchQuery(
                query=f"{base_entity} {keyword}",
                intitle=keyword
            ))

            if max_queries and max_queries > 0 and len(queries) >= max_queries:
                break
        
        # If context available, use related entities
        if context and 'related_entities' in context:
            for related_entity in context['related_entities']:
                queries.append(SearchQuery(
                    query=f"{base_entity} {related_entity}",
                    exact_phrases=[f"{base_entity} {related_entity}"]
                ))

                if max_queries and max_queries > 0 and len(queries) >= max_queries:
                    break
        
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
        
        # Iteration 1: Explicit OSINT cross-reference + contextual search
        context_entities = context.get('related_entities', []) if context else []

        crossref = self._generate_osint_crossref_queries(entity_name, entity_type, context=context)

        if context_entities:
            contextual = self._generate_contextual_initial_queries(entity_name, entity_type, context_entities)
        else:
            contextual = [
                SearchQuery(query=f'"{entity_name}"'),
                SearchQuery(query=f'"{entity_name}" profile OR biography'),
                SearchQuery(query=f'"{entity_name}"', filetype="pdf")
            ]

        # Combine, de-dupe, and keep iteration bounded if configured.
        combined = crossref + contextual
        seen_built = set()
        initial_queries: List[SearchQuery] = []
        max_initial = None
        try:
            max_initial = getattr(self.crawler.config, 'osint_max_queries', None)
            if max_initial is not None:
                max_initial = int(max_initial)
        except Exception:
            max_initial = None
        for q in combined:
            built = q.build()
            if built in seen_built:
                continue
            seen_built.add(built)
            initial_queries.append(q)
            if max_initial and max_initial > 0 and len(initial_queries) >= max_initial:
                break

        logger.info(f"Generated {len(initial_queries)} initial queries for '{entity_name}' (cross-ref + contextual).")

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
