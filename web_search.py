"""
Web search module for discovering relevant URLs via search engines.
"""
import requests
from typing import List, Tuple, Optional
from urllib.parse import quote_plus, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import setup_logger
from advanced_search import SearchQuery, AdvancedSearchBuilder

logger = setup_logger(__name__)


class WebSearcher:
    """Handles web searches to discover relevant URLs."""
    
    def __init__(self, max_results: int = 20):
        """
        Initialize the web searcher.
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    def search_duckduckgo(self, query: str, max_results: int | None = None) -> List[Tuple[str, str]]:
        """
        Search using DuckDuckGo HTML (no API key needed).
        
        Args:
            query: Search query
            max_results: Maximum results to return (default: self.max_results)
            
        Returns:
            List of (url, title) tuples
        """
        max_results = max_results or self.max_results
        results = []
        
        try:
            # DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML to extract links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result links
            for result in soup.find_all('a', class_='result__a', limit=max_results):
                url = result.get('href')
                title = result.get_text(strip=True)
                
                if url and title:
                    # DuckDuckGo uses redirect URLs, extract actual URL
                    if 'uddg=' in url:
                        # Extract from redirect
                        import re
                        match = re.search(r'uddg=([^&]+)', url) # type: ignore
                        if match:
                            from urllib.parse import unquote
                            url = unquote(match.group(1))
                    
                    # Validate URL
                    if url.startswith('http'):
                        results.append((url, title))
                        logger.debug(f"Found: {title} - {url}")
            
            logger.info(f"DuckDuckGo search found {len(results)} results for '{query}'")
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return results
    
    def search_bing(self, query: str, max_results: int | None = None) -> List[Tuple[str, str]]:
        """
        Search using Bing (no API key needed for basic scraping).
        
        Args:
            query: Search query
            max_results: Maximum results to return (default: self.max_results)
            
        Returns:
            List of (url, title) tuples
        """
        max_results = max_results or self.max_results
        results = []
        
        try:
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML to extract links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result links (Bing uses different classes)
            for result in soup.find_all('li', class_='b_algo', limit=max_results):
                link_tag = result.find('a')
                if link_tag:
                    url = link_tag.get('href')
                    title = link_tag.get_text(strip=True)
                    
                    if url and title and url.startswith('http'): # type: ignore
                        results.append((url, title))
                        logger.debug(f"Found: {title} - {url}")
            
            logger.info(f"Bing search found {len(results)} results for '{query}'")
            
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
        
        return results
    
    def search_multi(self, query: str, max_results: int | None = None) -> List[Tuple[str, str, str]]:
        """
        Search using multiple search engines and combine results.
        
        Args:
            query: Search query
            max_results: Maximum total results to return (default: self.max_results)
            
        Returns:
            List of (url, title, source) tuples
        """
        max_results = max_results or self.max_results
        all_results = []
        seen_urls = set()
        
        # Try DuckDuckGo first (more privacy-friendly, no API key)
        ddg_results = self.search_duckduckgo(query, max_results)
        for url, title in ddg_results:
            normalized_url = url.lower().rstrip('/')
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                all_results.append((url, title, 'duckduckgo'))
        
        # If we need more results, try Bing
        if len(all_results) < max_results:
            remaining = max_results - len(all_results)
            bing_results = self.search_bing(query, remaining)
            for url, title in bing_results:
                normalized_url = url.lower().rstrip('/')
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    all_results.append((url, title, 'bing'))
        
        logger.info(f"Multi-search found {len(all_results)} unique results for '{query}'")
        return all_results[:max_results]
    
    def parallel_search(self, queries: List[str], max_results_per_query: int | None = None) -> dict[str, List[Tuple[str, str, str]]]:
        """
        Execute multiple search queries in parallel for better performance.
        
        Args:
            queries: List of search queries to execute
            max_results_per_query: Maximum results per query (default: self.max_results)
            
        Returns:
            Dictionary mapping query to list of (url, title, source) tuples
        """
        if not queries:
            return {}
        
        max_results_per_query = max_results_per_query or self.max_results
        results_map = {}
        
        logger.info(f"Starting parallel search for {len(queries)} queries...")
        
        # Use ThreadPoolExecutor to run searches in parallel
        with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(self.search_multi, query, max_results_per_query): query
                for query in queries
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result(timeout=30)
                    results_map[query] = results
                    completed += 1
                    logger.debug(f"Completed search {completed}/{len(queries)}: '{query}' ({len(results)} results)")
                except Exception as e:
                    logger.warning(f"Parallel search failed for '{query}': {e}")
                    results_map[query] = []
        
        total_results = sum(len(r) for r in results_map.values())
        logger.info(f"Parallel search completed: {total_results} total results from {len(queries)} queries")
        
        return results_map
    
    def filter_urls_by_relevance(self, urls: List[Tuple[str, str, str]], query: str, 
                                 min_score: float = 0.3) -> List[Tuple[str, str, str, float]]:
        """
        Filter and score URLs by relevance to query.
        
        Args:
            urls: List of (url, title, source) tuples
            query: Search query
            min_score: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            List of (url, title, source, score) tuples, sorted by score descending
        """
        scored_results = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_words = {w for w in query_words if len(w) > 2}  # Filter short words
        
        for url, title, source in urls:
            score = 0.0
            
            url_lower = url.lower()
            title_lower = title.lower()
            
            # Score by query presence in URL and title
            if query_lower in url_lower:
                score += 5.0
            if query_lower in title_lower:
                score += 5.0
            
            # Score by individual words
            for word in query_words:
                if word in url_lower:
                    score += 1.0
                if word in title_lower:
                    score += 1.0
            
            # Normalize score
            max_possible = 10.0 + (2.0 * len(query_words))
            normalized_score = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
            
            if normalized_score >= min_score:
                scored_results.append((url, title, source, normalized_score))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        logger.info(f"Filtered {len(scored_results)}/{len(urls)} URLs with score >= {min_score}")
        return scored_results
    
    def build_queries_for_entity(
        self,
        entity_name: str,
        entity_type: str,
        custom_query: Optional[str] = None
    ) -> List[SearchQuery]:
        """
        Build advanced search queries for an entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type (PERSON, ORG, GPE, EVENT, etc.)
            custom_query: Optional custom query string
            
        Returns:
            List of SearchQuery objects
        """
        builder = AdvancedSearchBuilder()
        
        if custom_query:
            # Use custom query as base
            return [SearchQuery(query=custom_query)]
        
        # Build queries based on entity type
        entity_type_upper = entity_type.upper()
        
        if entity_type_upper == "PERSON":
            return builder.create_person_query(entity_name)
        elif entity_type_upper in ["ORG", "ORGANIZATION"]:
            return builder.create_organization_query(entity_name)
        elif entity_type_upper == "EVENT":
            return builder.create_event_query(entity_name)
        elif entity_type_upper in ["GPE", "LOC", "LOCATION"]:
            # For locations, search for information about the place
            return [
                SearchQuery(
                    query=f"{entity_name} information",
                    site="wikipedia.org"
                ),
                SearchQuery(
                    query=f"{entity_name} news",
                    intitle=entity_name
                ),
                SearchQuery(
                    query=f"{entity_name} economy demographics",
                    filetype="pdf"
                )
            ]
        else:
            # Generic search
            return [SearchQuery(query=entity_name)]
    
    def search_with_advanced_query(
        self,
        search_query: SearchQuery,
        max_results: int | None = None
    ) -> List[Tuple[str, str, str]]:
        """
        Execute an advanced search query.
        
        Args:
            search_query: SearchQuery object with operators
            max_results: Maximum results to return
            
        Returns:
            List of (url, title, source) tuples
        """
        # Build query string with operators
        query_str = search_query.build()
        logger.info(f"Executing advanced search: {query_str}")
        
        # Execute multi-engine search
        return self.search_multi(query_str, max_results)
    
    def search_entity_advanced(
        self,
        entity_name: str,
        entity_type: str,
        max_results_per_query: int = 10
    ) -> dict[str, List[Tuple[str, str, str]]]:
        """
        Perform comprehensive advanced search for an entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Entity type (PERSON, ORG, etc.)
            max_results_per_query: Max results per query
            
        Returns:
            Dictionary mapping query string to results
        """
        # Build advanced queries
        queries = self.build_queries_for_entity(entity_name, entity_type)
        
        # Convert SearchQuery objects to strings
        query_strings = [q.build() for q in queries]
        
        logger.info(f"Performing advanced search for {entity_type} '{entity_name}' with {len(query_strings)} queries")
        
        # Execute queries in parallel
        return self.parallel_search(query_strings, max_results_per_query)

