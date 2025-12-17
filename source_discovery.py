"""
Multi-source entity discovery module.
Discovers URLs from multiple sources beyond Wikipedia.
"""
from typing import List, Dict, Set, Optional
from logger import setup_logger
from urllib.parse import quote

logger = setup_logger(__name__)


class MultiSourceDiscovery:
    """Handles entity discovery across multiple sources."""

    def __init__(self, sources: Optional[List[str]] = None, preferred: Optional[List[str]] = None, blacklisted: Optional[List[str]] = None):
        """
        Initialize multi-source discovery.

        Args:
            sources: List of source names to use (wikipedia, wikidata, dbpedia, web)
            preferred: Optional list of preferred source tokens (priority only)
            blacklisted: Optional list of blacklisted source tokens (deny)
        """
        self.sources = sources or ['wikipedia', 'wikidata', 'dbpedia']
        self.preferred_set = {s.strip().lower() for s in (preferred or []) if s and s.strip()}
        self.blacklist_set = {s.strip().lower() for s in (blacklisted or []) if s and s.strip()}
        logger.info(f"Multi-source discovery enabled for: {', '.join(self.sources)}")
    
    def discover_urls_for_entity(self, entity: str, lang: str = 'en') -> Dict[str, List[str]]:
        """
        Discover URLs for an entity across multiple sources.
        
        Args:
            entity: Entity name to discover URLs for
            lang: Language code (en, ro, etc.)
            
        Returns:
            Dictionary mapping source name to list of URLs
        """
        urls = {}
        entity_encoded = quote(entity.replace(' ', '_'))
        
        # Wikipedia - Try both the specified language and English
        if 'wikipedia' in self.sources and 'wikipedia' not in self.blacklist_set:
            wiki_urls = [
                f"https://{lang}.wikipedia.org/wiki/{entity_encoded}"
            ]
            # Add English Wikipedia if not already English
            if lang != 'en':
                wiki_urls.append(f"https://en.wikipedia.org/wiki/{entity_encoded}")
            urls['wikipedia'] = wiki_urls
        
        # Wikidata - Skip direct crawling, but entity linker fetches metadata via API
        # Wikidata pages are not good for text extraction (mostly structured data)
        # The EntityLinker already provides Wikidata QIDs and metadata
        
        # DBpedia - Only if explicitly requested
        # DBpedia resources have limited text content, mostly structured data
        # Better to rely on Wikipedia which has richer narrative content
        if 'dbpedia' in self.sources and 'dbpedia' not in self.blacklist_set:
            urls['dbpedia'] = [
                f"http://dbpedia.org/resource/{entity_encoded}"
            ]
        
        # General web search (DuckDuckGo) - Usually too noisy, skip unless explicitly requested
        if 'web' in self.sources and 'web' not in self.blacklist_set:
            search_query = quote(entity)
            urls['web'] = [
                f"https://duckduckgo.com/?q={search_query}"
            ]
        
        logger.debug(f"Discovered {sum(len(v) for v in urls.values())} URLs for '{entity}' across {len(urls)} sources")
        return urls
    
    def get_all_urls_for_entities(self, entities: List[str], lang: str = 'en') -> List[tuple[str, str, str]]:
        """
        Get all URLs for a list of entities across all sources.
        
        Args:
            entities: List of entity names
            lang: Language code
            
        Returns:
            List of (url, entity, source) tuples
        """
        all_urls = []
        
        for entity in entities:
            entity_urls = self.discover_urls_for_entity(entity, lang)
            
            for source, urls in entity_urls.items():
                # Skip blacklisted sources at generation time
                if source in self.blacklist_set:
                    continue
                for url in urls:
                    all_urls.append((url, entity, source))
        
        logger.info(f"Discovered {len(all_urls)} URLs for {len(entities)} entities")
        return all_urls
    
    def prioritize_sources(self, urls_with_sources: List[tuple[str, str, str]]) -> List[tuple[str, str, str]]:
        """
        Prioritize URLs by source reliability.
        Wikipedia > Wikidata > DBpedia > Web
        
        Args:
            urls_with_sources: List of (url, entity, source) tuples
            
        Returns:
            Sorted list of (url, entity, source) tuples
        """
        source_priority = {
            'wikipedia': 1,
            'wikidata': 2,
            'dbpedia': 3,
            'web': 4
        }
        
        def key_fn(item: tuple[str, str, str]) -> float:
            src = item[2]
            base = source_priority.get(src, 999)
            # Preferred sources get slightly better priority (lower key)
            if src in self.preferred_set:
                return base - 0.5
            return base

        sorted_urls = sorted(urls_with_sources, key=key_fn)
        
        return sorted_urls
