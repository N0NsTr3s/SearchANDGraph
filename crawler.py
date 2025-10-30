"""
Web crawler module for collecting content from web pages.
"""
import asyncio
from pathlib import Path
import re
import hashlib
import pickle
import unicodedata
from typing import List, Set, Tuple, Dict, Optional, TYPE_CHECKING
from urllib.parse import urljoin, urlparse
from collections import deque
import nodriver as uc
import inspect
import trafilatura
from bs4 import BeautifulSoup
from config import CrawlerConfig, CacheConfig
from logger import setup_logger
import diskcache

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from web_search import WebSearcher


class WebCrawler:
    """Handles web crawling and content extraction."""
    
    def __init__(
        self,
        config: CrawlerConfig,
        cache_config: Optional[CacheConfig] = None,
        web_searcher: Optional['WebSearcher'] = None
    ):
        """
        Initialize the web crawler.
        
        Args:
            config: Crawler configuration settings
            cache_config: Cache configuration settings
            web_searcher: Optional web search helper for reseeding the crawl queue
        """
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.browser = None
        self.discovered_entities = set()  # Track entities found during crawling
        self.entity_relationships = {}  # entity -> set of related entities
        self.query_entities = set()  # Entities directly from the query
        self.max_relationship_hops = 2  # How many hops away to consider
        self.entity_mention_count = {}  # Track how often entities are mentioned
        self.high_value_entities = set()  # Entities worth searching for
        self.web_searcher = web_searcher
        self._used_search_urls: set[str] = set()
        self._search_attempts_by_query: dict[str, int] = {}
        self._max_search_attempts_per_query = 3
        
        # Initialize document extractor for PDFs and images
        from document_extractor import DocumentExtractor
        self.document_extractor = DocumentExtractor(
            download_dir=str(Path(self.config.checkpoint_file).parent / "downloads") if self.config.checkpoint_file else "downloads"
        )
        logger.info("Document extractor initialized")
        
        # Initialize cache
        if self.cache_config.enabled:
            self.cache = diskcache.Cache(
                self.cache_config.cache_dir,
                size_limit=self.cache_config.max_cache_size
            )
            logger.info(f"Disk cache enabled at {self.cache_config.cache_dir}")
        else:
            self.cache = None
            logger.info("Disk cache disabled")
    async def start(self):
        """Start the browser instance."""
        logger.info("Starting browser...")
        self.browser = await uc.start(headless=self.config.browser_headless)
        
    async def close(self):
        """Close the browser instance robustly.

        Some browser backend implementations expose a synchronous close/stop method
        or an async coroutine. Calling `await` on a non-awaitable (None) raises
        "object NoneType can't be used in 'await' expression". Handle both sync
        and async patterns safely and set `self.browser` to None after attempting
        to close.
        """
        if not self.browser:
            logger.info("No browser to close.")
            return

        logger.info("Closing browser...")
        try:
            # Try a set of common shutdown method names. Call the first available
            # and await it only if it returns an awaitable.
            closed = False
            for method_name in ("stop", "close", "close_session", "disconnect"):
                method = getattr(self.browser, method_name, None)
                if callable(method):
                    try:
                        result = method()
                    except TypeError:
                        # If the method requires arguments unexpectedly, skip it
                        logger.debug(f"Browser.{method_name}() raised TypeError, skipping")
                        continue

                    if inspect.isawaitable(result):
                        await result
                        closed = True
                    else:
                        # Synchronous close (returned None or similar)
                        closed = True

                    if closed:
                        break

            if not closed:
                # As a last resort, try to call .close() directly without assumptions
                direct_close = getattr(self.browser, "close", None)
                if callable(direct_close):
                    try:
                        res = direct_close()
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        logger.debug("Direct browser.close() failed or raised exception")

        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        finally:
            # Ensure we drop reference so subsequent runs don't try to reuse it
            self.browser = None

    async def search_google(self, query: str, max_results: int = 10) -> List[Tuple[str, str, str]]:
        """
        Performs a Google search using the browser and scrapes the results.
        Returns a list of (url, title, source) tuples.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of (url, title, source) tuples
        """
        if not self.browser:
            # Try to start the browser automatically if it isn't running.
            logger.info("Browser not started for search_google(); attempting to start browser...")
            try:
                await self.start()
            except Exception as e:
                logger.error(f"Failed to start browser for search_google(): {e}")
                raise RuntimeError("Browser not started. Call start() before searching.") from e

        from urllib.parse import quote_plus
        
        logger.info(f"Performing Google search in browser: '{query}'")
        search_url="https://www.google.com/"
        await asyncio.sleep(1)
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        results = []
        tab = None
        try:
            tab = await self.browser.get(search_url, new_tab=True)
            await tab.wait_for('load', timeout=15)
            
            # Wait for the main search results container to appear
            await tab.wait_for_selector('#search', timeout=10)

            # Try to dismiss common consent dialogs (Google cookie banners) if present
            try:
                # Common id used by Google for consent dialog
                consent_btn = await tab.select_one('#L2AGLb')
                if consent_btn:
                    try:
                        await consent_btn.click()
                        await tab.wait_for('load', timeout=3)
                    except Exception:
                        # ignore click failures and continue
                        pass
            except Exception:
                pass

            # Use a single-page JS evaluation to extract result links/titles to avoid
            # working with potentially stale element handles (which can cause
            # "Could not find node with given id" CDP errors when the DOM mutates).
            try:
                script = '''() => {
                    const out = [];
                    // Search result containers can be found under #search; use common result wrapper classes
                    const nodes = document.querySelectorAll('#search .g');
                    for (const node of nodes) {
                        const a = node.querySelector('a');
                        const h3 = node.querySelector('h3');
                        if (a && h3 && a.href && a.href.startsWith('http')) {
                            out.push({href: a.href, title: h3.innerText});
                        }
                    }
                    // As a fallback, also try any link with an h3 inside
                    if (out.length === 0) {
                        const anchors = document.querySelectorAll('#search a');
                        for (const a of anchors) {
                            const h3 = a.querySelector('h3');
                            if (a.href && h3 && a.href.startsWith('http')) {
                                out.push({href: a.href, title: h3.innerText});
                            }
                        }
                    }
                    return out;
                }'''

                scraped = await tab.evaluate(script)
                if isinstance(scraped, list):
                    for item in scraped:
                        try:
                            href = item.get('href')
                            title = item.get('title')
                            if href and title and not any(junk in href for junk in ['google.com/search', 'accounts.google.com']):
                                results.append((href, title, 'google'))
                                if len(results) >= max_results:
                                    break
                        except Exception:
                            continue
            except Exception as e:
                # Last-resort: fall back to element handles (less robust)
                logger.debug(f"JS evaluation of search results failed, falling back to element handles: {e}")
                links = await tab.select_all('#search a')
                for link in links[:max_results*2]:  # Get more links to filter down
                    try:
                        href = await link.get_attribute('href')
                        h3 = await link.select('h3')
                        
                        if href and h3 and href.startswith('http'):
                            title = await h3.get_text()
                            # Basic validation to filter out non-result links
                            if title and not any(junk in href for junk in ['google.com/search', 'accounts.google.com']):
                                results.append((href, title, 'google'))
                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue  # Ignore links that can't be parsed
                        
        except Exception as e:
            logger.error(f"Browser-based Google search for '{query}' failed: {e}")
        finally:
            if tab:
                try:
                    await tab.close()
                except Exception:
                    pass  # Ignore errors on tab close
        
        logger.info(f"Browser-based search found {len(results)} results.")
        return results

    def set_web_searcher(self, searcher: Optional['WebSearcher']) -> None:
        """Attach or replace the web search helper used for reseeding."""
        self.web_searcher = searcher
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for a URL."""
        return f"page:{hashlib.md5(url.encode()).hexdigest()}"
    
    def get_cached_page(self, url: str) -> Optional[str]:
        """
        Get cached page content if available and not expired.
        
        Args:
            url: URL to check cache for
            
        Returns:
            Cached content or None
        """
        if not self.cache:
            return None
        
        try:
            key = self._get_cache_key(url)
            content = self.cache.get(key)
            if content and isinstance(content, str):
                logger.debug(f"Cache HIT for {url}")
                return content
            logger.debug(f"Cache MISS for {url}")
        except Exception as e:
            logger.warning(f"Cache read error for {url}: {e}")
        
        return None
    
    def cache_page(self, url: str, content: str):
        """
        Cache page content with TTL.
        
        Args:
            url: URL to cache
            content: Page content
        """
        if not self.cache:
            return
        
        try:
            key = self._get_cache_key(url)
            self.cache.set(
                key,
                content,
                expire=self.cache_config.page_cache_ttl
            )
            logger.debug(f"Cached page: {url}")
        except Exception as e:
            logger.warning(f"Cache write error for {url}: {e}")
    
    def add_entities(self, entities: list[str], mark_as_query: bool = False):
        """
        Add discovered entities to track for link finding.
        This allows the crawler to find links related to any entity in the graph,
        not just the original query.
        
        Args:
            entities: List of entity names to track
            mark_as_query: If True, marks these entities as query entities (distance 0)
        """
        # Clean and normalize entity names
        for entity in entities:
            # Convert to lowercase for case-insensitive matching
            normalized = entity.lower().strip()
            if normalized and len(normalized) > 2:  # Skip very short entities
                self.discovered_entities.add(normalized)
                
                if mark_as_query:
                    self.query_entities.add(normalized)
        
        logger.info(f"Tracking {len(self.discovered_entities)} entities for link discovery")
        if mark_as_query:
            logger.info(f"Query entities: {len(self.query_entities)}")
    
    def add_relationships(self, relations: dict):
        """
        Add entity relationships to track connections in the graph.
        This allows the crawler to understand which entities are connected
        and only follow links to entities that have a path to the query.
        
        Args:
            relations: Dictionary mapping (entity1, entity2) -> [reasons]
        """
        added_count = 0
        
        for (source, target), reasons in relations.items():
            # Normalize entity names
            source_normalized = source.lower().strip()
            target_normalized = target.lower().strip()
            
            # Build bidirectional relationship map
            if source_normalized not in self.entity_relationships:
                self.entity_relationships[source_normalized] = set()
            if target_normalized not in self.entity_relationships:
                self.entity_relationships[target_normalized] = set()
            
            # Add both directions (undirected graph)
            self.entity_relationships[source_normalized].add(target_normalized)
            self.entity_relationships[target_normalized].add(source_normalized)
            added_count += 1
        
        logger.info(f"Added {added_count} relationships, tracking {len(self.entity_relationships)} connected entities")
    
    def is_entity_connected_to_query(self, entity: str, max_hops: int | None = None) -> bool:
        """
        Check if an entity is connected to any query entity within max_hops.
        Uses BFS (Breadth-First Search) to find connection path.
        
        Args:
            entity: Entity name to check
            max_hops: Maximum number of relationship hops to search (default: self.max_relationship_hops)
            
        Returns:
            True if entity is connected to query within max_hops
        """
        if max_hops is None:
            max_hops = self.max_relationship_hops
        
        entity_normalized = entity.lower().strip()
        
        # If no query entities set, treat all entities as connected
        if not self.query_entities:
            return True
        
        # If entity is a query entity, it's connected by definition
        if entity_normalized in self.query_entities:
            return True
        
        # If no relationships tracked yet, allow all entities
        if not self.entity_relationships:
            return True
        
        # BFS to find if entity is reachable from any query entity
        from collections import deque
        
        # Start from all query entities
        queue = deque()
        visited = set()
        
        for query_ent in self.query_entities:
            queue.append((query_ent, 0))  # (entity, hop_distance)
        
        while queue:
            current_entity, hops = queue.popleft()
            
            # Found the target entity
            if current_entity == entity_normalized:
                logger.debug(f"Entity '{entity}' connected to query at {hops} hops")
                return True
            
            # Max hops reached
            if hops >= max_hops:
                continue
            
            # Already visited
            if current_entity in visited:
                continue
            visited.add(current_entity)
            
            # Add neighbors to queue
            neighbors = self.entity_relationships.get(current_entity, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))
        
        # Not connected within max_hops
        logger.debug(f"Entity '{entity}' NOT connected to query within {max_hops} hops")
        return False
    
    def _matches_any_entity(self, text: str, check_connectivity: bool = True) -> bool:
        """
        Check if text contains any of the discovered entities that are connected to the query.
        
        Args:
            text: Text to check (URL, link text, etc.)
            check_connectivity: If True, only match entities connected to query
            
        Returns:
            True if text contains any tracked entity (that is connected to query if checking)
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check if any entity appears in the text
        for entity in self.discovered_entities:
            # Replace underscores and dashes with spaces for URL matching
            entity_variants = [
                entity,
                entity.replace(' ', '_'),
                entity.replace(' ', '-'),
                entity.replace(' ', '%20')
            ]
            
            for variant in entity_variants:
                if variant in text_lower:
                    # If checking connectivity, verify entity is connected to query
                    if check_connectivity:
                        if self.is_entity_connected_to_query(entity):
                            logger.debug(f"Matched connected entity '{entity}' in text")
                            return True
                        else:
                            logger.debug(f"Skipping unconnected entity '{entity}'")
                    else:
                        # Not checking connectivity, match any entity
                        return True
        
        return False
    
    def track_entity_mention(self, entity: str, count: int = 1):
        """
        Track mentions of an entity to identify high-value entities.
        Only tracks entities that appear to be in English.
        
        Args:
            entity: Entity name
            count: Number of mentions to add (default 1)
        """
        # Skip non-English entities (simple heuristic: check for Romanian diacritics)
        if any(char in entity for char in 'ăâîșțĂÂÎȘȚ'):
            logger.debug(f"Skipping non-English entity: '{entity}'")
            return
        
        # Additional check: skip if entity contains common Romanian words
        romanian_words = {'din', 'pentru', 'între', 'după', 'ministerul', 'guvernului', 
                         'români', 'româniei', 'către', 'despre', 'conform'}
        entity_lower = entity.lower()
        if any(word in entity_lower for word in romanian_words):
            logger.debug(f"Skipping Romanian entity: '{entity}'")
            return
        
        if entity not in self.entity_mention_count:
            self.entity_mention_count[entity] = 0
        self.entity_mention_count[entity] += count
        
        # Mark as high-value if mentioned frequently
        if self.entity_mention_count[entity] >= 3:  # Threshold
            if entity not in self.high_value_entities:
                self.high_value_entities.add(entity)
                logger.info(f"Identified high-value entity: '{entity}' ({self.entity_mention_count[entity]} mentions)")
    
    def get_top_entities(self, top_n: int = 10, exclude_query: bool = True) -> List[Tuple[str, int]]:
        """
        Get the most frequently mentioned entities.
        
        Args:
            top_n: Number of top entities to return
            exclude_query: Exclude query entities from results
            
        Returns:
            List of (entity, mention_count) tuples, sorted by frequency
        """
        sorted_entities = sorted(
            self.entity_mention_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if exclude_query:
            # Filter out query entities
            query_lower = [q.lower() for q in self.query_entities]
            sorted_entities = [
                (entity, count) for entity, count in sorted_entities
                if entity.lower() not in query_lower
            ]
        
        return sorted_entities[:top_n]
    
    async def fetch_page_content(self, url: str) -> Tuple[str, str]:
        """
        Fetch and extract text content from a URL.
        Uses disk cache to avoid re-fetching pages.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (html_content, extracted_text)
            
        Raises:
            Exception: If page cannot be fetched or processed
        """
        # Check cache first
        cached = self.get_cached_page(url)
        if cached:
            # Parse cached data
            try:
                data = pickle.loads(cached.encode('latin1'))
                logger.info(f"Using cached content for: {url}")
                return data['html'], data['text']
            except Exception as e:
                logger.warning(f"Failed to parse cached data for {url}: {e}")
        
        if not self.browser:
            raise RuntimeError("Browser not started. Call start() first.")
        
        logger.info(f"Fetching: {url}")
        
        page = await self.browser.get(url)
        
        # Wait for page to be fully loaded
        await self._wait_for_page_load(page)
        
        # Check for 404 or error pages
        is_error_page = await self._check_for_error_page(page)
        if is_error_page:
            raise ValueError(f"Page not found or error page (404): {url}")
        
        # Try to accept cookies
        await self._accept_cookies(page)
        await asyncio.sleep(0.5)  # Brief pause after cookie acceptance
        
        # Remove any persistent overlays
        await self._remove_persistent_overlays(page)
        await asyncio.sleep(0.3)
        
        html_content = await page.get_content()
        
        # Extract main text content using trafilatura with better settings
        text_content = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,  # Include tables for Wikipedia infoboxes
            include_links=False,
            favor_precision=True,  # Favor precision to reduce boilerplate
            favor_recall=False,
            no_fallback=False
        )
        
        # Fallback: try with default settings if nothing extracted
        if not text_content:
            logger.debug("First extraction failed, trying with default settings...")
            text_content = trafilatura.extract(html_content)
        
        # Second fallback: extract visible text from page directly
        if not text_content:
            logger.debug("Trafilatura extraction failed, trying direct text extraction...")
            try:
                text_content = await page.get_content()
                # Basic cleaning: remove script/style tags content
                import re
                text_content = re.sub(r'<script[^>]*>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                text_content = re.sub(r'<[^>]+>', ' ', text_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
            except Exception as e:
                logger.warning(f"Direct text extraction also failed: {e}")
        
        if not text_content or len(text_content.strip()) < 100:
            raise ValueError(f"Could not extract sufficient content from {url}")
        
        # Apply additional block-level filtering
        text_content = self._filter_noisy_blocks(text_content)
        
        # Cache the result
        try:
            cache_data = pickle.dumps({'html': html_content, 'text': text_content})
            self.cache_page(url, cache_data.decode('latin1'))
        except Exception as e:
            logger.warning(f"Failed to cache page {url}: {e}")
        
        return html_content, text_content
    
    def _filter_noisy_blocks(self, text: str) -> str:
        """
        Remove short, noisy, or boilerplate blocks from extracted text.
        This provides an additional layer of filtering beyond trafilatura.
        
        Args:
            text: Extracted text content
            
        Returns:
            Cleaned text with noisy blocks removed
        """
        if not text:
            return text
        
        # Split into blocks (paragraphs separated by double newlines)
        blocks = text.split('\n\n')
        clean_blocks = []
        
        # Navigation and boilerplate keywords (multilingual)
        nav_keywords = [
            # English
            'menu', 'navigation', 'subscribe', 'copyright', 'cookie', 'privacy policy', 
            'terms of service', 'all rights reserved', 'sign in', 'log in', 'register',
            'follow us', 'share', 'tweet', 'facebook', 'twitter', 'instagram',
            'related articles', 'you may also like', 'recommended for you',
            'advertisement', 'sponsored', 'loading...', 'click here',
            # Romanian
            'meniu', 'navigare', 'abonare', 'drepturi rezervate', 'politica de confidentialitate',
            'termeni si conditii', 'termeni și condiții', 'autentificare', 'inregistrare', 'înregistrare',
            'urmareste', 'distribuie', 'articole similare', 'publicitate',
            # French
            'droits réservés', 'politique de confidentialité', 'conditions d\'utilisation',
            # Spanish
            'derechos reservados', 'política de privacidad', 'términos de servicio',
            # German
            'alle rechte vorbehalten', 'datenschutz', 'nutzungsbedingungen'
        ]
        
        for block in blocks:
            block_stripped = block.strip()
            
            # Skip empty blocks
            if not block_stripped:
                continue
            
            # Skip very short blocks (likely captions, nav items, or fragments)
            if len(block_stripped) < 40:
                continue
            
            # Skip blocks with high symbol density (navigation, menus, metadata)
            symbol_count = sum(c in '[]{}()©®™|→←↑↓•·»«' for c in block_stripped)
            if len(block_stripped) > 0 and symbol_count / len(block_stripped) > 0.15:
                continue
            
            # Skip blocks with navigation/boilerplate keywords
            block_lower = block_stripped.lower()
            if any(keyword in block_lower for keyword in nav_keywords):
                continue
            
            # Skip blocks that are mostly numbers/dates (likely metadata or timestamps)
            digit_count = sum(c.isdigit() for c in block_stripped)
            if len(block_stripped) > 0 and digit_count / len(block_stripped) > 0.4:
                continue
            
            # Skip blocks with very high punctuation density (likely lists or menus)
            punct_count = sum(c in '.,;:!?-–—' for c in block_stripped)
            word_count = len(block_stripped.split())
            if word_count > 0 and punct_count / word_count > 2:  # More than 2 punctuation per word
                continue
            
            # Skip blocks that look like URLs or email lists
            if block_stripped.count('@') > 2 or block_stripped.count('http') > 3:
                continue
            
            # Block passed all filters
            clean_blocks.append(block_stripped)
        
        # Rejoin cleaned blocks
        cleaned_text = '\n\n'.join(clean_blocks)
        
        # Log filtering statistics
        original_length = len(text)
        cleaned_length = len(cleaned_text)
        if original_length > 0:
            reduction_pct = ((original_length - cleaned_length) / original_length) * 100
            if reduction_pct > 10:  # Only log if significant filtering occurred
                logger.debug(
                    f"Block filtering: reduced text by {reduction_pct:.1f}% "
                    f"({original_length} → {cleaned_length} chars, "
                    f"{len(blocks)} → {len(clean_blocks)} blocks)"
                )
        
        return cleaned_text
    
    async def find_related_links(
        self, 
        page: uc.Tab, 
        base_url: str, 
        query: str | None = None,
        return_link_text: bool = False
    ):
        """
        Find and return absolute URLs on the page.
        
        Args:
            page: Browser page/tab object
            base_url: Base URL for resolving relative links
            query: Optional search query for filtering links (if None, returns all)
            
        Returns:
            If return_link_text is False: List of absolute URLs (strings).
            If return_link_text is True: List of tuples (absolute_url, link_text).
        """
        related_links = set()
        related_with_text = []
        
        try:
            links = await page.select_all('a')
            
            for link in links:
                try:
                    # Handle different nodriver API versions
                    link_text = None
                    if hasattr(link, 'text'):
                        link_text = link.text if isinstance(link.text, str) else None
                    if not link_text and hasattr(link, 'text_content'):
                        try:
                            link_text = await link.text_content  # type: ignore
                        except:
                            pass
                    
                    # If query is None, capture all links; otherwise filter by query
                    should_include = (query is None) or (link_text and query.lower() in link_text.lower())
                    
                    if should_include:
                        href = None
                        if hasattr(link, 'attrs') and isinstance(link.attrs, dict):
                            href = link.attrs.get('href')
                        if not href and hasattr(link, 'get_attribute'):
                            try:
                                href = await link.get_attribute('href')  # type: ignore
                            except:
                                pass
                        
                        if href:
                            absolute_url = urljoin(base_url, href)
                            
                            # Normalize URLs to avoid duplicates (removes fragments, trailing slashes)
                            normalized_url = self._normalize_url(absolute_url)
                            normalized_base = self._normalize_url(base_url)
                            
                            # Skip if link points to the same page (self-reference)
                            if normalized_url == normalized_base:
                                logger.debug(f"Skipping self-referencing link: {absolute_url}")
                                continue
                            
                            # Validate URL (checks for action links, images, etc.)
                            if self._is_valid_url(normalized_url):
                                if return_link_text:
                                    # preserve order with list of tuples (url, text)
                                    link_text_safe = (link_text or '').strip()
                                    related_with_text.append((normalized_url, link_text_safe))
                                else:
                                    related_links.add(normalized_url)
                                
                except Exception as e:
                    logger.debug(f"Error processing link: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error finding links on {base_url}: {e}")
        
        if return_link_text:
            return related_with_text
        return list(related_links)
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """
        Check if URL is valid for crawling.
        Excludes action links (javascript:, mailto:, tel:), most image links, action URLs (edit, add, remove, etc.),
        authentication/login pages in multiple languages, and non-HTTPS URLs.
        ALLOWS: PDFs, DOC, PPT, XLS files, and some images for OCR.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            url_lower = url.lower()
            
            # SECURITY: Only allow HTTPS URLs (exclude HTTP)
            if parsed.scheme != 'https':
                logger.debug(f"Filtering non-HTTPS URL: {url}")
                return False
            
            # Check for action protocols that should be excluded
            action_protocols = ['javascript:', 'mailto:', 'tel:', 'sms:', 'data:', 'file:']
            
            for protocol in action_protocols:
                if url_lower.startswith(protocol):
                    return False
            
            # NEW: Allow document extensions for document extraction
            document_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
            is_document = any(url_lower.endswith(ext) for ext in document_extensions)
            
            # NEW: Allow certain image extensions for OCR
            ocr_image_extensions = ['.png', '.jpg', '.jpeg']
            is_ocr_image = any(url_lower.endswith(ext) for ext in ocr_image_extensions)
            
            # If it's a document or OCR-capable image, allow it
            if is_document or is_ocr_image:
                logger.debug(f"Allowing document/image URL: {url}")
                return True
            
            # Check for other image file extensions (still exclude these)
            other_image_extensions = ['.gif', '.bmp', '.svg', '.webp', '.ico', '.tiff']
            for ext in other_image_extensions:
                if url_lower.endswith(ext):
                    return False
            
            # Check if URL contains image-related paths
            image_patterns = ['/file:', '/image:', '/special:filepath', '/special:upload']
            for pattern in image_patterns:
                if pattern in url_lower:
                    return False
            
            
            # ENHANCED: Multilingual authentication/login/edit keywords
            # English, Romanian, French, Spanish, German, Italian, etc.
            auth_and_action_keywords = [
                # Edit/Modify (multilingual)
                'edit', 'editar', 'modifier', 'bearbeiten', 'modificare', 'modifica',
                'editare', 'editeaza', 'editează',
                
                # Login/Authentication (multilingual)
                'login', 'log-in', 'log_in', 'signin', 'sign-in', 'sign_in',
                'autentificare', 'authentification', 'authentication', 'auth',
                'conectare', 'logare', 'inloggen', 'anmelden', 'connexion',
                'iniciar', 'acceder', 'entrar',
                
                # Logout
                'logout', 'log-out', 'log_out', 'signout', 'sign-out', 'sign_out',
                'deconectare', 'iesire', 'ieșire',
                
                # Register/Signup
                'register', 'signup', 'sign-up', 'sign_up', 'inregistrare', 
                'înregistrare', 'inscriere', 'înscriere', 'registrar',
                
                # Account/Profile/User management
                'account', 'cont', 'profile', 'profil', 'user', 'utilizator',
                'settings', 'setari', 'setări', 'preferences', 'preferinte', 'preferințe',
                
                # CRUD operations
                'add', 'create', 'delete', 'remove', 'update', 'modify',
                'submit', 'post', 'upload', 'download', 'adauga', 'adaugă',
                'sterge', 'șterge', 'actualizeaza', 'actualizează',
                
                # Social/Interactive actions
                'share', 'comment', 'reply', 'partajeaza', 'partajează', 
                'comentariu', 'raspunde', 'răspunde',
                
                # Query parameters indicating actions
                'action=', 'do=', 'cmd=', 'mode=edit', 'mode=add', 'mode=login',
                'veaction=edit', 'section=',  # Wikipedia edit links
                
                # Special pages
                'special:', 'especial:', 'spezial:'
            ]
            
            url_path_and_query = (parsed.path + '?' + parsed.query).lower()
            
            for keyword in auth_and_action_keywords:
                if keyword in url_path_and_query:
                    logger.debug(f"Filtering action/auth URL containing '{keyword}': {url}")
                    return False
            
            # Check for standalone year pages (e.g., /1990, /2020, /Year_1985)
            # These are typically not relevant unless they're part of a larger context
            path_parts = parsed.path.split('/')
            for part in path_parts:
                # Match patterns like "1900-2099", "Year_1990", "AD_2000", etc.
                if re.match(r'^(year[_\-]?)?\d{4}(_AD|_BC)?$', part.lower()):
                    logger.debug(f"Filtering year page: {url}")
                    return False
            
            # Validate HTTPS scheme and has domain
            return parsed.scheme == 'https' and bool(parsed.netloc)
            
        except Exception:
            return False
    
    @staticmethod
    def _is_document_url(url: str) -> bool:
        """
        Check if URL points to a downloadable document or image for OCR.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is a document/image that needs special processing
        """
        url_lower = url.lower()
        document_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
                              '.png', '.jpg', '.jpeg']
        return any(url_lower.endswith(ext) for ext in document_extensions)
    
    async def _check_for_error_page(self, page: uc.Tab) -> bool:
        """
        Check if the current page is a 404 or other error page.
        
        Args:
            page: Browser page/tab object
            
        Returns:
            True if page is an error page (404, 403, 500, etc.), False otherwise
        """
        try:
            # Strategy 1: Check HTTP status code from page title or meta tags
            page_title = await page.evaluate('document.title')
            page_title_str = str(page_title) if page_title else ''
            page_title_lower = page_title_str.lower()
            
            # Common 404 indicators in title
            error_indicators = [
                '404', 'not found', 'page not found', 
                '403', 'forbidden', 'access denied',
                '500', 'internal server error', 'server error',
                'error', 'unavailable', 'does not exist',
                'pagina nu a fost gasita',  # Romanian: page not found
                'pagină negăsită',
                'eroare 404',
            ]
            
            for indicator in error_indicators:
                if indicator in page_title_lower:
                    logger.debug(f"Error page detected in title: '{page_title}'")
                    return True
            
            # Strategy 2: Check page content for error messages
            body_text = await page.evaluate('''
                () => {
                    const body = document.body;
                    if (!body) return '';
                    return body.innerText ? body.innerText.toLowerCase() : '';
                }
            ''')
            
            # Check for common error messages in body
            body_text_str = str(body_text) if body_text else ''
            if body_text_str:
                body_sample = body_text_str[:500]  # Check first 500 chars
                
                error_patterns = [
                    '404 - page not found',
                    'error 404',
                    '404 not found',
                    'the page you are looking for',
                    'this page does not exist',
                    'page cannot be found',
                    'pagina nu poate fi gasita',  # Romanian
                    'aceasta pagina nu exista',
                ]
                
                for pattern in error_patterns:
                    if pattern in body_sample:
                        logger.debug(f"Error page detected in content: contains '{pattern}'")
                        return True
            
            # Strategy 3: Check for very short content (often indicates error page)
            if body_text_str and len(body_text_str.strip()) < 100:
                logger.debug(f"Possible error page: very short content ({len(body_text_str)} chars)")
                return True
            
            # Strategy 4: Check for specific error page classes or IDs
            has_error_element = await page.evaluate('''
                () => {
                    // Check for common error page identifiers
                    const errorSelectors = [
                        '#error', '.error-page', '#not-found', '.not-found',
                        '#page-404', '.page-404', '[class*="error-404"]',
                        '[id*="error-404"]'
                    ];
                    
                    for (const selector of errorSelectors) {
                        if (document.querySelector(selector)) {
                            return true;
                        }
                    }
                    return false;
                }
            ''')
            
            if has_error_element:
                logger.debug("Error page detected: has error-related element")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking for error page: {e}")
            return False  # If check fails, assume page is OK
    
    async def _wait_for_page_load(self, page: uc.Tab, timeout: int = 20) -> bool:
        """
        Wait for page to be fully loaded before collecting data.
        Uses multiple strategies to ensure content is ready.
        
        Args:
            page: Browser page/tab object
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if page loaded successfully, False otherwise
        """
        try:
            logger.debug(f"Waiting for key content element on {getattr(page, 'url', '<unknown>')}...")

            # Define a list of common selectors for main content areas.
            # The crawler will wait for the first one of these to appear.
            content_selectors = 'main, article, [role="main"], #content, .content, #main, .main'

            # Use the browser's built-in waiting mechanism. Convert timeout to milliseconds
            # since many browser APIs expect ms.
            try:
                await page.wait_for_selector(content_selectors, timeout=timeout * 1000)
                logger.debug("Key content element found. Page content is likely rendered.")
            except Exception as e:
                # Different browser backends raise different timeout exceptions; treat them
                # uniformly as a timeout condition to avoid noisy logs.
                if isinstance(e, asyncio.TimeoutError):
                    raise
                # Some drivers raise their own TimeoutError subclasses; handle generically below
                logger.debug(f"wait_for_selector raised: {e}")

            # Add a brief, final delay to allow lazy-loaded assets or trailing scripts
            # to complete after the main content is present.
            await asyncio.sleep(1.5)

            return True

        except asyncio.TimeoutError:
            logger.warning(f"Timeout: Did not find a key content element within {timeout}s. Page may be empty or still loading.")
            return False
        except Exception as e:
            logger.warning(f"An unexpected error occurred while waiting for page elements: {e}")
            return False

    async def _accept_cookies(self, page: uc.Tab) -> bool:
        """
        Comprehensive cookie, ad, and overlay handler.
        Tries to:
        1. Accept cookie banners (or reject if needed)
        2. Close ad overlays and popups
        3. Dismiss newsletter signups
        4. Remove blocking elements
        """
        # Wait for page to stabilize and banners to appear
        await asyncio.sleep(2.5)

        try:
            import json

            # Accept button phrases (prioritized)
            accept_phrases = [
                'accept all', 'accept cookies', 'allow all', 'agree and close',
                'i agree', 'i accept', 'consent', 'agree', 'accept', 'allow',
                'acceptă', 'acceptă toate', 'sunt de acord',  # Romanian
                'acepto', 'aceptar todas',  # Spanish
                'accetto', 'accetta tutti',  # Italian
                'akzeptieren', 'alle akzeptieren',  # German
                'j\'accepte', 'tout accepter',  # French
                'ok', 'got it', 'understood', 'continue'
            ]
            
            # Reject button phrases (use as fallback if accept not found)
            reject_phrases = [
                'reject all', 'reject', 'refuse', 'decline', 'deny',
                'respinge', 'refuz',  # Romanian
                'rechazar', 'denegar',  # Spanish  
                'rifiuta', 'nega',  # Italian
                'ablehnen', 'verweigern',  # German
                'refuser', 'rejeter'  # French
            ]
            
            # Close/dismiss phrases for ads and popups
            close_phrases = [
                'close', 'dismiss', 'no thanks', 'maybe later', 'not now',
                'skip', 'cancel', 'later', 'închide', 'nu mulțumesc',
                'cerrar', 'chiudi', 'schließen', 'fermer', '×', '✕', 'x'
            ]
            
            # Phrases to explicitly EXCLUDE (never click these)
            exclude_phrases = [
                'share', 'partajeaza', 'partajează', 'compartir', 'teilen',
                'login', 'signin', 'register', 'signup', 'edit', 'comment',
                'reply', 'follow', 'subscribe', 'buy', 'shop', 'cart',
                'settings', 'customize', 'preferences', 'manage', 'learn more'
            ]
            
            accept_json = json.dumps(accept_phrases)
            reject_json = json.dumps(reject_phrases)
            close_json = json.dumps(close_phrases)
            exclude_json = json.dumps(exclude_phrases)

            # Comprehensive script to handle all blocking elements
            script = f"""
            (function(){{
                const acceptPhrases = {accept_json};
                const rejectPhrases = {reject_json};
                const closePhrases = {close_json};
                const excludePhrases = {exclude_json};
                
                let clickedSomething = false;
                
                // STEP 1: Remove overlay elements that block content
                function removeOverlays() {{
                    const overlaySelectors = [
                        '[class*="overlay"][style*="fixed"]',
                        '[class*="overlay"][style*="absolute"]',
                        '[class*="modal-backdrop"]',
                        '[class*="popup-overlay"]',
                        '[id*="overlay"]',
                        'div[style*="position: fixed"][style*="z-index"]',
                        'div[style*="position:fixed"][style*="z-index"]'
                    ];
                    
                    for (const selector of overlaySelectors) {{
                        try {{
                            const elements = document.querySelectorAll(selector);
                            elements.forEach(el => {{
                                const style = window.getComputedStyle(el);
                                const zIndex = parseInt(style.zIndex);
                                // Remove high z-index overlays (typically blocking elements)
                                if (zIndex > 100 && style.position === 'fixed') {{
                                    el.remove();
                                    console.log('Removed blocking overlay');
                                    clickedSomething = true;
                                }}
                            }});
                        }} catch(e) {{ }}
                    }}
                }}
                
                // STEP 2: Find and click buttons
                function findAndClickButtons(phraseSets, priority) {{
                    const selectors = [
                        'button:not([style*="display: none"]):not([style*="display:none"])', 
                        'a[role="button"]:not([style*="display: none"])', 
                        'div[role="button"]:not([style*="display: none"])',
                        'input[type="button"]:not([style*="display: none"])', 
                        'input[type="submit"]:not([style*="display: none"])',
                        // Cookie/consent specific
                        '*[class*="cookie"]:not([style*="display: none"])',
                        '*[class*="consent"]:not([style*="display: none"])',
                        '*[id*="cookie"]:not([style*="display: none"])',
                        '*[id*="consent"]:not([style*="display: none"])',
                        '*[class*="accept"]:not([style*="display: none"])',
                        '*[class*="banner"]:not([style*="display: none"])',
                        // Ad/popup close buttons
                        '*[class*="close"]:not([style*="display: none"])',
                        '*[class*="dismiss"]:not([style*="display: none"])',
                        '*[aria-label*="close" i]:not([style*="display: none"])',
                        '*[aria-label*="dismiss" i]:not([style*="display: none"])',
                        'button[class*="modal"]:not([style*="display: none"])'
                    ];
                    
                    let allElements = [];
                    for (const sel of selectors) {{
                        try {{
                            const elements = Array.from(document.querySelectorAll(sel));
                            // Filter to only visible elements
                            const visibleElements = elements.filter(el => {{
                                const style = window.getComputedStyle(el);
                                const rect = el.getBoundingClientRect();
                                return style.display !== 'none' && 
                                       style.visibility !== 'hidden' && 
                                       style.opacity !== '0' &&
                                       rect.width > 0 && rect.height > 0;
                            }});
                            allElements = allElements.concat(visibleElements);
                        }} catch(e) {{ }}
                    }}
                    
                    // Remove duplicates
                    allElements = Array.from(new Set(allElements));
                    
                    // Sort by priority: larger buttons first (more likely to be primary action)
                    allElements.sort((a, b) => {{
                        const aRect = a.getBoundingClientRect();
                        const bRect = b.getBoundingClientRect();
                        const aArea = aRect.width * aRect.height;
                        const bArea = bRect.width * bRect.height;
                        return bArea - aArea;
                    }});
                    
                    // Try each phrase set in order of priority
                    for (const phrases of phraseSets) {{
                        for (const el of allElements) {{
                            try {{
                                const text = (
                                    el.innerText || 
                                    el.value || 
                                    el.getAttribute('aria-label') || 
                                    el.getAttribute('title') || 
                                    el.textContent || 
                                    ''
                                ).toLowerCase().trim();
                                
                                // Skip empty text
                                if (!text) continue;
                                
                                // Skip if text contains any exclude phrase
                                let shouldExclude = false;
                                for (const ex of excludePhrases) {{
                                    if (text.includes(ex)) {{
                                        shouldExclude = true;
                                        break;
                                    }}
                                }}
                                if (shouldExclude) continue;
                                
                                // Check if text contains any phrase from current set
                                for (const phrase of phrases) {{
                                    if (text.includes(phrase)) {{
                                        try {{ 
                                            el.click();
                                            console.log(`Clicked ${{priority}} button:`, phrase, '→', text.substring(0, 50));
                                            return true;
                                        }} catch(e) {{ 
                                            console.log('Click failed:', e);
                                        }}
                                    }}
                                }}
                            }} catch(e) {{ }}
                        }}
                    }}
                    return false;
                }}
                
                // STEP 3: Try clicking in iframes
                function tryIframes(phrases) {{
                    try {{
                        const iframes = document.querySelectorAll('iframe');
                        for (const iframe of iframes) {{
                            try {{
                                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                                if (iframeDoc) {{
                                    const buttons = iframeDoc.querySelectorAll('button, a[role="button"], div[role="button"]');
                                    for (const btn of buttons) {{
                                        const text = (btn.innerText || btn.textContent || '').toLowerCase();
                                        for (const phrase of phrases) {{
                                            if (text.includes(phrase)) {{
                                                btn.click();
                                                console.log('Clicked in iframe:', phrase);
                                                return true;
                                            }}
                                        }}
                                    }}
                                }}
                            }} catch(e) {{ /* CORS/access denied */ }}
                        }}
                    }} catch(e) {{ }}
                    return false;
                }}
                
                // Execute strategies in order of priority
                
                // 1. Remove blocking overlays
                removeOverlays();
                
                // 2. Try to click ACCEPT buttons first (best for consent)
                if (findAndClickButtons([acceptPhrases], 'ACCEPT')) {{
                    return true;
                }}
                
                // 3. Try to click CLOSE buttons (for ads, popups, newsletters)
                if (findAndClickButtons([closePhrases], 'CLOSE')) {{
                    return true;
                }}
                
                // 4. If no accept/close found, try REJECT as fallback
                if (findAndClickButtons([rejectPhrases], 'REJECT')) {{
                    return true;
                }}
                
                // 5. Try iframes with accept phrases
                if (tryIframes(acceptPhrases)) {{
                    return true;
                }}
                
                // 6. Try iframes with close phrases
                if (tryIframes(closePhrases)) {{
                    return true;
                }}
                
                return clickedSomething;
            }})()
            """

            # Try multiple times with delays (some banners appear late)
            for attempt in range(4):  # Increased from 3 to 4 attempts
                try:
                    clicked = await page.evaluate(script)
                    if clicked:
                        logger.info(f"✓ Handled blocking element (cookies/ads/overlays) on attempt {attempt + 1}")
                        await asyncio.sleep(1.0)  # Give page time to process
                        
                        # Try once more to catch secondary popups
                        if attempt == 0:
                            await asyncio.sleep(0.5)
                            clicked_again = await page.evaluate(script)
                            if clicked_again:
                                logger.info("✓ Handled secondary blocking element")
                        
                        return True
                    
                    if attempt < 3:
                        await asyncio.sleep(0.7)  # Wait before retrying
                        
                except Exception as e:
                    logger.debug(f"Blocking element handler attempt {attempt + 1} failed: {e}")
                    if attempt < 3:
                        await asyncio.sleep(0.5)
            
            # Nothing handled after all attempts
            logger.debug("No blocking elements found or handled")
            return False
            
        except Exception as e:
            logger.debug(f"_accept_cookies unexpected error: {e}")
            return False
    
    async def _remove_persistent_overlays(self, page: uc.Tab) -> bool:
        """
        Aggressively remove persistent blocking overlays that remain after cookie/ad handling.
        This is called right before content extraction to ensure clean page access.
        """
        try:
            script = """
            (function(){
                let removedCount = 0;
                
                // Remove fixed/absolute positioned elements with high z-index
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    const style = window.getComputedStyle(el);
                    const position = style.position;
                    const zIndex = parseInt(style.zIndex) || 0;
                    
                    // Target fixed/absolute elements with very high z-index (likely overlays)
                    if ((position === 'fixed' || position === 'absolute') && zIndex > 999) {
                        // Check if it's covering significant screen area
                        const rect = el.getBoundingClientRect();
                        const screenArea = window.innerWidth * window.innerHeight;
                        const elementArea = rect.width * rect.height;
                        
                        // If element covers >30% of screen and is high z-index, it's likely blocking
                        if (elementArea > screenArea * 0.3) {
                            el.remove();
                            removedCount++;
                            console.log('Removed persistent overlay with z-index:', zIndex);
                        }
                    }
                });
                
                // Remove modal backdrops
                const backdropSelectors = [
                    '.modal-backdrop', '.overlay-backdrop', '[class*="backdrop"]',
                    '.popup-overlay', '[id*="overlay"]', '[class*="popup-overlay"]'
                ];
                backdropSelectors.forEach(selector => {
                    try {
                        document.querySelectorAll(selector).forEach(el => {
                            el.remove();
                            removedCount++;
                        });
                    } catch(e) {}
                });
                
                // Remove elements that prevent scrolling
                if (document.body.style.overflow === 'hidden') {
                    document.body.style.overflow = 'auto';
                    document.documentElement.style.overflow = 'auto';
                    removedCount++;
                }
                
                return removedCount > 0;
            })()
            """
            
            removed = await page.evaluate(script)
            if removed:
                logger.info("✓ Removed persistent blocking overlays")
                return True
            return False
            
        except Exception as e:
            logger.debug(f"Failed to remove persistent overlays: {e}")
            return False
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalize URL by removing fragments and trailing slashes.
        This helps prevent visiting the same page multiple times with different fragments.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL without fragment
        """
        # Remove fragment (everything after #)
        url_without_fragment = url.split('#')[0]
        
        # Remove trailing slash for consistency (but keep it for domain roots)
        parsed = urlparse(url_without_fragment)
        if parsed.path and parsed.path != '/' and url_without_fragment.endswith('/'):
            url_without_fragment = url_without_fragment.rstrip('/')
        
        return url_without_fragment
    
    def _calculate_url_relevance(self, url: str, query: str, discovered_entities: set | None = None) -> float:
        """
        Calculate relevance score for a URL based on query and discovered entities.
        Only matches WHOLE WORDS in URLs (separated by -, /, _, or other delimiters).
        
        Args:
            url: URL to score
            query: Main search query
            discovered_entities: Set of entity names already discovered
            
        Returns:
            Relevance score (0.0 to 1.0+, higher is better)
        """
        score = 0.0
        
        try:
            # Normalize inputs
            query_lower = query.lower() if query else ''
            url_lower = url.lower()
            
            # Extract query words (ignore stop words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            query_words = set(query_lower.split()) - stop_words
            
            # Parse URL into words (split by -, /, _, and other non-alphanumeric chars)
            import re
            # Split URL by common delimiters to extract individual words
            url_words = re.split(r'[-/_.\?&=]', url_lower)
            url_words = [w for w in url_words if w and w not in stop_words]
            url_words_set = set(url_words)
            
            logger.debug(f"URL relevance check for: {url}")
            logger.debug(f"  Query words: {query_words}")
            logger.debug(f"  URL words extracted: {url_words_set}")
            
            # Parse URL
            parsed = urlparse(url_lower)
            path = parsed.path
            
            # PRIORITY 1: Exact multi-word query match in URL (highest score)
            # Check if full query appears as connected words (e.g., "nicusor-dan" or "nicusor_dan")
            if query_lower:
                query_hyphenated = query_lower.replace(' ', '-')
                query_underscored = query_lower.replace(' ', '_')
                
                # Check if WHOLE phrase appears (not as substring)
                # Use word boundaries appropriate for URLs
                if (re.search(r'(?:^|/)' + re.escape(query_hyphenated) + r'(?:/|$|-)', url_lower) or
                    re.search(r'(?:^|/)' + re.escape(query_underscored) + r'(?:/|$|_)', url_lower)):
                    score += 10.0  # Very high score for exact match
            
            # PRIORITY 2: All query words appear in URL AS SEPARATE WORDS
            if query_words:
                # Count how many query words appear as complete words in URL
                words_matched = query_words & url_words_set
                words_in_url = len(words_matched)
                
                if words_in_url == len(query_words):
                    score += 5.0  # All query words present as whole words
                elif words_in_url > 0:
                    score += 2.0 * (words_in_url / len(query_words))  # Partial match
                    
                # Debug: log what matched
                if words_in_url > 0:
                    logger.debug(f"URL word match: {words_matched} in {url}")
            
            # PRIORITY 3: Discovered entity match in URL (as whole words)
            if discovered_entities:
                for entity in discovered_entities:
                    # Split entity into words
                    entity_words = set(entity.lower().split()) - stop_words
                    
                    # Check if all entity words appear in URL
                    if entity_words and entity_words.issubset(url_words_set):
                        score += 3.0  # Entity match
                        break
            
            # PRIORITY 4: Query words in path (not just domain)
            if query_words and path:
                # Extract words from path specifically
                path_words_list = re.split(r'[-/_.\?&=]', path)
                path_words = set(w for w in path_words_list if w and w not in stop_words)
                
                overlap = query_words & path_words
                if overlap:
                    score += 1.0 * len(overlap)
            
            # PENALTY: Deeper paths get slight penalty (prefer more general pages)
            path_depth = len([p for p in path.split('/') if p]) if path else 0
            if path_depth > 3:
                score -= 0.1 * (path_depth - 3)
            
        except Exception as e:
            logger.debug(f"Error calculating URL relevance for {url}: {e}")
        
        return max(0.0, score)  # Ensure non-negative
    
    def _find_contextual_links(self, all_links: List[str], current_url: str, query: str) -> List[str]:
        """
        Find links that are contextually related to the query or discovered entities.
        Prioritizes links by relevance to query/entities, not by domain.
        
        Args:
            all_links: All links found on the page
            current_url: Current page URL
            query: Search query
            
        Returns:
            List of contextually related URLs
        """
        contextual_links = []
        
        try:
            normalized_current = self._normalize_url(current_url)
            query_words = set(query.lower().split()) if query else set()
            
            # Remove common stop words that shouldn't be used for matching
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            query_words = query_words - stop_words
            
            for link in all_links:
                try:
                    # Skip self-referencing links using normalized URLs
                    normalized_link = self._normalize_url(link)
                    if normalized_link == normalized_current:
                        continue
                    
                    parsed_link = urlparse(link)
                    link_path = parsed_link.path.lower()
                    link_full = link.lower()
                    
                    # REMOVED DOMAIN RESTRICTION - prioritize by relevance instead
                    # Check if URL contains any discovered entity
                    if self._matches_any_entity(link_full):
                        contextual_links.append(link)
                        logger.debug(f"Contextual link (entity match): {link}")
                        continue
                    
                    # Check if URL path or full URL contains any query words
                    # This helps find pages like /Person_A when searching for "Person A"
                    path_words = set(link_path.replace('_', ' ').replace('-', ' ').replace('%20', ' ').split('/'))
                    path_words = path_words - stop_words
                    
                    # Check for overlap in meaningful query words.
                    # For multi-word queries require ALL non-stopword terms to appear
                    # in the path (to avoid matching pages that only contain one term).
                    matched_in_path = query_words & path_words
                    if (len(query_words) == 1 and matched_in_path) or (query_words and query_words.issubset(path_words)):
                        contextual_links.append(link)
                        logger.debug(f"Contextual link (path match): {link}")
                        continue
                    
                    # Also check if any query word appears in the full URL as a whole word
                    # (avoid matching substrings inside other words). Use a regex with
                    # alphanumeric boundaries to account for URL separators.
                    if query:
                        matched = False
                        matched_words = set()
                        for word in (w for w in query_words if len(w) > 2):
                            try:
                                # Look for the word with non-alphanumeric boundaries on both sides
                                pattern = rf'(?<![0-9a-zA-Z]){re.escape(word)}(?![0-9a-zA-Z])'
                                if re.search(pattern, link_full):
                                    matched_words.add(word)
                                    # For single-word queries we can stop early
                                    if len(query_words) == 1:
                                        matched = True
                                        break
                            except re.error:
                                # Fallback to simple substring if regex fails for some reason
                                if word in link_full:
                                    matched_words.add(word)
                                    if len(query_words) == 1:
                                        matched = True
                                        break
                        # For multi-word queries ensure all words matched as whole words
                        if matched or (matched_words and (len(query_words) == 1 or query_words.issubset(matched_words))):
                            contextual_links.append(link)
                            logger.debug(f"Contextual link (URL contains whole query word(s)): {link}")
                            continue
                
                except Exception as e:
                    logger.debug(f"Error processing link {link}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error in contextual link finding: {e}")
        
        return contextual_links
    
    async def _process_page_in_tab(
        self,
        url: str,
        depth: int,
        query: str,
        current_domain: str,
        discovered_entities_snapshot: set[str] | None = None
    ) -> dict:
        """
        Process a single page in its own tab.
        
        Args:
            url: URL to process
            depth: Current depth in BFS traversal
            query: Search query for finding related links
            current_domain: Domain to stay within
            discovered_entities_snapshot: Entities known at dispatch time for quick relevance scoring
            
        Returns:
            Dict containing:
                - success: bool indicating if processing succeeded
                - url: the processed URL
                - normalized_url: normalized version of the URL
                - content: extracted text content (if success)
                - all_links: all links found on page (if success)
                - related_links: query-related links (if success)
                - error: error message (if not success)
                - is_error_page: whether this was a 404/error page
        """
        result = {
            'success': False,
            'url': url,
            'normalized_url': self._normalize_url(url),
            'content': None,
            'all_links': [],
            'related_links': [],
            'error': None,
            'is_error_page': False,
            'is_relevant': False,
            'relevance_score': 0.0
        }
        
        tab = None
        try:
            # Create a new tab for this URL
            logger.debug(f"Creating new tab for: {url}")
            tab = await self.browser.get(url, new_tab=True) # type: ignore
            logger.debug(f"Tab created, navigating to: {url}")
            
            # Wait for initial navigation and page load
            await asyncio.sleep(1.0)  # Give the page 1 second to start loading
            await self._wait_for_page_load(tab)

            # Check for 404 or error pages before processing
            is_error_page = await self._check_for_error_page(tab)
            if is_error_page:
                logger.warning(f"Skipping error page (404 or similar): {url}")
                result['is_error_page'] = True
                result['success'] = True  # Mark as successful processing (just nothing to extract)
                return result

            # Attempt to accept cookie banners if present
            cookie_clicked = await self._accept_cookies(tab)
            if cookie_clicked:
                await asyncio.sleep(0.5)

            # Remove any persistent overlays that might still be blocking content
            await self._remove_persistent_overlays(tab)
            await asyncio.sleep(0.3)  # Brief pause after overlay removal

            html_content = await tab.get_content()
            try:
                page_title = await tab.get_title()  # type: ignore[attr-defined]
            except Exception:
                page_title = ''
            
            # DEBUG: Log HTML content length
            logger.debug(f"HTML content length for {url}: {len(html_content)} chars")
            
            # Extract text content with improved settings
            text_content = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=False,
                favor_recall=True
            )
            
            # Log extraction result
            if text_content:
                logger.debug(f"Trafilatura extracted {len(text_content)} chars from {url}")
            else:
                logger.debug(f"Trafilatura extraction returned None for {url}")
            
            # Fallback: try with default settings if nothing extracted
            if not text_content:
                logger.debug(f"First extraction failed for {url}, trying with default settings...")
                text_content = trafilatura.extract(html_content)
                if text_content:
                    logger.debug(f"Default settings extracted {len(text_content)} chars")
            
            # Second fallback: BeautifulSoup extraction (better for dynamic sites)
            if not text_content:
                logger.debug(f"Trafilatura extraction failed for {url}, trying BeautifulSoup...")
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        element.decompose()
                    
                    # Get text from main content areas (prioritize article content)
                    content_selectors = [
                        'article',
                        'main',
                        '[role="main"]',
                        '.article-content',
                        '.post-content',
                        '.entry-content',
                        '.content',
                        'body'
                    ]
                    
                    for selector in content_selectors:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            text_content = content_elem.get_text(separator=' ', strip=True)
                            if text_content and len(text_content) > 200:
                                logger.debug(f"BeautifulSoup extracted {len(text_content)} chars using selector '{selector}'")
                                break
                    
                    # Fallback to full body text if selectors didn't work
                    if not text_content:
                        text_content = soup.get_text(separator=' ', strip=True)
                        if text_content:
                            logger.debug(f"BeautifulSoup extracted {len(text_content)} chars from full body")
                    
                except Exception as e:
                    logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
            
            # Third fallback: extract visible text directly with regex
            if not text_content:
                logger.debug(f"BeautifulSoup failed for {url}, trying regex extraction...")
                try:
                    import re
                    text_content = html_content
                    text_content = re.sub(r'<script[^>]*>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                    text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                    text_content = re.sub(r'<[^>]+>', ' ', text_content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    if text_content:
                        logger.debug(f"Regex extraction got {len(text_content)} chars")
                except Exception as e:
                    logger.warning(f"Regex text extraction also failed for {url}: {e}")
            
            # Lower threshold for content length (50 chars instead of 100)
            min_content_length = 50
            
            if text_content and len(text_content.strip()) >= min_content_length:
                logger.debug(f"Content extraction successful: {len(text_content.strip())} chars (min: {min_content_length})")
            else:
                if text_content:
                    logger.warning(f"Content too short from {url}: {len(text_content.strip())} chars (min: {min_content_length})")
                    logger.debug(f"Short content preview: {text_content[:200]}")
                else:
                    logger.warning(f"No text content extracted from {url}")
            
            if text_content and len(text_content.strip()) >= min_content_length:
                result['content'] = text_content
                combined_text = f"{page_title}\n{text_content}" if page_title else text_content
                
                # Log what we're checking
                preview_len = 200
                text_preview = combined_text[:preview_len].replace('\n', ' ')
                logger.debug(f"Checking relevance for {url}")
                logger.debug(f"  Title: {page_title}")
                logger.debug(f"  Text preview ({len(combined_text)} chars): {text_preview}...")
                logger.debug(f"  Query: {query}")
                
                is_relevant, relevance_score, relevance_debug = self._quick_content_relevance(
                    combined_text,
                    query,
                    discovered_entities_snapshot
                )
                result['is_relevant'] = is_relevant
                result['relevance_score'] = relevance_score
                result['relevance_debug'] = relevance_debug
                
                # Find ALL links on this page first
                all_links_raw = await self.find_related_links(tab, url, query=None)
                
                logger.debug(f"  Found {len(all_links_raw)} total links on page {url}")
                
                # Find query-related links for BFS queue
                query_exact_links = await self.find_related_links(tab, url, query)
                links_with_text = await self.find_related_links(tab, url, query=None, return_link_text=True)

                # Build set of candidate links - only those related to query or entities
                related_links = set()
                query_words = set((query or '').lower().split()) if query else set()

                for url_item in links_with_text:
                    try:
                        url_candidate, link_text = url_item
                    except Exception:
                        continue

                    lower_url = url_candidate.lower()
                    lower_text = (link_text or '').lower()

                    # Match if query appears in link text
                    if query and any(q in lower_text for q in query_words):
                        related_links.add(url_candidate)
                        continue

                    # Match if query term appears in URL path
                    if query and any(q in lower_url for q in query_words if len(q) > 2):
                        related_links.add(url_candidate)
                        continue
                    
                    # Match if any discovered entity appears in link text or URL
                    if self._matches_any_entity(lower_text) or self._matches_any_entity(lower_url):
                        related_links.add(url_candidate)
                        logger.debug(f"    Entity match: {url_candidate}")
                        continue

                # Include contextual matches (only for query, not all links)
                url_only_contextual = self._find_contextual_links(all_links_raw, url, query)
                related_links.update(url_only_contextual)

                # Combine all candidate links
                all_candidate_links = set(query_exact_links) | set(related_links)
                
                # Store only related links in the graph (not all links)
                result['all_links'] = list(all_candidate_links)
                result['related_links'] = list(all_candidate_links)
                
                logger.debug(f"  Found {len(query_exact_links)} exact matches, {len(related_links)} contextually related for {url}")
                
                result['success'] = True
            else:
                logger.warning(f"No content extracted from {url}")
                result['error'] = "No content extracted"
                result['success'] = True  # Not really an error, just no content
                
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            result['error'] = str(e)
            result['success'] = False
        finally:
            # Close the tab after processing
            if tab:
                try:
                    await tab.close()
                except Exception as e:
                    logger.debug(f"Error closing tab for {url}: {e}")
            
        return result
    
    @staticmethod
    def _normalize_text_for_match(text: str | None) -> str:
        """Lowercase and strip diacritics for accent-insensitive comparisons."""
        if not text:
            return ""

        normalized = unicodedata.normalize('NFD', text)
        without_marks = ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')
        return without_marks.lower()

    def _quick_content_relevance(
        self,
        text: str | None,
        query: str | None,
        discovered_entities: set[str] | None = None
    ) -> tuple[bool, float, dict[str, object]]:
        """Lightweight relevance check using query terms, aliases, and entity mentions."""
        debug: dict[str, object] = {
            'matched_query_terms': [],
            'matched_entities': [],
            'query': query,
        }

        if not text:
            debug['reason'] = 'no_text'
            return False, 0.0, debug

        text_norm = self._normalize_text_for_match(text)
        if not text_norm:
            debug['reason'] = 'empty_normalized_text'
            return False, 0.0, debug

        if not query:
            debug['reason'] = 'no_query'
            query_norm = ''
            query_terms: list[str] = []
        else:
            query_norm = self._normalize_text_for_match(query)
            query_terms = [term for term in query_norm.split() if len(term) > 2]

        # Debug: Log first 500 chars of normalized text to see what we're matching against
        logger.debug(f"  Normalized text preview (first 500 chars): {text_norm[:500]}")
        logger.debug(f"  Query terms to match: {query_terms}")

        if query_norm and query_norm in text_norm:
            debug['reason'] = 'full_query_match'
            debug['matched_query_terms'] = [query_norm]
            return True, 1.0, debug

        matches: list[str] = []
        for term in query_terms:
            if term in text_norm:
                matches.append(term)
                logger.debug(f"  ✓ Query term '{term}' found in content")
            else:
                logger.debug(f"  ✗ Query term '{term}' NOT found in content")

        score = (len(matches) / len(query_terms)) if query_terms else 0.0
        debug['matched_query_terms'] = matches

        # Include aliases and entity names tracked globally
        entity_sources: list[tuple[str, str]] = []
        if discovered_entities:
            entity_sources.extend((entity, 'discovered_run') for entity in discovered_entities)
        if self.discovered_entities:
            entity_sources.extend((entity, 'discovered_global') for entity in self.discovered_entities)
        if self.query_entities:
            entity_sources.extend((entity, 'query_entity') for entity in self.query_entities)
        if self.high_value_entities:
            entity_sources.extend((entity, 'high_value') for entity in self.high_value_entities)

        matched_entities: list[str] = []
        for entity, source in entity_sources:
            entity_norm = self._normalize_text_for_match(entity)
            if entity_norm and entity_norm in text_norm:
                matched_entities.append(entity_norm)
                # Boost score slightly depending on provenance
                if source == 'high_value':
                    score += 0.2
                else:
                    score += 0.1

        if matched_entities:
            debug['matched_entities'] = matched_entities

        score = max(0.0, min(score, 1.2))

        if matches:
            boosted_score = min(score + 0.2, 1.0)
            debug['reason'] = 'query_term_match'
            debug['score'] = boosted_score
            return True, boosted_score, debug

        threshold = 0.25 if matched_entities else 0.3
        is_relevant = score >= threshold

        debug['score'] = score
        if not is_relevant:
            debug['reason'] = 'below_threshold'
        else:
            debug['reason'] = 'threshold_met'

        return is_relevant, min(score, 1.0), debug
    
    async def _process_document_url(
        self,
        url: str,
        depth: int,
        query: str,
        normalized_url: str
    ) -> dict:
        """
        Process a document URL (PDF, image, etc.) without using a browser.
        Downloads and extracts content using DocumentExtractor.
        
        Args:
            url: Document URL to process
            depth: Current depth in BFS traversal
            query: Search query for relevance checking
            normalized_url: Normalized URL
            
        Returns:
            Dict containing processing results
        """
        result = {
            'success': False,
            'url': url,
            'normalized_url': normalized_url,
            'content': None,
            'is_relevant': False,
            'relevance_score': 0.0,
            'all_links': [],  # Documents don't have links
            'related_links': [],
            'is_error_page': False,
        }
        
        try:
            logger.info(f"Processing document: {url}")
            
            # Extract query name for folder organization
            query_slug = re.sub(r'[^\w\s-]', '_', query)[:50]
            
            # Download and extract content
            text, tables = await self.document_extractor.process_document_url(
                url,
                query_name=query_slug
            )
            
            if not text or len(text) < 100:
                logger.warning(f"  Document extraction failed or insufficient content: {url}")
                return result
            
            logger.info(f"  Extracted {len(text)} characters from document")
            if tables:
                logger.info(f"  Extracted {len(tables)} table(s) from document")
            
            # Check relevance of extracted text
            # Simple keyword-based relevance check
            query_lower = query.lower()
            text_lower = text.lower()
            
            relevance_score = 0.0
            if query_lower in text_lower:
                relevance_score += 0.5
            
            # Check for query words
            query_words = set(query_lower.split())
            query_words = {w for w in query_words if len(w) > 2}
            
            for word in query_words:
                if word in text_lower:
                    relevance_score += 0.1
            
            is_relevant = relevance_score >= 0.3  # 30% threshold
            
            if is_relevant:
                logger.info(f"  Document is relevant (score: {relevance_score:.2f})")
                result['success'] = True
                result['content'] = text
                result['is_relevant'] = True
                result['relevance_score'] = relevance_score
            else:
                logger.info(f"  Document not relevant (score: {relevance_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {url}: {e}")
            result['error'] = str(e)
            return result

    async def _process_opened_tab(
        self,
        tab: uc.Tab,
        url: str,
        depth: int,
        query: str,
        current_domain: str,
        discovered_entities_snapshot: set[str] | None = None
    ) -> dict:
        """
        Process a tab that has already been opened and is loading.
        This is called after sequential tab opening to process content.
        
        Args:
            tab: Already opened tab
            url: URL being processed
            depth: Current depth in BFS traversal
            query: Search query for finding related links
            current_domain: Domain to stay within
            
        Returns:
            Dict containing processing results
        """
        result = {
            'success': False,
            'url': url,
            'normalized_url': self._normalize_url(url),
            'content': None,
            'all_links': [],
            'related_links': [],
            'error': None,
            'is_error_page': False,
            'is_relevant': False,
            'relevance_score': 0.0
        }
        
        try:
            # Wait for page to be fully loaded (tab was opened 1 second ago)
            await self._wait_for_page_load(tab)

            # Check for 404 or error pages before processing
            is_error_page = await self._check_for_error_page(tab)
            if is_error_page:
                logger.warning(f"Skipping error page (404 or similar): {url}")
                result['is_error_page'] = True
                result['success'] = True
                return result

            # Attempt to accept cookie banners if present
            cookie_clicked = await self._accept_cookies(tab)
            if cookie_clicked:
                await asyncio.sleep(0.5)

            # Remove any persistent overlays that might still be blocking content
            await self._remove_persistent_overlays(tab)
            await asyncio.sleep(0.3)  # Brief pause after overlay removal

            html_content = await tab.get_content()
            
            # DEBUG: Log HTML content length
            logger.debug(f"HTML content length for {url}: {len(html_content)} chars")
            
            # Extract text content with improved settings
            text_content = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=False,
                favor_recall=True
            )
            
            # Log extraction result
            if text_content:
                logger.debug(f"Trafilatura extracted {len(text_content)} chars from {url}")
            else:
                logger.debug(f"Trafilatura extraction returned None for {url}")
            
            # Fallback: try with default settings if nothing extracted
            if not text_content:
                logger.debug(f"First extraction failed for {url}, trying with default settings...")
                text_content = trafilatura.extract(html_content)
                if text_content:
                    logger.debug(f"Default settings extracted {len(text_content)} chars")
            
            # Second fallback: BeautifulSoup extraction (better for dynamic sites)
            if not text_content:
                logger.debug(f"Trafilatura failed for {url}, trying BeautifulSoup extraction...")
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script, style, nav, header, footer, aside elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        element.decompose()
                    
                    # Try to find main content with common selectors (priority order)
                    content_elem = None
                    for selector in ['article', 'main', '[role="main"]', '.article-content', 
                                   '.post-content', '.entry-content', '#content', '.content']:
                        content_elem = soup.select_one(selector)
                        if content_elem:
                            text_content = content_elem.get_text(separator=' ', strip=True)
                            if len(text_content) > 200:  # Only use if substantial content
                                logger.debug(f"BeautifulSoup extracted {len(text_content)} chars using selector '{selector}'")
                                break
                    
                    # If no content found with selectors, try body
                    if not text_content or len(text_content) < 200:
                        body = soup.find('body')
                        if body:
                            text_content = body.get_text(separator=' ', strip=True)
                            logger.debug(f"BeautifulSoup extracted {len(text_content)} chars from full body")
                    
                except Exception as e:
                    logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
            
            # Third fallback: extract visible text directly with regex
            if not text_content:
                logger.debug(f"BeautifulSoup failed for {url}, trying regex extraction...")
                try:
                    import re
                    text_content = html_content
                    text_content = re.sub(r'<script[^>]*>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                    text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                    text_content = re.sub(r'<[^>]+>', ' ', text_content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    if text_content:
                        logger.debug(f"Regex extraction got {len(text_content)} chars")
                except Exception as e:
                    logger.warning(f"Regex text extraction also failed for {url}: {e}")
            
            # Lower threshold for content length (50 chars instead of 100)
            min_content_length = 50
            
            if text_content and len(text_content.strip()) >= min_content_length:
                logger.debug(f"Content extraction successful: {len(text_content.strip())} chars (min: {min_content_length})")
            else:
                if text_content:
                    logger.warning(f"Content too short from {url}: {len(text_content.strip())} chars (min: {min_content_length})")
                    logger.debug(f"Short content preview: {text_content[:200]}")
                else:
                    logger.warning(f"No text content extracted from {url}")
            
            if text_content and len(text_content.strip()) >= min_content_length:
                result['content'] = text_content
                # Get page title for combined relevance check
                try:
                    page_title = await tab.get_title()  # type: ignore[attr-defined]
                except Exception:
                    page_title = ''
                
                combined_text = f"{page_title}\n{text_content}" if page_title else text_content
                
                # Log what we're checking
                preview_len = 200
                text_preview = combined_text[:preview_len].replace('\n', ' ')
                logger.debug(f"Checking relevance for {url}")
                logger.debug(f"  Title: {page_title}")
                logger.debug(f"  Text preview ({len(combined_text)} chars): {text_preview}...")
                logger.debug(f"  Query: {query}")
                
                # Check relevance with combined title + text
                is_relevant, relevance_score, relevance_debug = self._quick_content_relevance(
                    combined_text,
                    query,
                    discovered_entities_snapshot
                )
                result['is_relevant'] = is_relevant
                result['relevance_score'] = relevance_score
                result['relevance_debug'] = relevance_debug
                
                # Find ALL links on this page first
                all_links_raw = await self.find_related_links(tab, url, query=None)
                
                logger.debug(f"  Found {len(all_links_raw)} total links on page {url}")
                
                # Find query-related links for BFS queue
                query_exact_links = await self.find_related_links(tab, url, query)
                links_with_text = await self.find_related_links(tab, url, query=None, return_link_text=True)

                # Build set of candidate links - only those related to query or entities
                related_links = set()
                query_words = set((query or '').lower().split()) if query else set()

                for url_item in links_with_text:
                    try:
                        link_url = url_item if isinstance(url_item, str) else url_item[0]
                        link_text = url_item[1] if isinstance(url_item, tuple) and len(url_item) > 1 else ''
                        link_lower = link_url.lower()
                        text_lower = link_text.lower()
                        
                        # Check if link or link text contains query words
                        if any(word in link_lower or word in text_lower for word in query_words):
                            related_links.add(link_url)
                    except Exception as e:
                        logger.debug(f"Error processing link: {e}")
                        continue
                
                # Combine sets and convert to list
                all_candidate_links = list(set(query_exact_links) | related_links)
                
                # Store only related links in the graph (not all links)
                result['all_links'] = list(all_candidate_links)
                result['related_links'] = list(all_candidate_links)
                
                logger.debug(f"  Found {len(query_exact_links)} exact matches, {len(related_links)} contextually related for {url}")
                
                result['success'] = True
            else:
                logger.warning(f"No content extracted from {url}")
                result['error'] = "No content extracted"
                result['success'] = True  # Not really an error, just no content
                
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            result['error'] = str(e)
            result['success'] = False
        finally:
            # Close the tab after processing
            try:
                await tab.close()
            except Exception as e:
                logger.debug(f"Error closing tab for {url}: {e}")
            
        return result
    
    def _build_reseed_queries(
        self,
        base_query: str,
        discovered_entities: set[str]
    ) -> list[str]:
        """Create a small set of follow-up search queries for reseeding."""
        candidates: list[str] = []

        if base_query:
            candidates.append(base_query)

        # Prioritize entities mentioned frequently in crawled pages
        for entity, _ in self.get_top_entities(top_n=5, exclude_query=False):
            candidates.append(entity)

        if self.high_value_entities:
            high_value_sorted = sorted(
                self.high_value_entities,
                key=lambda ent: self.entity_mention_count.get(ent, 0),
                reverse=True
            )
            candidates.extend(high_value_sorted)

        if discovered_entities:
            contextual_entities = sorted(
                discovered_entities,
                key=lambda e: (-len(e), e.lower())
            )
            candidates.extend(contextual_entities[:5])

        # Dedupe while preserving order and cap total queries
        unique_queries: list[str] = []
        seen = set()
        for candidate in candidates:
            candidate_clean = candidate.strip()
            if not candidate_clean:
                continue
            key = candidate_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_queries.append(candidate_clean)
            if len(unique_queries) >= 6:
                break

        return unique_queries

    async def _reseed_from_web_search(
        self,
        base_query: str,
        discovered_entities: set[str],
        processed_urls: set[str],
        queued_normalized: set[str],
        relevant_urls: set[str],
        remaining_slots: int
    ) -> list[tuple[str, float, int]]:
        """Use the web search helper to find new URLs when the queue runs dry."""
        if not self.web_searcher:
            logger.debug("Web searcher not configured; skipping reseed.")
            return []

        queries = self._build_reseed_queries(base_query, discovered_entities)
        if not queries:
            logger.debug("No candidate queries available for reseeding.")
            return []

        eligible_queries = [
            q for q in queries
            if self._search_attempts_by_query.get(q.lower(), 0) < self._max_search_attempts_per_query
        ]

        if not eligible_queries:
            logger.debug("Reseed skipped: all candidate queries exhausted their retry budget.")
            return []

        max_results_per_query = min(
            max(5, remaining_slots * 5),
            self.config.web_search_max_results
        )

        try:
            results_map = await asyncio.to_thread(
                self.web_searcher.parallel_search,
                eligible_queries,
                max_results_per_query
            )
        except Exception as exc:
            logger.warning(f"Parallel web search reseed failed: {exc}")
            return []

        new_entries: list[tuple[str, float, int]] = []
        target_count = max(remaining_slots, 1) * 3

        for query_text, results in results_map.items():
            query_key = query_text.lower()
            self._search_attempts_by_query[query_key] = self._search_attempts_by_query.get(query_key, 0) + 1

            if not results:
                continue

            try:
                filtered_results = self.web_searcher.filter_urls_by_relevance(
                    results,
                    query_text,
                    min_score=self.config.web_search_min_relevance
                )
            except Exception as exc:
                logger.debug(f"Relevance filtering failed for '{query_text}': {exc}")
                filtered_results = []

            for url, title, source, score in filtered_results:
                normalized_url = self._normalize_url(url)

                if (
                    normalized_url in processed_urls
                    or normalized_url in relevant_urls
                    or normalized_url in queued_normalized
                    or normalized_url in self._used_search_urls
                ):
                    continue

                combined_score = self._calculate_url_relevance(
                    normalized_url,
                    base_query,
                    discovered_entities
                )
                combined_score = max(combined_score, score * 10.0)

                new_entries.append((url, combined_score, 0))
                self._used_search_urls.add(normalized_url)
                queued_normalized.add(normalized_url)

                if len(new_entries) >= target_count:
                    break

            if len(new_entries) >= target_count:
                break

        return new_entries

    async def crawl(
        self,
        start_url: str,
        query: str,
        max_pages: int,
        additional_seed_urls: List[str] | None = None
    ) -> Tuple[List[Tuple[str, str]], Set[str], Dict[str, List[str]]]:
        """
        Crawl web pages using BFS starting from a URL with parallel tab processing.
        Prioritizes pages by relevance to the query and newly discovered entities.

        Args:
            start_url: Starting URL for crawling
            query: Query for finding related pages
            max_pages: Maximum number of relevant pages to crawl
            additional_seed_urls: Optional list of additional seed URLs to add to queue

        Returns:
            Tuple of (content_url_pairs, relevant_urls, link_graph)
            - content_url_pairs: List of (content, url) tuples maintaining order
            - relevant_urls: Set of normalized URLs deemed relevant
            - link_graph: Dict mapping URL to list of outgoing links
        """
        normalized_start = self._normalize_url(start_url)

        # Reset per-run web search tracking
        self._used_search_urls = set()
        self._search_attempts_by_query = {}

        # Priority queue stored as (url, relevance_score, depth)
        urls_to_crawl: list[tuple[str, float, int]] = [(start_url, 100.0, 0)]
        queued_normalized: set[str] = {normalized_start}
        self._used_search_urls.add(normalized_start)

        if additional_seed_urls:
            logger.info(f"Adding {len(additional_seed_urls)} additional seed URLs to queue")
            for seed_url in additional_seed_urls:
                normalized_seed = self._normalize_url(seed_url)
                if normalized_seed in queued_normalized:
                    continue
                relevance = self._calculate_url_relevance(normalized_seed, query, None)
                urls_to_crawl.append((seed_url, relevance, 0))
                queued_normalized.add(normalized_seed)
                self._used_search_urls.add(normalized_seed)
                logger.debug(f"  Added seed (relevance {relevance:.2f}): {seed_url}")

        content_url_pairs: list[tuple[str, str]] = []
        processed_urls: set[str] = set()
        relevant_urls: set[str] = set()
        link_graph: dict[str, list[str]] = {}
        discovered_entities: set[str] = set()
        reseed_attempts = 0
        max_reseed_attempts = 4

        concurrent_tabs = getattr(self.config, 'concurrent_tabs', 5)
        logger.info(f"Using {concurrent_tabs} concurrent tabs for parallel crawling")
        logger.info("Pages will be prioritized by relevance to query and discovered entities")

        from urllib.parse import urlparse

        parsed_start = urlparse(start_url)
        current_domain = parsed_start.netloc

        # Only start browser if not already started
        if not self.browser:
            await self.start()

        try:
            while urls_to_crawl and len(relevant_urls) < max_pages:
                urls_to_crawl.sort(key=lambda x: x[1], reverse=True)

                batch: list[str] = []
                batch_info: list[tuple[str, int, float, str]] = []

                while (
                    len(batch) < concurrent_tabs
                    and urls_to_crawl
                    and len(relevant_urls) + len(batch) < max_pages
                ):
                    current_url, relevance, depth = urls_to_crawl.pop(0)
                    normalized_current = self._normalize_url(current_url)

                    if normalized_current in processed_urls:
                        queued_normalized.discard(normalized_current)
                        continue

                    batch.append(current_url)
                    batch_info.append((current_url, depth, relevance, normalized_current))
                    queued_normalized.discard(normalized_current)

                if not batch:
                    if (
                        not urls_to_crawl
                        and len(relevant_urls) < max_pages
                        and reseed_attempts < max_reseed_attempts
                    ):
                        remaining_slots = max_pages - len(relevant_urls)
                        new_urls = await self._reseed_from_web_search(
                            query,
                            discovered_entities,
                            processed_urls,
                            queued_normalized,
                            relevant_urls,
                            remaining_slots,
                        )
                        reseed_attempts += 1
                        if new_urls:
                            urls_to_crawl.extend(new_urls)
                            logger.info(
                                f"Reseeded crawl queue with {len(new_urls)} URLs after batch depletion"
                            )
                            continue

                    if not urls_to_crawl:
                        logger.info("Crawl queue exhausted; no further pages to process.")
                        break

                    continue

                logger.info(
                    f"Processing batch of {len(batch)} pages in parallel "
                    f"(relevant: {len(relevant_urls)}/{max_pages})"
                )
                top_relevances = [f"{info[2]:.1f}" for info in batch_info[:3]]
                if top_relevances:
                    logger.info(f"Top relevance scores in batch: {', '.join(top_relevances)}")
                logger.info(f"Opening {len(batch)} new tabs with sequential navigation...")

                tabs: list[tuple[str, int, uc.Tab | None]] = []
                document_urls: list[tuple[str, int, float, str]] = []  # Track documents separately
                
                for idx, (url, depth, relevance, normalized_url) in enumerate(batch_info, 1):
                    # Check if this is a document URL that needs special processing
                    if self._is_document_url(url):
                        logger.info(f"  [{idx}/{len(batch)}] Document detected: {url}")
                        document_urls.append((url, depth, relevance, normalized_url))
                        tabs.append((url, depth, None))  # Placeholder to maintain order
                        continue
                    
                    try:
                        logger.debug(f"  [{idx}/{len(batch)}] Opening tab for: {url}")
                        tab = await self.browser.get(url, new_tab=True)  # type: ignore
                        await asyncio.sleep(1.0)
                        tabs.append((url, depth, tab))
                        logger.debug(f"  [{idx}/{len(batch)}] Tab opened and loading: {url}")
                    except Exception as e:
                        logger.warning(f"  [{idx}/{len(batch)}] Failed to open tab for {url}: {e}")
                        tabs.append((url, depth, None))

                logger.info(f"All {len(tabs)} tabs opened, now processing content...")

                tasks = []
                entities_snapshot = discovered_entities.copy() if discovered_entities else None
                
                # Process web pages (tabs with browsers)
                for url, depth, tab in tabs:
                    if tab is not None:
                        tasks.append(
                            self._process_opened_tab(
                                tab,
                                url,
                                depth,
                                query,
                                current_domain,
                                entities_snapshot,
                            )
                        )
                    else:
                        async def failed_result(url: str) -> dict:
                            return {
                                'success': False,
                                'url': url,
                                'normalized_url': self._normalize_url(url),
                                'error': 'Failed to open tab',
                                'is_relevant': False,
                                'relevance_score': 0.0,
                                'all_links': [],
                                'related_links': [],
                            }

                        tasks.append(failed_result(url))
                
                # Process documents separately (no browser needed)
                if document_urls:
                    logger.info(f"Processing {len(document_urls)} document(s)...")
                    for doc_url, doc_depth, doc_relevance, doc_normalized_url in document_urls:
                        tasks.append(
                            self._process_document_url(
                                doc_url,
                                doc_depth,
                                query,
                                doc_normalized_url
                            )
                        )

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for (url, depth, relevance, normalized_url), result in zip(batch_info, results):
                    processed_urls.add(normalized_url)

                    if isinstance(result, Exception):
                        logger.error(f"Exception processing {url}: {result}")
                        link_graph[url] = []
                        continue

                    if not isinstance(result, dict):
                        logger.warning(f"Unexpected result type for {url}: {type(result)}")
                        link_graph[url] = []
                        continue

                    if result.get('is_error_page'):
                        logger.debug(f"  Skipping error page: {url}")
                        link_graph[url] = []
                        continue

                    if not result.get('success'):
                        logger.warning(f"  Failed to process: {url}")
                        link_graph[url] = []
                        continue

                    is_relevant = bool(result.get('is_relevant'))
                    relevance_debug = result.get('relevance_debug')
                    link_graph[url] = result.get('all_links', []) or []

                    content = result.get('content')
                    if content and is_relevant:
                        if normalized_url not in relevant_urls:
                            relevant_urls.add(normalized_url)
                            content_url_pairs.append((content, url))
                            logger.info(
                                f"  [{len(relevant_urls)}/{max_pages}] Relevant page (depth {depth}, score {relevance:.2f}): {url}"
                            )

                        # Entity tracking moved to NLP processor (after translation)
                        # This ensures we only track canonical, translated entities
                        try:
                            import re

                            entity_pattern = r'\b[A-ZĂÂÎȘȚ][a-zăâîșț]+(?:\s+[A-ZĂÂÎȘȚ][a-zăâîșț]+)*\b'
                            found_entities = re.findall(entity_pattern, content)
                            for entity in found_entities:
                                if 3 <= len(entity) <= 50 and entity.lower() not in {'the', 'this', 'that'}:
                                    discovered_entities.add(entity)
                                    # self.track_entity_mention(entity)  # DISABLED - moved to NLP processor
                        except Exception as e:
                            logger.debug(f"    Error extracting entities: {e}")

                    elif content:
                        debug_suffix = ''
                        if isinstance(relevance_debug, dict):
                            matched_terms = ','.join(relevance_debug.get('matched_query_terms', []))
                            matched_entities = ','.join(relevance_debug.get('matched_entities', []))
                            score = relevance_debug.get('score')
                            reason = relevance_debug.get('reason')
                            query_used = relevance_debug.get('query')
                            if score is not None:
                                debug_suffix = (
                                    f" [query='{query_used}' score={float(score):.2f} reason={reason}"
                                    f" matches=[{matched_terms}] entities=[{matched_entities}]]"
                                )
                        else:
                            debug_suffix = f" [relevance_debug type: {type(relevance_debug)}]"
                        logger.info(f"  Irrelevant content skipped (depth {depth}): {url}{debug_suffix}")
                    else:
                        logger.warning(f"  No content extracted from {url}")
                        continue

                    related_links = result.get('related_links', []) or []
                    logger.info(f"    Found {len(related_links)} query-related links")

                    for link in related_links:
                        normalized_link = self._normalize_url(link)

                        if (
                            normalized_link in processed_urls
                            or normalized_link in relevant_urls
                            or normalized_link in queued_normalized
                        ):
                            continue

                        new_depth = depth + 1
                        link_relevance = self._calculate_url_relevance(
                            normalized_link,
                            query,
                            discovered_entities,
                        )

                        urls_to_crawl.append((link, link_relevance, new_depth))
                        queued_normalized.add(normalized_link)
                        logger.debug(
                            f"    Added to queue (relevance {link_relevance:.2f}, depth {new_depth}): {link}"
                        )

                    # Drop any URLs that were already processed to keep the queue clean
                    if urls_to_crawl:
                        urls_to_crawl = [
                            (url, score, depth)
                            for (url, score, depth) in urls_to_crawl
                            if self._normalize_url(url) not in processed_urls
                        ]
                        queued_normalized = {self._normalize_url(url) for url, _, _ in urls_to_crawl}

                    if len(relevant_urls) < max_pages and not urls_to_crawl:
                        if reseed_attempts < max_reseed_attempts:
                            remaining_slots = max_pages - len(relevant_urls)
                            new_urls = await self._reseed_from_web_search(
                                query,
                                discovered_entities,
                                processed_urls,
                                queued_normalized,
                                relevant_urls,
                                remaining_slots,
                            )
                            reseed_attempts += 1
                            if new_urls:
                                urls_to_crawl.extend(new_urls)
                                logger.info(
                                    f"Reseeded crawl queue with {len(new_urls)} URLs after processing batch"
                                )
                                continue
                            logger.info("Web search reseed produced no new URLs after processing batch.")
                        else:
                            logger.info("Reached reseed attempt limit; stopping crawl.")
                        break

                    if not urls_to_crawl:
                        logger.info("Crawl queue empty after batch processing; finishing crawl loop.")
                        break

        finally:
            await self.close()

        logger.info(f"BFS crawl complete: {len(relevant_urls)} relevant pages visited")
        logger.info(f"Link graph contains {len(link_graph)} nodes")

        return content_url_pairs, relevant_urls, link_graph
    
    async def discover_related_urls(self, entities: list[str], base_domain: str = "wikipedia.org") -> list[str]:
        """
        Discover additional URLs related to extracted entities.
        Searches for Wikipedia pages and other resources about each entity.
        
        Args:
            entities: List of entity names to search for
            base_domain: Domain to search within (default: wikipedia.org)
            
        Returns:
            List of discovered URLs
        """
        discovered_urls = []
        
        logger.info(f"Discovering related URLs for {len(entities)} entities...")
        
        for entity in entities[:10]:  # Limit to top 10 entities to avoid too many requests
            try:
                # Build search URL for entity
                entity_encoded = entity.replace(' ', '_')
                
                # Try different URL patterns
                candidate_urls = []
                
                if 'wikipedia.org' in base_domain:
                    # Wikipedia patterns
                    candidate_urls.extend([
                        f"https://ro.wikipedia.org/wiki/{entity_encoded}",
                        f"https://en.wikipedia.org/wiki/{entity_encoded}",
                        f"https://ro.wikipedia.org/wiki/Category:{entity_encoded}",
                    ])
                
                for url in candidate_urls:
                    if self._is_valid_url(url):
                        discovered_urls.append(url)
                        logger.debug(f"  Discovered: {url}")
                
            except Exception as e:
                logger.warning(f"Failed to discover URLs for entity '{entity}': {e}")
        
        logger.info(f"Discovered {len(discovered_urls)} potential URLs")
        return discovered_urls
    
    async def enrich_crawl_with_entities(
        self,
        initial_entities: list[str],
        max_additional_pages: int = 10
    ) -> tuple[list[tuple[str, str]], set[str], dict[str, list[str]]]:
        """
        Enriched crawling that discovers and crawls pages related to extracted entities.
        
        This method:
        1. Takes a list of entities from initial crawl
        2. Discovers Wikipedia/web pages about each entity
        3. Crawls those pages to find more connections
        
        Args:
            initial_entities: List of entity names discovered so far
            max_additional_pages: Maximum number of additional pages to crawl
            
        Returns:
            Tuple of (content_url_pairs, visited_urls, link_graph)
            - content_url_pairs: List of (content, url) tuples maintaining order
            - visited_urls: Set of normalized URLs visited
            - link_graph: Dict mapping URL to list of outgoing links
        """
        logger.info("=" * 60)
        logger.info("Starting enriched entity-based discovery...")
        logger.info(f"Discovering pages for {len(initial_entities)} entities")
        logger.info(f"Max additional pages: {max_additional_pages}")
        logger.info("=" * 60)
        
        # Discover URLs for entities
        discovered_urls = await self.discover_related_urls(initial_entities)
        
        if not discovered_urls:
            logger.warning("No additional URLs discovered")
            return [], set(), {}
        
        # Crawl discovered URLs
        content_url_pairs: list[tuple[str, str]] = []
        all_visited = set()
        all_link_graph = {}
        
        # Only start browser if not already started
        if not self.browser:
            await self.start()
        
        try:
            for idx, url in enumerate(discovered_urls[:max_additional_pages], 1):
                logger.info(f"[{idx}/{min(len(discovered_urls), max_additional_pages)}] Crawling discovered: {url}")
                
                try:
                    # Fetch and process page
                    html_content, text_content = await self.fetch_page_content(url)
                    
                    content_url_pairs.append((text_content, url))
                    all_visited.add(self._normalize_url(url))
                    
                    # Get the page to extract links
                    page = await self.browser.get(url)  # type: ignore
                    await self._wait_for_page_load(page)
                    
                    # Find all links on this page
                    links = await self.find_related_links(page, url, query=None)
                    all_link_graph[url] = links
                    
                    logger.info(f"  OK Extracted content ({len(text_content)} chars) and {len(links)} links")
                    
                except Exception as e:
                    logger.warning(f"  ✗ Failed to crawl {url}: {e}")
                    all_link_graph[url] = []
                    continue
                
        finally:
            await self.close()
        
        logger.info(f"Enriched crawl complete: {len(content_url_pairs)} additional pages")
        return content_url_pairs, all_visited, all_link_graph
