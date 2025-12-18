"""
Web search module for discovering relevant URLs via search engines.
"""
import os
import re
import mimetypes
from utils.http_helper import HTTPClient
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import quote_plus, urlparse, unquote, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from ..utils.logger import setup_logger
from advanced_search import SearchQuery, AdvancedSearchBuilder
import requests
logger = setup_logger(__name__)


class WebSearcher:
    """Handles web searches to discover relevant URLs and download content."""
    
    # File type mappings for filtering and categorization
    DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.csv', '.rtf', '.odt'}
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.ico', '.tiff', '.tif'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma'}
    
    def __init__(self, max_results: int = 20, download_dir: str = "./downloads"):
        """
        Initialize the web searcher.
        
        Args:
            max_results: Maximum number of search results to return
            download_dir: Directory to save downloaded files
        """
        self.max_results = max_results
        self.download_dir = Path(download_dir)
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        self._setup_directories()

        # Shared HTTP client for retries/timeouts
        self.http = HTTPClient(timeout=10.0, max_retries=2)

        # Manifest for dedupe across crawler + web search downloads.
        try:
            from utils.download_manifest import DownloadManifest

            self._manifest = DownloadManifest(self.download_dir / "manifest.json")
        except Exception:
            self._manifest = None
    
    def _setup_directories(self) -> None:
        """Create download directories if they don't exist."""
        # Use a single flat download folder.
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_extension(self, url: str, content_type: Optional[str] = None) -> str:
        """Extract file extension from URL or content-type header."""
        # Try from URL first
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        
        if ext and len(ext) <= 5:  # Valid extension
            return ext
        
        # Fall back to content-type header
        if content_type:
            mime_type = content_type.split(';')[0].strip()
            guessed_ext = mimetypes.guess_extension(mime_type)
            return guessed_ext or ''
        
        return ''
    
    def _get_save_directory(self, extension: str) -> Path:
        """Determine the appropriate save directory based on file extension."""
        return self.download_dir
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        # Remove invalid characters for Windows/Unix
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename
    
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
            
            response = self.http.session.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML to extract links
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result links
            for result in soup.find_all('a', class_='result__a', limit=max_results):
                url = result.get('href')
                title = result.get_text(strip=True)
                
                if url and title:
                    # DuckDuckGo uses redirect URLs, extract actual URL
                    if 'uddg=' in url:
                        # Extract from redirect
                        match = re.search(r'uddg=([^&]+)', url)
                        if match:
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
            
            response = self.http.session.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML to extract links
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
        scored_results: List[Tuple[str, str, str, float]] = []

        query_lower = (query or "").lower()
        # Normalize query terms (remove quotes/operators).
        raw_tokens = re.findall(r"[\w\-]+", query_lower)
        stop = {
            'the', 'and', 'or', 'for', 'with', 'from', 'about', 'this', 'that',
            'what', 'when', 'where', 'how', 'filetype', 'pdf', 'doc', 'docx',
            'official', 'reports', 'analysis', 'relationship', 'influence', 'on'
        }
        query_tokens = [t for t in raw_tokens if len(t) >= 3 and t not in stop]
        query_token_set = set(query_tokens)
        
        for url, title, source in urls:
            url_lower = (url or "").lower()
            title_lower = (title or "").lower()

            # If query is too short/empty after normalization, keep results (caller can clamp).
            if not query_token_set:
                scored_results.append((url, title, source, 1.0))
                continue

            # Require at least one meaningful query token in the TITLE.
            # This dramatically reduces noisy results where tokens only appear in the URL/path.
            title_matches = sum(1 for t in query_token_set if t in title_lower)
            if title_matches == 0:
                continue

            url_matches = sum(1 for t in query_token_set if t in url_lower)

            score = 0.0
            # Strong bonus for full phrase match (rare but high-signal)
            if query_lower and query_lower in title_lower:
                score += 6.0
            if query_lower and query_lower in url_lower:
                score += 3.0

            # Emphasize title over URL
            score += 2.0 * title_matches
            score += 1.0 * url_matches

            max_possible = 6.0 + 3.0 + (2.0 * len(query_token_set)) + (1.0 * len(query_token_set))
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

    def search_with_advanced_query_and_download(
        self,
        search_query: SearchQuery,
        max_results: int | None = None,
        file_types: Optional[List[str]] = None,
        download: bool = True
    ) -> List[Dict[str, Any]]:
        """Run an advanced search query and download direct file results.

        This is the missing piece for callers who expect "advanced search" to
        actually save documents/images instead of only returning URLs.

        Args:
            search_query: SearchQuery with operators
            max_results: Maximum results to return
            file_types: Extensions to download (e.g. ['.pdf', '.jpg']).
                If None, downloads known document/image/media types.
            download: If False, returns results without downloading.

        Returns:
            List of dicts with keys: url, title, source, filepath, downloaded
        """
        results = self.search_with_advanced_query(search_query, max_results=max_results)

        if file_types is None:
            allowed_extensions = (
                self.DOCUMENT_EXTENSIONS |
                self.IMAGE_EXTENSIONS |
                self.VIDEO_EXTENSIONS |
                self.AUDIO_EXTENSIONS
            )
        else:
            allowed_extensions = {
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in file_types
            }

        output: List[Dict[str, Any]] = []
        for url, title, source in results:
            ext = self._get_file_extension(url)
            should_download = ext in allowed_extensions

            record: Dict[str, Any] = {
                'url': url,
                'title': title,
                'source': source,
                'filepath': None,
                'downloaded': False,
                'extension': ext,
            }

            if download and should_download:
                path = self.download_file(url)
                if path:
                    record['filepath'] = str(path)
                    record['downloaded'] = True

            output.append(record)

        return output
    
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

    def search_entity_advanced_and_download(
        self,
        entity_name: str,
        entity_type: str,
        max_results_per_query: int = 10,
        file_types: Optional[List[str]] = None,
        download: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Advanced entity search + download direct file results.

        Args:
            entity_name: Entity name
            entity_type: Entity type (PERSON, ORG, etc.)
            max_results_per_query: Max results per query
            file_types: Extensions to download. If None, downloads known doc/image/media types.
            download: If False, returns results without downloading.

        Returns:
            Dict mapping query string -> list of result dicts with filepath info.
        """
        # Build advanced queries
        queries = self.build_queries_for_entity(entity_name, entity_type)
        query_strings = [q.build() for q in queries]

        results_map = self.parallel_search(query_strings, max_results_per_query)

        if file_types is None:
            allowed_extensions = (
                self.DOCUMENT_EXTENSIONS |
                self.IMAGE_EXTENSIONS |
                self.VIDEO_EXTENSIONS |
                self.AUDIO_EXTENSIONS
            )
        else:
            allowed_extensions = {
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in file_types
            }

        out: Dict[str, List[Dict[str, Any]]] = {}
        for query_str, results in results_map.items():
            converted: List[Dict[str, Any]] = []
            for url, title, source in results:
                ext = self._get_file_extension(url)
                should_download = ext in allowed_extensions
                record: Dict[str, Any] = {
                    'url': url,
                    'title': title,
                    'source': source,
                    'filepath': None,
                    'downloaded': False,
                    'extension': ext,
                }
                if download and should_download:
                    path = self.download_file(url)
                    if path:
                        record['filepath'] = str(path)
                        record['downloaded'] = True
                converted.append(record)

            out[query_str] = converted

        return out
    
    # ==================== FILE DOWNLOAD METHODS ====================
    
    def download_file(
        self, 
        url: str, 
        filename: Optional[str] = None,
        timeout: int = 60
    ) -> Optional[Path]:
        """
        Download a file from URL and save it to the appropriate directory.
        
        Args:
            url: URL to download
            filename: Optional custom filename (auto-generated if not provided)
            timeout: Request timeout in seconds
            
        Returns:
            Path to saved file, or None if download failed
        """
        try:
            # Dedupe by URL via manifest
            if self._manifest is not None:
                existing = self._manifest.get_by_url(url)
                if existing and existing.path and Path(existing.path).exists():
                    return Path(existing.path)

            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Determine file extension and category
            content_type = response.headers.get('Content-Type', '')
            ext = self._get_file_extension(url, content_type)
            
            # Determine save directory
            save_dir = self._get_save_directory(ext)
            
            # Generate filename if not provided
            if not filename:
                parsed_url = urlparse(url)
                filename = Path(parsed_url.path).name
                if not filename or filename == '/':
                    # Generate from URL hash
                    filename = f"download_{abs(hash(url)) % 10000000}{ext}"
                elif not Path(filename).suffix:
                    filename = f"{filename}{ext}"
            
            filename = self._sanitize_filename(filename)
            filepath = save_dir / filename

            # Download to a temporary file first (so we can content-dedupe safely).
            tmp_path = filepath.with_suffix(filepath.suffix + ".part")
            
            # Download with streaming for large files
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            import hashlib
            h = hashlib.sha256()
            downloaded = 0

            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        h.update(chunk)

            digest = h.hexdigest()

            # Dedupe by content hash
            if self._manifest is not None:
                existing_by_hash = self._manifest.get_by_hash(digest)
                if existing_by_hash and existing_by_hash.path and Path(existing_by_hash.path).exists():
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._manifest.upsert(url=url, content_hash=digest, path=Path(existing_by_hash.path))
                    return Path(existing_by_hash.path)

            # Use stable name to avoid collisions
            stable_name = f"{digest[:10]}_{filepath.name}"
            filepath = save_dir / stable_name
            if filepath.exists():
                # If it exists but isn't in manifest yet, keep it.
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                tmp_path.replace(filepath)

            if self._manifest is not None:
                try:
                    self._manifest.upsert(url=url, content_hash=digest, path=filepath)
                except Exception:
                    pass
            
            logger.info(f"Downloaded: {filepath} ({downloaded} bytes)")
            return filepath
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
        except IOError as e:
            logger.error(f"Failed to save file from {url}: {e}")
            return None
    
    def download_files_parallel(
        self,
        urls: List[str],
        max_workers: int = 5
    ) -> Dict[str, Optional[Path]]:
        """
        Download multiple files in parallel.
        
        Args:
            urls: List of URLs to download
            max_workers: Maximum parallel downloads
            
        Returns:
            Dictionary mapping URL to saved filepath (or None if failed)
        """
        results = {}
        
        if not urls:
            return results
        
        logger.info(f"Starting parallel download of {len(urls)} files...")
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as executor:
            future_to_url = {
                executor.submit(self.download_file, url): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    filepath = future.result(timeout=120)
                    results[url] = filepath
                except Exception as e:
                    logger.error(f"Parallel download failed for {url}: {e}")
                    results[url] = None
        
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Parallel download completed: {successful}/{len(urls)} successful")
        
        return results
    
    def extract_downloadable_links(
        self, 
        page_url: str,
        file_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract downloadable file links from a webpage.
        
        Args:
            page_url: URL of the page to scan
            file_types: Filter by extensions (e.g., ['.pdf', '.jpg']). None = all supported types
            
        Returns:
            List of file URLs found on the page
        """
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(page_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all supported extensions if not specified
            if file_types is None:
                all_extensions = (
                    self.DOCUMENT_EXTENSIONS | 
                    self.IMAGE_EXTENSIONS | 
                    self.VIDEO_EXTENSIONS | 
                    self.AUDIO_EXTENSIONS
                )
            else:
                all_extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in file_types}
            
            file_urls = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Make absolute URL
                full_url = urljoin(page_url, href)
                
                # Check extension
                ext = self._get_file_extension(full_url)
                if ext in all_extensions:
                    file_urls.append(full_url)
            
            # Also check img tags for images
            if file_types is None or any(ext in self.IMAGE_EXTENSIONS for ext in all_extensions):
                for img in soup.find_all('img', src=True):
                    src = img['src']
                    full_url = urljoin(page_url, src)
                    ext = self._get_file_extension(full_url)
                    if ext in self.IMAGE_EXTENSIONS:
                        file_urls.append(full_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in file_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            logger.info(f"Found {len(unique_urls)} downloadable files on {page_url}")
            return unique_urls
            
        except Exception as e:
            logger.error(f"Failed to extract links from {page_url}: {e}")
            return []
    
    def search_and_download(
        self, 
        query: str, 
        file_types: Optional[List[str]] = None,
        download: bool = True,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for content and optionally download matching files.
        
        Args:
            query: Search query
            file_types: Filter by file extensions (e.g., ['.pdf', '.docx'])
            download: Whether to download found files
            max_results: Maximum number of results
            
        Returns:
            List of result dictionaries with url, title, source, and filepath
        """
        # Build file type query if specified
        search_query = query
        if file_types:
            # Add filetype operators to query
            type_operators = []
            for ft in file_types:
                ft_clean = ft.lstrip('.')
                type_operators.append(f'filetype:{ft_clean}')
            
            if len(type_operators) == 1:
                search_query = f"{query} {type_operators[0]}"
            else:
                # Use OR for multiple file types
                search_query = f"{query} ({' OR '.join(type_operators)})"
        
        # Execute search
        results = self.search_multi(search_query, max_results)
        
        output = []
        for url, title, source in results:
            result = {
                'url': url, 
                'title': title, 
                'source': source,
                'filepath': None,
                'downloaded': False
            }
            
            # Check if URL matches file types
            if file_types:
                ext = self._get_file_extension(url)
                valid_extensions = {ft.lower() if ft.startswith('.') else f'.{ft.lower()}' for ft in file_types}
                if ext not in valid_extensions:
                    # URL doesn't directly point to file type, skip download
                    output.append(result)
                    continue
            
            if download:
                filepath = self.download_file(url)
                if filepath:
                    result['filepath'] = str(filepath)
                    result['downloaded'] = True
            
            output.append(result)
        
        downloaded_count = sum(1 for r in output if r['downloaded'])
        logger.info(f"Search and download completed: {len(output)} results, {downloaded_count} downloaded")
        
        return output
    
    def search_and_download_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for documents and download them.
        
        Args:
            query: Search query
            document_types: Document extensions to search for (default: PDF, DOCX, etc.)
            max_results: Maximum results
            
        Returns:
            List of result dictionaries
        """
        if document_types is None:
            document_types = ['.pdf', '.docx', '.doc', '.xlsx', '.pptx']
        
        return self.search_and_download(
            query=query,
            file_types=document_types,
            download=True,
            max_results=max_results
        )
    
    def search_and_download_images(
        self,
        query: str,
        image_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for images and download them.
        
        Args:
            query: Search query
            image_types: Image extensions to search for (default: JPG, PNG, etc.)
            max_results: Maximum results
            
        Returns:
            List of result dictionaries
        """
        if image_types is None:
            image_types = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        
        return self.search_and_download(
            query=query,
            file_types=image_types,
            download=True,
            max_results=max_results
        )
    
    def crawl_and_download(
        self,
        start_url: str,
        file_types: Optional[List[str]] = None,
        max_files: int = 50
    ) -> Dict[str, Optional[Path]]:
        """
        Crawl a webpage and download all matching files.
        
        Args:
            start_url: URL to start crawling from
            file_types: File types to download (None = all supported)
            max_files: Maximum files to download
            
        Returns:
            Dictionary mapping URL to filepath
        """
        logger.info(f"Crawling {start_url} for downloadable files...")
        
        # Extract file links from page
        file_urls = self.extract_downloadable_links(start_url, file_types)
        
        # Limit to max_files
        file_urls = file_urls[:max_files]
        
        if not file_urls:
            logger.info(f"No downloadable files found on {start_url}")
            return {}
        
        logger.info(f"Found {len(file_urls)} files to download")
        
        # Download all files
        return self.download_files_parallel(file_urls)
