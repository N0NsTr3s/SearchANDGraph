"""
Main entry point for the Knowledge Graph Extractor.

This script orchestrates the web crawling, NLP processing, and graph visualization
to build a knowledge graph from web content.
"""
import asyncio
import pickle
from pathlib import Path
import networkx as nx
from typing import cast, Dict, Tuple, List, Any
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from utils.config import AppConfig
from scraper.crawler import WebCrawler
from processor.nlp_processor import NLPProcessor
from processor.graph_builder import KnowledgeGraph
from vizualization.visualizer import GraphVisualizer
from scraper.source_discovery import MultiSourceDiscovery
from utils.logger import setup_logger
from scraper.scan_manager import get_scan_paths
from scraper.web_search import WebSearcher

logger = setup_logger("main")


def configure_asyncio_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Suppress benign websocket close errors emitted after shutdown."""
    previous_handler = loop.get_exception_handler()

    def is_benign_websocket_error(exc: BaseException | None) -> bool:
        if exc is None:
            return False
        if isinstance(exc, (ConnectionClosedError, ConnectionClosedOK)):
            return True
        message = str(exc).lower()
        return "websocket" in message and "closed" in message

    def handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
        exception = context.get("exception")
        if is_benign_websocket_error(exception):
            logger.debug("Suppressed websocket exception during shutdown: %s", exception)
            return

        future = context.get("future")
        if isinstance(future, asyncio.Future):  # type: ignore[arg-type]
            if future.done():
                try:
                    fut_exception = future.exception()
                except asyncio.CancelledError:
                    logger.debug("Suppressed cancelled websocket task during shutdown")
                    return
                except Exception as future_error:  # pragma: no cover - defensive safety
                    fut_exception = future_error

                if is_benign_websocket_error(fut_exception):
                    logger.debug("Suppressed websocket future exception during shutdown: %s", fut_exception)
                    return

        if previous_handler is not None:
            previous_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def get_optimal_tab_count() -> int:
    """
    Calculate optimal number of concurrent tabs based on available system memory.
    
    Formula: max(5, min(20, available_gb * 2))
    - Minimum 5 tabs to maintain reasonable speed
    - Maximum 20 tabs to avoid browser overhead
    - 2 tabs per GB of available memory
    
    Returns:
        Optimal number of concurrent tabs
    """
    try:
        import psutil
        
        # Get available memory in GB
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        
        # Calculate optimal tabs: 2 per GB of available memory
        optimal_tabs = int(available_gb * 2)
        
        # Clamp between 5 and 20
        optimal_tabs = max(5, min(20, optimal_tabs))
        
        logger.info(f"System memory: {mem.total / (1024**3):.1f} GB total, {available_gb:.1f} GB available")
        logger.info(f"Optimal tab count: {optimal_tabs} (formula: max(5, min(20, {available_gb:.1f} * 2)))")
        
        return optimal_tabs
        
    except ImportError:
        logger.warning("psutil not installed - using default tab count of 10")
        logger.info("To enable dynamic tab allocation, install psutil: pip install psutil")
        return 10
    except Exception as e:
        logger.warning(f"Failed to detect system memory: {e} - using default tab count of 10")
        return 10


async def save_checkpoint(
    knowledge_graph: KnowledgeGraph,
    visited_urls: set,
    crawled_count: int,
    checkpoint_file: str
):
    """
    Save a checkpoint of the current crawling progress.
    
    Args:
        knowledge_graph: Current knowledge graph
        visited_urls: Set of visited URLs
        crawled_count: Number of pages crawled
        checkpoint_file: File to save checkpoint to
    """
    try:
        checkpoint_data = {
            'graph': knowledge_graph,
            'visited_urls': visited_urls,
            'crawled_count': crawled_count,
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {crawled_count} pages processed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


async def load_checkpoint(checkpoint_file: str) -> dict | None:
    """
    Load a checkpoint if it exists.
    
    Args:
        checkpoint_file: File to load checkpoint from
        
    Returns:
        Checkpoint data or None if not found
    """
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Checkpoint loaded: {data['crawled_count']} pages previously processed")
        return data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def check_relevance_parallel(pages_data, nlp_processor, query, config):
    """
    Check relevance of multiple pages in parallel.
    
    Args:
        pages_data: List of (content, url, source_url) tuples
        nlp_processor: NLP processor instance
        query: Search query
        config: Configuration object
        
    Returns:
        List of (translated_text, url, source_url) tuples for relevant pages
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    relevant_pages = []
    irrelevant_urls = []
    
    logger.info(f"Checking relevance for {len(pages_data)} pages in parallel...")
    
    with ThreadPoolExecutor(max_workers=config.nlp.max_workers) as executor:
        # Submit all relevance checks
        future_to_page = {
            executor.submit(
                nlp_processor.is_content_relevant, 
                content, 
                query, 
                0.1
            ): (content, url, source_url)
            for content, url, source_url in pages_data
        }
        
        # Process results as they complete
        for idx, future in enumerate(as_completed(future_to_page), 1):
            content, url, source_url = future_to_page[future]
            try:
                is_relevant, relevance_score, translated_text = future.result()
                
                if is_relevant:
                    logger.info(f"  OK Page {idx} is relevant (score: {relevance_score:.2f}): {url}")
                    relevant_pages.append((translated_text, url, source_url))
                else:
                    logger.info(f"  X Page {idx} not relevant (score: {relevance_score:.2f}): {url}")
                    irrelevant_urls.append(url)
            except Exception as e:
                logger.warning(f"  Failed to check relevance for {url}: {e}")
                irrelevant_urls.append(url)
    
    logger.info(f"Relevance check complete: {len(relevant_pages)} relevant, {len(irrelevant_urls)} irrelevant")
    return relevant_pages, irrelevant_urls


async def main(
    query: str | None = None,
    start_url: str | None = None,
    max_pages: int | None = None,
    add_timestamp: bool = False,
    base_dir: str = "scans",
    browser_headless: bool | None = None,
    enable_web_search: bool | None = None,
    web_search_download_pdfs: bool | None = None,
    preferred_sources: list[str] | None = None,
    blacklisted_sources: list[str] | None = None,
    viz_max_nodes: int | None = None,
    viz_min_edge_confidence: float | None = None,
    viz_remove_isolated_nodes: bool | None = None,
    enable_phase2: bool | None = None,
    phase2_max_pages: int | None = None,
    phase2_concurrent_tabs: int | None = None,
    document_min_relevance: float | None = None,
    downloads_prune_irrelevant: bool | None = None,
    downloads_prune_mode: str | None = None,
    web_search_max_pdf_downloads: int | None = None,
    web_search_min_relevance: float | None = None,
    nlp_min_confidence: float | None = None,
    nlp_min_relation_confidence: float | None = None,
):
    """Run a full scan and build visualizations.

    Args:
        query: Search query string. If None, uses config default.
        start_url: Optional seed URL. If None, uses config default.
        max_pages: Optional crawl budget. If None, uses config default.
        add_timestamp: If True, creates a unique scan directory per run.
    """
    loop = asyncio.get_running_loop()
    configure_asyncio_exception_handler(loop)

    # Load configuration
    config = AppConfig.default()
    
    # Set optimal tab count based on available memory
    optimal_tabs = get_optimal_tab_count()
    config.crawler.concurrent_tabs = optimal_tabs
    
    # Override config with provided parameters (API-friendly)
    if query is not None:
        config.crawler.query = query
    if start_url is not None:
        config.crawler.start_url = start_url
    if max_pages is not None:
        config.crawler.max_pages = max_pages

    # Optional UI overrides
    if browser_headless is not None:
        config.crawler.browser_headless = browser_headless
    if enable_web_search is not None:
        config.crawler.enable_web_search = enable_web_search
    if web_search_download_pdfs is not None:
        config.crawler.web_search_download_pdfs = web_search_download_pdfs

    # Phase 2 overrides
    if enable_phase2 is not None:
        config.crawler.enable_phase2 = enable_phase2
    if phase2_max_pages is not None:
        config.crawler.phase2_max_pages = phase2_max_pages
    if phase2_concurrent_tabs is not None:
        config.crawler.phase2_concurrent_tabs = phase2_concurrent_tabs

    # Document handling overrides
    if document_min_relevance is not None:
        config.crawler.document_min_relevance = document_min_relevance
    if downloads_prune_irrelevant is not None:
        config.crawler.downloads_prune_irrelevant = downloads_prune_irrelevant
    if downloads_prune_mode is not None:
        mode = str(downloads_prune_mode).strip().lower()
        if mode in {"move", "delete"}:
            config.crawler.downloads_prune_mode = mode
    if web_search_max_pdf_downloads is not None:
        config.crawler.web_search_max_pdf_downloads = web_search_max_pdf_downloads
    if web_search_min_relevance is not None:
        config.crawler.web_search_min_relevance = web_search_min_relevance

    # Optional config overrides driven by the desktop UI
    # Only apply merging when the UI provided at least one meaningful (non-empty)
    # preferred or blacklisted source. Empty lists/strings mean "no override".
    prefer_has_items = bool(preferred_sources and any(s and s.strip() for s in preferred_sources))
    blacklist_has_items = bool(blacklisted_sources and any(s and s.strip() for s in blacklisted_sources))

    if prefer_has_items or blacklist_has_items:
        current_sources = list(getattr(config.crawler, 'sources', []) or [])
        deny = {s.strip().lower() for s in (blacklisted_sources or []) if s and s.strip()}
        prefer = [s.strip() for s in (preferred_sources or []) if s and s.strip()]

        # Remove blacklisted
        filtered = [s for s in current_sources if s.strip().lower() not in deny]

        # Move preferred to front (keep order, avoid duplicates)
        preferred_ordered: list[str] = []
        for s in prefer:
            if s.strip().lower() in deny:
                continue
            if s in preferred_ordered:
                continue
            preferred_ordered.append(s)

        remainder = [s for s in filtered if s not in preferred_ordered]
        merged = preferred_ordered + remainder
        if merged:
            config.crawler.sources = merged

    if viz_max_nodes is not None:
        config.visualization.max_nodes = viz_max_nodes
    if viz_min_edge_confidence is not None:
        config.visualization.min_edge_confidence = viz_min_edge_confidence
    if viz_remove_isolated_nodes is not None:
        config.visualization.remove_isolated_nodes = viz_remove_isolated_nodes

    # Confidence-related overrides (NLP)
    if nlp_min_confidence is not None:
        config.nlp.min_confidence = nlp_min_confidence
    if nlp_min_relation_confidence is not None:
        config.nlp.min_relation_confidence = nlp_min_relation_confidence
    
    # IMPORTANT: Get query-specific paths
    # This creates a separate folder for each query to avoid merging different scans
    scan_paths = get_scan_paths(config.crawler.query, base_dir=base_dir, add_timestamp=add_timestamp)
    
    # Update configuration to use query-specific paths
    config.cache.cache_dir = str(scan_paths['cache_dir'])
    config.crawler.checkpoint_file = str(scan_paths['checkpoint_file'])
    config.graph.graph_file = str(scan_paths['graph_file'])
    config.visualization.output_file = str(scan_paths['viz_file'])
    config.visualization.interactive_file = str(scan_paths['interactive_viz_file'])
    
    logger.info("=" * 60)
    logger.info("Starting Knowledge Graph Extractor")
    logger.info("=" * 60)
    logger.info(f"Query: {config.crawler.query}")
    logger.info(f"Scan directory: {scan_paths['scan_dir']}")
    logger.info(f"Start URL: {config.crawler.start_url}")
    logger.info(f"Max pages: {config.crawler.max_pages}")
    logger.info(f"Concurrent tabs: {config.crawler.concurrent_tabs}")
    logger.info(f"BFS enabled: {config.crawler.use_bfs}")
    logger.info(f"Caching enabled: {config.cache.enabled}")
    logger.info(f"Cache directory: {config.cache.cache_dir}")
    logger.info(f"Parallel NLP: {config.nlp.parallel_processing}")
    logger.info(f"Confidence scoring: {config.nlp.enable_confidence_scoring}")
    logger.info(f"Multi-source discovery: {config.crawler.multi_source_discovery}")
    logger.info(f"Phase 2 enabled: {getattr(config.crawler, 'enable_phase2', True)}")
    
    # Graph configuration
    distance_info = f"{config.graph.max_query_distance} hops" if config.graph.max_query_distance else "unlimited"
    logger.info(f"Entity disambiguation: {config.graph.enable_disambiguation}")
    logger.info(f"Max distance from query: {distance_info}")
    logger.info("=" * 60)
    
    # Initialize components with cache support
    # Respect UI-provided preferred/blacklist lists if they are meaningful (non-empty).
    ui_prefer = preferred_sources if preferred_sources else None
    ui_blacklist = blacklisted_sources if blacklisted_sources else None

    if ui_prefer is not None:
        prefer_clean = [s.strip() for s in ui_prefer if s and s.strip()]
        if prefer_clean:
            config.crawler.preferred_sources = prefer_clean

    if ui_blacklist is not None:
        blacklist_clean = [s.strip() for s in ui_blacklist if s and s.strip()]
        if blacklist_clean:
            config.crawler.blacklisted_sources = blacklist_clean

    crawler = WebCrawler(config.crawler, config.cache)
    nlp_processor = NLPProcessor(config.nlp, config.cache)
    
    # Initialize KnowledgeGraph with entity disambiguation if enabled
    if config.graph.enable_disambiguation:
        logger.info(f"Entity disambiguation enabled (threshold={config.graph.similarity_threshold})")
        knowledge_graph = KnowledgeGraph(
            use_disambiguation=True,
            similarity_threshold=config.graph.similarity_threshold
        )
    else:
        knowledge_graph = KnowledgeGraph(use_disambiguation=False)
    
    visualizer = GraphVisualizer(config.visualization)
    
    # Initialize multi-source discovery if enabled
    source_discovery = None
    if config.crawler.multi_source_discovery:
        source_discovery = MultiSourceDiscovery(
            config.crawler.sources,
            preferred=getattr(config.crawler, 'preferred_sources', None),
            blacklisted=getattr(config.crawler, 'blacklisted_sources', None)
        )
    
    # Try to load existing graph if incremental building is enabled
    if config.graph.enable_incremental:
        if knowledge_graph.load_from_file(config.graph.graph_file, merge=True):
            logger.info("Loaded existing knowledge graph for incremental building")
    
    # Try to load checkpoint if enabled
    checkpoint = None
    if config.crawler.enable_checkpoints:
        checkpoint = await load_checkpoint(config.crawler.checkpoint_file)
        if checkpoint:
            knowledge_graph = checkpoint['graph']
            logger.info("Resumed from checkpoint")
    
    # Initialize crawler with query as the starting entity
    # This marks the query as the root for relationship-based discovery
    query_words = [word for word in config.crawler.query.split() if len(word) > 2]
    crawler.add_entities([config.crawler.query] + query_words, mark_as_query=True)
    
    # Expand query with synonyms if enabled
    if config.nlp.enable_query_expansion:
        expanded_query = nlp_processor.expand_query_with_synonyms(config.crawler.query)
        logger.info(f"Expanded query to include: {expanded_query}")
    else:
        expanded_query = {config.crawler.query}
    
    logger.info(f"Initialized crawler with query entity: {config.crawler.query}")
    
    # Determine starting URLs using web search if enabled
    starting_urls = []
    
    if config.crawler.enable_web_search:
        logger.info("=" * 60)
        logger.info("Discovering URLs via web search...")
        logger.info("=" * 60)
        
        
        # Store all downloads under the scan's single downloads directory
        search_download_dir = str(scan_paths['scan_dir'] / "downloads")
        Path(search_download_dir).mkdir(parents=True, exist_ok=True)
        searcher = WebSearcher(
            max_results=config.crawler.web_search_max_results,
            download_dir=search_download_dir
        )
        crawler.set_web_searcher(searcher)

        # Run parallel searches for expanded query set (always includes base query)
        search_queries = sorted(
            expanded_query,
            key=lambda q: (0 if q == config.crawler.query else 1, q.lower())
        )
        search_results = []
        seen_urls = set()

        query_results_map = searcher.parallel_search(
            search_queries,
            max_results_per_query=config.crawler.web_search_max_results
        )

        total_results = 0
        for query_str in search_queries:
            results_for_query = query_results_map.get(query_str, [])
            total_results += len(results_for_query)
            logger.info(f"Query '{query_str}' returned {len(results_for_query)} results")
            for url, title, source in results_for_query:
                normalized_url = url.lower().rstrip('/')
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    search_results.append((url, title, source))

        logger.info(f"Aggregated {len(search_results)} unique search results from {total_results} total hits")
        
        # Filter and score by relevance
        scored_results = searcher.filter_urls_by_relevance(
            search_results, 
            config.crawler.query,
            min_score=config.crawler.web_search_min_relevance
        )

        # Optionally persist PDFs to disk from web search results.
        # This complements crawler document handling (which downloads PDFs it actually visits).
        if getattr(config.crawler, 'web_search_download_pdfs', False):
            max_pdf_downloads = int(getattr(config.crawler, 'web_search_max_pdf_downloads', 0) or 0)
            if max_pdf_downloads > 0:
                try:
                    from urllib.parse import urlparse

                    def is_pdf_url(candidate_url: str) -> bool:
                        try:
                            parsed = urlparse(candidate_url)
                            return Path((parsed.path or '').lower()).suffix == '.pdf'
                        except Exception:
                            return False

                    pdf_urls: list[str] = []
                    if scored_results:
                        for url, _title, _source, _score in scored_results:
                            if is_pdf_url(url):
                                pdf_urls.append(url)

                    # If we didn't get enough direct PDF URLs, run a PDF-biased query.
                    if len(pdf_urls) < max_pdf_downloads:
                        suffix = str(getattr(config.crawler, 'web_search_pdf_query_suffix', ''))
                        pdf_query = f"{config.crawler.query}{suffix}" if suffix else config.crawler.query
                        pdf_results = searcher.search_multi(pdf_query, max_results=min(config.crawler.web_search_max_results, 50))
                        # Filter the PDF-biased results by relevance too.
                        try:
                            pdf_scored = searcher.filter_urls_by_relevance(
                                pdf_results,
                                config.crawler.query,
                                min_score=config.crawler.web_search_min_relevance
                            )
                            for url, _title, _source, _score in pdf_scored:
                                if is_pdf_url(url):
                                    pdf_urls.append(url)
                        except Exception:
                            for url, _title, _source in pdf_results:
                                if is_pdf_url(url):
                                    pdf_urls.append(url)

                    # De-dupe while preserving order
                    seen_pdf = set()
                    unique_pdf_urls: list[str] = []
                    for url in pdf_urls:
                        key = url.lower().rstrip('/')
                        if key in seen_pdf:
                            continue
                        seen_pdf.add(key)
                        unique_pdf_urls.append(url)

                    to_download = unique_pdf_urls[:max_pdf_downloads]
                    if to_download:
                        logger.info(f"Downloading up to {len(to_download)} PDF(s) from web search into: {search_download_dir}")
                        download_results = searcher.download_files_parallel(to_download, max_workers=5)
                        downloaded_ok = sum(1 for v in download_results.values() if v is not None)
                        logger.info(f"Web-search PDF download complete: {downloaded_ok}/{len(to_download)} successful")
                    else:
                        logger.info("No PDF URLs found in web search results to download")
                except Exception as e:
                    logger.warning(f"Web-search PDF download step failed (continuing without PDFs): {e}")
        
        if scored_results:
            logger.info(f"Filtered to {len(scored_results)} relevant URLs (score >= {config.crawler.web_search_min_relevance})")
            logger.info("Top search results:")
            for url, title, source, score in scored_results[:10]:
                logger.info(f"  [{score:.2f}] {title}")
                logger.info(f"           {url}")
                starting_urls.append(url)
        else:
            logger.warning("No relevant URLs found via search!")
    
    # Add start_url if not ignoring it
    if not config.crawler.ignore_start_url and config.crawler.start_url:
        if config.crawler.start_url not in starting_urls:
            logger.info(f"Adding configured start URL: {config.crawler.start_url}")
            starting_urls.insert(0, config.crawler.start_url)  # Add at beginning
        else:
            logger.info(f"Start URL already in search results: {config.crawler.start_url}")
    elif config.crawler.ignore_start_url:
        logger.info("Ignoring configured start_url, using only search results")
    
    if not starting_urls:
        logger.error("No starting URLs available! Enable web search or provide a start_url.")
        return
    
    logger.info(f"Starting crawl with {len(starting_urls)} seed URLs")
    
    pipeline = None

    try:
        # Start the browser once for the entire crawling session
        await crawler.start()
        
        try:
            # Crawl web pages with BFS, starting from all seed URLs
            logger.info("Starting BFS web crawling from search results...")
            logger.info(f"Seed URLs: {starting_urls[:5]}{'...' if len(starting_urls) > 5 else ''}")
            
            # Start with first URL, then immediately seed the queue with others
            # The crawler will prioritize them by relevance automatically
            primary_url = starting_urls[0]
            additional_seeds = starting_urls[1:min(len(starting_urls), config.crawler.max_pages)]
            
            crawled_contents, visited_urls, link_graph = await crawler.crawl(
                primary_url,
                config.crawler.query,
                config.crawler.max_pages,
                additional_seed_urls=additional_seeds  # Pass additional URLs to crawler
            )
            
            # crawled_contents is now a list of (content, url) tuples
            # Extract into separate lists while preserving the association
            content_list = [content for content, url in crawled_contents]
            url_list = [url for content, url in crawled_contents]
            
            logger.info(f"Crawling complete. Visited {len(visited_urls)} pages.")
            logger.info(f"Extracted content from {len(crawled_contents)} pages.")
            logger.info(f"Link graph contains {len(link_graph)} nodes.")
            
            if len(content_list) == 0:
                logger.error("No content was extracted from any pages!")
                logger.error("This could be due to:")
                logger.error("  1. Cookie banners blocking content")
                logger.error("  2. JavaScript-heavy pages not loading fully")
                logger.error("  3. Content extraction filters being too strict")
                logger.error("  4. Network/connection issues")
                logger.error("Please check the logs above for specific page errors.")
                return
            
            # ========== CANONICAL PIPELINE PROCESSING ==========
            logger.info("=" * 60)
            logger.info("Using Canonical Pipeline: Crawl -> Translate -> Process -> Graph")
            logger.info("=" * 60)
            
            from processor.canonical_pipeline import CanonicalPipeline
            
            # Initialize canonical pipeline with parallel processing
            pipeline = CanonicalPipeline(
                nlp_processor=nlp_processor,
                knowledge_graph=knowledge_graph,
                query=config.crawler.query,
                target_language="en",
                max_workers=config.nlp.max_workers  # Use max_workers from NLP config
            )
            
            # Stage 1: Enqueue all crawled content for translation
            logger.info("Stage 1: Enqueuing crawled content...")
            for content, url in crawled_contents:
                await pipeline.enqueue_content(
                    url=url,
                    raw_content=content,
                    content_type='webpage',
                    source_url=url
                )
            
            # Run full pipeline: Translate -> Process -> Aggregate
            await pipeline.run_full_pipeline()
            
            # Get pipeline statistics
            pipeline_stats = pipeline.get_statistics()
            logger.info("=" * 60)
            logger.info("Canonical Pipeline Complete")
            logger.info(f"  Processed: {pipeline_stats['total_processed']} pages")
            logger.info(f"  Added to graph: {pipeline_stats['total_added_to_graph']} pages")
            logger.info(f"  Skipped (irrelevant): {pipeline_stats['skipped_irrelevant']}")
            logger.info("=" * 60)
            
            # Track URLs for statistics
            relevant_urls = [url for _, url in crawled_contents[:pipeline_stats['total_added_to_graph']]]
            irrelevant_urls = [url for _, url in crawled_contents[pipeline_stats['total_added_to_graph']:]]
            
            # Log relevance statistics
            logger.info("=" * 60)
            logger.info("Content Relevance Summary:")
            logger.info(f"  Relevant pages: {len(relevant_urls)}")
            logger.info(f"  Irrelevant pages (skipped): {len(irrelevant_urls)}")
            logger.info(f"  Relevance rate: {len(relevant_urls)/max(len(crawled_contents), 1)*100:.1f}%")
            
            if irrelevant_urls:
                logger.info("  Irrelevant URLs:")
                for url in irrelevant_urls[:5]:  # Show first 5
                    logger.info(f"    - {url}")
                if len(irrelevant_urls) > 5:
                    logger.info(f"    ... and {len(irrelevant_urls) - 5} more")
            logger.info("=" * 60)
            
            # Log entity extraction summary
            logger.info("=" * 60)
            logger.info("Entity Extraction Summary:")
            current_stats = knowledge_graph.get_statistics()
            logger.info(f"  Total entities extracted: {current_stats['unique_entities']}")
            logger.info(f"  Total relations extracted: {current_stats['unique_relations']}")
            logger.info(f"  Graph nodes: {current_stats['nodes']}")
            logger.info(f"  Graph edges: {current_stats['edges']}")
            if current_stats['edges'] == 0:
                logger.warning("  WARNING: No relations found between entities!")
                logger.warning("     All entities are isolated. They will be filtered out during visualization.")
            logger.info("=" * 60)
            
            # Get initial statistics
            stats = knowledge_graph.get_statistics()
            logger.info("=" * 60)
            logger.info("Knowledge Graph Statistics (initial crawl):")
            logger.info(f"  Nodes: {stats['nodes']}")
            logger.info(f"  Edges: {stats['edges']}")
            logger.info(f"  Unique Entities: {stats['unique_entities']}")
            logger.info(f"  Unique Relations: {stats['unique_relations']}")
            logger.info("=" * 60)
            
            # SMART ENTITY DISCOVERY: REMOVED - Phase 2 handles targeted discovery using canonical entities
            
            # ========== TRANSITION: ANALYZE PHASE 1 GRAPH ==========
            logger.info("=" * 60)
            logger.info("TRANSITION: Analyzing Phase 1 Graph for High-Value Entities")
            logger.info("=" * 60)
            
            # Use canonical pipeline to identify high-value entities from Phase 1 graph
            high_value_list = pipeline.analyze_and_reprompt(max_entities=3)
            high_value_entities = [
                (entity, entity_type, degree)
                for entity, entity_type, degree in high_value_list
            ]
            
            logger.info(f"Identified {len(high_value_entities)} high-value entities for deep dive:")
            for entity, entity_type, degree in high_value_entities:
                logger.info(f"  - {entity} ({entity_type}, degree: {degree})")
            
            # ========== PHASE 2: TARGETED DEEP DIVE ==========
            logger.info("=" * 60)
            logger.info("PHASE 2: Targeted Deep Dive with Iterative Search")
            logger.info("=" * 60)
            
            # Track Phase 2 budget separately
            phase2_pages_crawled = 0
            phase2_max = config.crawler.phase2_max_pages

            # Normalize visited set for reliable de-dupe across Phase 1/2.
            visited_urls_normalized = {crawler._normalize_url(u) for u in visited_urls}
            
            logger.info(f"Phase 2 Budget:")
            logger.info(f"  Phase 1 pages crawled: {len(visited_urls)}")
            logger.info(f"  Phase 2 budget: {phase2_max} pages {'(unlimited)' if phase2_max == 0 else ''}")
            logger.info(f"  High-value entities to explore: {len(high_value_entities)}")
            
            # Check if Phase 2 should run. Requires config flag plus budget and entities.
            should_run_phase2 = (
                getattr(config.crawler, 'enable_phase2', True) and
                len(high_value_entities) > 0 and
                (phase2_max == 0 or phase2_pages_crawled < phase2_max)
            )
            
            if should_run_phase2:
                from processor.iterative_search_orchestrator import IterativeSearchOrchestrator
                # Browser is already running from Phase 1
                logger.info("Using existing browser session for Phase 2 deep-dive...")

                # Loop through each high-value entity
                for entity_idx, (entity_name, entity_type, entity_degree) in enumerate(high_value_entities, 1):
                    # Check remaining Phase 2 budget before each entity
                    if phase2_max > 0 and phase2_pages_crawled >= phase2_max:
                        logger.info(f"Phase 2 budget exhausted ({phase2_pages_crawled}/{phase2_max} pages). Stopping deep dive.")
                        break

                    logger.info("=" * 60)
                    logger.info(f"Phase 2 Deep Dive [{entity_idx}/{len(high_value_entities)}]: {entity_name}")
                    logger.info("=" * 60)
                    logger.info(f"  Entity type: {entity_type}")
                    logger.info(f"  Graph degree: {entity_degree}")
                    logger.info(f"  Phase 2 pages crawled so far: {phase2_pages_crawled}/{phase2_max if phase2_max > 0 else 'unlimited'}")

                    # Build context from graph
                    entity_context = {}
                    # Ensure entity_name_for_search is always defined to avoid UnboundLocalError
                    entity_name_for_search = entity_name

                    if knowledge_graph.graph.has_node(entity_name):
                        # Get neighbors sorted by their degree (most important first)
                        neighbors = list(knowledge_graph.graph.neighbors(entity_name))
                        neighbors.sort(key=lambda n: knowledge_graph.graph.degree(n), reverse=True)

                        # Use display names when available for better search queries
                        neighbor_names = [knowledge_graph.graph.nodes[n].get('display_name', n) for n in neighbors]

                        # Pass the original crawler query as the primary context, then neighbors
                        main_query = getattr(config.crawler, 'query', None) or ''
                        context_with_main_query = [main_query] + [name for name in neighbor_names if name.lower() != main_query.lower()]

                        entity_context['related_entities'] = context_with_main_query[:5]
                        logger.info(f"  Context for '{entity_name_for_search}': {entity_context['related_entities']}")
                    else:
                        # Node not found: use empty context and warn
                        logger.warning(f"High-value entity '{entity_name}' not present in knowledge graph. Using empty context for search.")
                        entity_context['related_entities'] = []

                    # Resolve QID -> human-readable display name for web search
                    try:
                        if knowledge_graph.graph.has_node(entity_name):
                            entity_name_for_search = knowledge_graph.graph.nodes[entity_name].get('display_name', entity_name)
                        else:
                            # Fallback: search nodes for matching qid field
                            entity_name_for_search = entity_name
                            for n, d in knowledge_graph.graph.nodes(data=True):
                                qid_val = d.get('qid') or d.get('node_qid') or d.get('id')
                                if qid_val and str(qid_val) == str(entity_name):
                                    entity_name_for_search = d.get('display_name', n)
                                    break
                    except Exception:
                        # Defensive: if anything goes wrong, fall back to original entity_name
                        entity_name_for_search = entity_name

                    try:
                        # Create orchestrator for this entity with the crawler instance
                        orchestrator = IterativeSearchOrchestrator(
                            crawler=crawler,
                            max_iterations=2,
                            max_results_per_query=5,
                            min_relevance_score=float(getattr(config.crawler, 'web_search_min_relevance', 0.15) or 0.15)
                        )

                        # Run autonomous iterative search
                        logger.info(f"Running iterative search for '{entity_name_for_search}' (orig: {entity_name})...")
                        iterations = await orchestrator.run_autonomous_search(
                            entity_name_for_search,
                            entity_type,
                            context=entity_context
                        )

                        # Get discovered URLs
                        deep_dive_urls = orchestrator.get_all_discovered_urls()
                        logger.info(f"Discovered {len(deep_dive_urls)} URLs across {len(iterations)} iterations")

                        if not deep_dive_urls:
                            logger.info(f"No URLs discovered for '{entity_name}'. Moving to next entity.")
                            continue

                        # Calculate how many URLs to crawl from this entity
                        if phase2_max > 0:
                            remaining_phase2_budget = phase2_max - phase2_pages_crawled
                            urls_to_crawl = min(len(deep_dive_urls), remaining_phase2_budget)
                        else:
                            urls_to_crawl = len(deep_dive_urls)
                        
                        logger.info(f"Crawling top {urls_to_crawl} URLs for '{entity_name}'...")

                        # Crawl discovered URLs (concurrently, multi-tab)
                        entity_crawl_count = 0
                        candidates: list[str] = []
                        meta: dict[str, tuple[str, float]] = {}

                        for url, title, _source, score in deep_dive_urls[:urls_to_crawl]:
                            normalized = crawler._normalize_url(url)
                            if normalized in visited_urls_normalized:
                                continue
                            candidates.append(url)
                            meta[url] = (title, float(score))

                        if not candidates:
                            logger.info("No new URLs to crawl for this entity (all already visited)")
                            continue

                        phase2_tabs = getattr(config.crawler, 'phase2_concurrent_tabs', None)
                        results = await crawler.fetch_urls_parallel(
                            candidates,
                            query=config.crawler.query,
                            concurrent_tabs=phase2_tabs
                        )

                        for idx, (url, result) in enumerate(zip(candidates, results), 1):
                            title, score = meta.get(url, (url, 0.0))
                            logger.info(f"  [{idx}/{len(candidates)}] Crawling: {str(title)[:60]}...")
                            logger.info(f"    URL: {url}")
                            logger.info(f"    Relevance: {score:.2f}")

                            normalized = crawler._normalize_url(url)
                            visited_urls.add(url)
                            visited_urls_normalized.add(normalized)

                            try:
                                text_content = (result or {}).get('content') if isinstance(result, dict) else None
                                if text_content and len(str(text_content).strip()) >= 100:
                                    content_type = 'pdf' if crawler._is_document_url(url) else 'webpage'
                                    before = pipeline.get_statistics().get('total_added_to_graph', 0)
                                    await pipeline.enqueue_content(
                                        url=url,
                                        raw_content=text_content,
                                        content_type=content_type,
                                        source_url=url
                                    )
                                    await pipeline.run_full_pipeline()
                                    after = pipeline.get_statistics().get('total_added_to_graph', 0)
                                    added = max(0, int(after) - int(before))

                                    entity_crawl_count += 1
                                    phase2_pages_crawled += 1
                                    if added > 0:
                                        logger.info(f"    ✓ Added to graph (+{added})")
                                    else:
                                        logger.info("    ✓ Processed (no new graph additions)")
                                else:
                                    logger.info("    ✗ Skipped (no usable content)")
                            except Exception as e:
                                logger.warning(f"    ✗ Failed to process: {e}")

                            # Check if we've hit the Phase 2 budget limit
                            if phase2_max > 0 and phase2_pages_crawled >= phase2_max:
                                logger.info(f"    Phase 2 budget limit reached ({phase2_pages_crawled}/{phase2_max})")
                                break
                        
                        logger.info(f"Entity '{entity_name}' deep dive complete: {entity_crawl_count} pages added to graph")
                        
                    except Exception as e:
                        logger.error(f"Deep dive failed for '{entity_name}': {e}")
                        logger.info("Continuing to next entity...")
                
                # Phase 2 summary
                logger.info("=" * 60)
                logger.info(f"Phase 2 Complete: Crawled {phase2_pages_crawled} additional pages")
                logger.info(f"  Total pages crawled (Phase 1 + Phase 2): {len(visited_urls)}")
                logger.info("=" * 60)
            
            else:
                if not high_value_entities:
                    logger.info("Skipping Phase 2: No high-value entities identified")
                elif phase2_max > 0 and phase2_pages_crawled >= phase2_max:
                    logger.info(f"Skipping Phase 2: Budget already exhausted")
                logger.info("=" * 60)
            
            # Get final statistics after Phase 2
            stats = knowledge_graph.get_statistics()
            logger.info("=" * 60)
            logger.info("Knowledge Graph Statistics (after Phase 2 deep dive):")
            logger.info(f"  Total nodes: {stats['nodes']}")
            logger.info(f"  Total edges: {stats['edges']}")
            logger.info(f"  Unique entities: {stats['unique_entities']}")
            logger.info(f"  Unique relations: {stats['unique_relations']}")
            logger.info("=" * 60)
            
            # Get final statistics after entity discovery
            stats = knowledge_graph.get_statistics()
            logger.info("=" * 60)
            logger.info("Knowledge Graph Statistics (after entity discovery):")
            logger.info(f"  Total nodes: {stats['nodes']}")
            logger.info(f"  Total edges: {stats['edges']}")
            logger.info(f"  Unique entities: {stats['unique_entities']}")
            logger.info(f"  Unique relations: {stats['unique_relations']}")
            logger.info("=" * 60)
            
            # ENRICHED DISCOVERY: Find more entities based on discovered nodes (reusing browser session)
            # This can be disabled via config if you want to stay strictly focused on the query
            if not config.crawler.enable_enriched_discovery:
                logger.info("Enriched discovery disabled. Skipping entity-based discovery.")
            elif config.crawler.max_pages <= len(visited_urls):
                logger.info("Page limit reached. Skipping enriched discovery.")
            else:
                logger.info("=" * 60)
                logger.info("Starting Enriched Entity Discovery...")
                logger.info("=" * 60)
                
                # Get top entities for further discovery
                all_entities = knowledge_graph.entities
                all_relations = knowledge_graph.relations
                
                # Use configuration values for entity discovery
                top_entities = nlp_processor.get_top_entities_for_discovery(
                    all_entities,
                    all_relations,
                    max_entities=15,  # Get 15 candidates, but only use top 3
                    original_query=config.crawler.query,
                    relevance_threshold=config.crawler.enriched_discovery_relevance_threshold
                )
                
                if not top_entities:
                    logger.info("No entities found for enriched discovery.")
                else:
                    # Calculate how many additional pages we can crawl
                    remaining_budget = config.crawler.max_pages - len(visited_urls)
                    max_additional = min(remaining_budget, config.crawler.enriched_discovery_max_pages)
                    
                    if max_additional > 0:
                        logger.info(f"Entity discovery budget: {max_additional} pages (remaining: {remaining_budget})")
                        logger.info(f"Selected {len(top_entities)} top entities for discovery (filtered by {config.crawler.enriched_discovery_relevance_threshold:.0%} relevance threshold)")
                        
                        # Only discover top N entities (configurable)
                        entities_to_discover = min(len(top_entities), config.crawler.enriched_discovery_max_entities)
                        entity_texts = [e.get('text', e) if isinstance(e, dict) else e for e in top_entities[:entities_to_discover]]
                        logger.info(f"Discovering pages for top {entities_to_discover} entities: {entity_texts}")
                        
                        # Inline URL discovery with multi-source support
                        discovered_urls = []
                        
                        if source_discovery:
                            # Use multi-source discovery
                            logger.info("Using multi-source discovery (Wikipedia, Wikidata, DBpedia)")
                            urls_with_sources = source_discovery.get_all_urls_for_entities(
                                top_entities[:entities_to_discover],
                                lang='ro'
                            )
                            # Prioritize sources
                            urls_with_sources = source_discovery.prioritize_sources(urls_with_sources)
                            
                            # Extract URLs we haven't visited yet
                            for url, entity, source in urls_with_sources:
                                if url not in visited_urls:
                                    discovered_urls.append((url, entity, source))
                                    knowledge_graph.metadata['sources'].add(source)
                                    if len(discovered_urls) >= max_additional:
                                        break
                            
                            logger.info(f"Discovered {len(discovered_urls)} URLs from multiple sources")
                        else:
                            # Fall back to Wikipedia-only discovery
                            for entity in top_entities[:entities_to_discover]:
                                try:
                                    entity_text = entity.get('text', entity) if isinstance(entity, dict) else entity
                                    entity_encoded = entity_text.replace(' ', '_')
                                    # Only try Romanian Wikipedia first (more relevant for Romanian queries)
                                    candidate_urls = [
                                        f"https://ro.wikipedia.org/wiki/{entity_encoded}",
                                    ]
                                    
                                    for url in candidate_urls:
                                        if crawler._is_valid_url(url) and url not in visited_urls:
                                            discovered_urls.append((url, entity_text, 'wikipedia'))
                                            logger.debug(f"  Discovered: {url}")
                                            if len(discovered_urls) >= max_additional:
                                                break
                                    if len(discovered_urls) >= max_additional:
                                        break
                                except Exception as e:
                                    logger.warning(f"Failed to discover URLs for entity '{entity_text}': {e}")
                        
                        logger.info(f"Discovered {len(discovered_urls)} potential URLs from Wikipedia")
                    
                    # Crawl discovered URLs (browser already started)
                    enriched_count = 0
                    for idx, url_data in enumerate(discovered_urls[:max_additional], 1):
                        # Unpack URL data (url, entity, source)
                        if isinstance(url_data, tuple):
                            url, entity, source = url_data
                        else:
                            url = url_data
                            entity = ""
                            source = "unknown"
                        
                        logger.info(f"[{idx}/{min(len(discovered_urls), max_additional)}] Crawling {source}: {url}")
                        
                        try:
                            # Fetch and process page
                            html_content, text_content = await crawler.fetch_page_content(url)
                            
                            if not text_content or len(text_content.strip()) < 100:
                                logger.info(f"  X Insufficient content extracted")
                                continue
                            
                            # Check relevance
                            is_relevant, relevance_score, translated_text = nlp_processor.is_content_relevant(
                                text_content,
                                config.crawler.query,
                                min_relevance_score=0.1
                            )
                            
                            if not is_relevant:
                                logger.info(f"  X Page not relevant (score: {relevance_score:.2f})")
                                continue
                            
                            logger.info(f"  OK Page is relevant (score: {relevance_score:.2f})")
                            
                            # Extract entities and relations using already-translated text
                            entities, relations, entity_metadata = nlp_processor.extract_entities_and_relations(
                                translated_text, 
                                url,
                                skip_translation=False  # Already translated
                            )
                            
                            if entities or relations:
                                knowledge_graph.merge_entities_and_relations(
                                    entities,
                                    cast(Dict[Tuple[str, str], List[Any]], relations),
                                    entity_metadata
                                )
                                logger.info(f"  OK Found {len(entities)} entities and {len(relations)} relations")
                                enriched_count += 1
                            
                        except Exception as e:
                            logger.warning(f"  ✗ Failed to crawl {url}: {e}")
                            continue
                    
                    # Update statistics after enrichment
                    if enriched_count > 0:
                        enriched_stats = knowledge_graph.get_statistics()
                        logger.info("=" * 60)
                        logger.info("Knowledge Graph Statistics (after enrichment):")
                        logger.info(f"  Total nodes: {enriched_stats['nodes']} (+{enriched_stats['nodes'] - stats['nodes']})")
                        logger.info(f"  Total edges: {enriched_stats['edges']} (+{enriched_stats['edges'] - stats['edges']})")
                        logger.info(f"  Unique entities: {enriched_stats['unique_entities']} (+{enriched_stats['unique_entities'] - stats['unique_entities']})")
                        logger.info(f"  Successfully enriched with {enriched_count} pages")
                        logger.info("=" * 60)
                    else:
                        logger.info("No remaining page budget for enriched discovery")
           
        
        finally:
            # Always shutdown the pipeline executor (if created)
            try:
                if pipeline is not None:
                    pipeline.shutdown()
            except Exception:
                pass

            # Always close the browser
            await crawler.close()
        
        # =================================================================
        # ========== NEW: FINAL GRAPH UNIFICATION STEP ====================
        # =================================================================
        logger.info("=" * 60)
        logger.info("Running Final Graph Unification...")
        logger.info("=" * 60)
        try:
            merged = knowledge_graph.unify_duplicate_nodes()
            logger.info(f"Unified duplicate nodes: {merged} merged")
        except Exception as e:
            logger.warning(f"Final graph unification failed: {e}")
        # =================================================================

        # Apply filters if enabled
        logger.info("=" * 60)
        logger.info("Applying Graph Filters...")
        logger.info("=" * 60)
        
        if config.visualization.remove_isolated_nodes:
            logger.info("Removing isolated nodes...")
            removed = knowledge_graph.remove_isolated_nodes()
            logger.info(f"Removed {removed} isolated nodes")
            logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Filter out year nodes that are not related to the query
        logger.info("Removing unrelated year nodes...")
        query_terms = config.crawler.query.split()
        removed_years = knowledge_graph.filter_year_nodes(query_terms)
        if removed_years > 0:
            logger.info(f"Removed {removed_years} unrelated year nodes")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Remove all nodes not connected to the query (at any distance, or limited by config)
        logger.info("Removing nodes not connected to query...")
        removed_unconnected = knowledge_graph.filter_unconnected_nodes(
            query_terms, 
            max_distance=config.graph.max_query_distance
        )
        if removed_unconnected > 0:
            distance_msg = f"within {config.graph.max_query_distance} hops" if config.graph.max_query_distance else "in component"
            logger.info(f"Removed {removed_unconnected} nodes not connected to query ({distance_msg})")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # NOTE: Previously we removed nodes with noisy NER labels here
        # (LANGUAGE, ORDINAL, CARDINAL, DATE, UNKNOWN, WORK_OF_ART, PRODUCT,
        # PERCENT, QUANTITY, TIME). The UI now supports filtering these labels
        # at display time, so we keep them in the graph to preserve data fidelity
        # and let the user decide what to hide. No removal is performed here.
        logger.info("Skipping removal of noisy NER label nodes (UI filtering is enabled).")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Remove image and file path nodes/edges
        logger.info("Removing image and file path references...")
        removed_images_files = knowledge_graph.remove_image_and_file_nodes()
        if removed_images_files > 0:
            logger.info(f"Removed {removed_images_files} image/file nodes and edges")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # DEBUG: Check what the reasons look like before filtering
        logger.info("DEBUG: Checking sample edge reasons before file:// filtering...")
        sample_edges = list(knowledge_graph.graph.edges(data=True))[:5]
        for source, target, data in sample_edges:
            reasons = data.get('reasons', [])
            logger.info(f"  Edge: {source} -> {target}")
            for i, reason in enumerate(reasons[:2]):  # Show first 2 reasons
                if '|||' in reason:
                    text, url = reason.rsplit('|||', 1)
                    logger.info(f"    Reason {i+1}: text='{text[:50]}...' url='{url[:100]}'")
                else:
                    logger.info(f"    Reason {i+1}: '{reason[:100]}'")
        
        # Remove relations with file:// protocol and cleanup isolated nodes
        # NOTE: This was removing all edges - if it happens again, check reason formatting
        logger.info("Removing file:// protocol relations...")
        removed_file_protocol = knowledge_graph.remove_file_protocol_relations()
        if removed_file_protocol > 0:
            logger.info(f"Removed {removed_file_protocol} edges/nodes with file:// protocol")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Deduplicate reasons and merge multiple sources
        logger.info("Deduplicating reasons and merging sources...")
        merged_reasons = knowledge_graph.deduplicate_and_merge_reasons()
        if merged_reasons > 0:
            logger.info(f"Merged {merged_reasons} duplicate reasons")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Remove completely irrelevant nodes (more aggressive filtering)
        """
        logger.info("Removing completely irrelevant nodes...")
        removed_irrelevant = knowledge_graph.remove_irrelevant_nodes(query_terms, min_relevance=0.0)
        if removed_irrelevant > 0:
            logger.info(f"Removed {removed_irrelevant} completely irrelevant nodes")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        """
        logger.info("Skipping removal of completely irrelevant nodes (disabled for data fidelity).")
        # Final cleanup: Remove any nodes that became disconnected after all the filtering above
        logger.info("Final cleanup: Removing any disconnected components created by filtering...")
        removed_final = knowledge_graph.filter_unconnected_nodes(
            query_terms, 
            max_distance=config.graph.max_query_distance
        )
        if removed_final > 0:
            logger.info(f"Removed {removed_final} nodes that became disconnected after filtering")
        logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Also remove any remaining isolated nodes (nodes with no edges at all)
        if config.visualization.remove_isolated_nodes:
            logger.info("Final isolated node cleanup...")
            removed_isolated_final = knowledge_graph.remove_isolated_nodes()
            if removed_isolated_final > 0:
                logger.info(f"Removed {removed_isolated_final} isolated nodes in final cleanup")
            logger.info(f"Graph now has {knowledge_graph.graph.number_of_nodes()} nodes, {knowledge_graph.graph.number_of_edges()} edges")
        
        # Get filtered graph for visualization
        graph = knowledge_graph.get_graph()
        logger.info(f"Retrieved graph copy for visualization: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        if config.visualization.filter_by_relevance:
            logger.info("Filtering by relevance to query...")
            query_terms = config.crawler.query.split()
            graph = knowledge_graph.filter_by_relevance(query_terms, min_relevance=0.0)
            logger.info(f"After relevance filtering: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        if config.visualization.min_node_degree > 0:
            logger.info(f"Filtering by minimum degree {config.visualization.min_node_degree}...")
            graph = knowledge_graph.filter_by_degree(config.visualization.min_node_degree)
            logger.info(f"After degree filtering: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # CRITICAL: Remove any disconnected components from the visualization graph
        # This ensures that filter_by_relevance and filter_by_degree didn't create disconnected subgraphs
        logger.info("=" * 60)
        logger.info("Final connectivity check: Removing all disconnected components...")
        logger.info("=" * 60)
        
        # DEBUG: Show what nodes we have before this check
        current_node_count = graph.number_of_nodes()
        logger.info(f"Current graph has {current_node_count} nodes before connectivity check")
        
        # If graph is already empty, skip this check
        if current_node_count == 0:
            logger.warning("Graph is already empty - skipping final connectivity check")
            return graph, knowledge_graph
        
        if current_node_count > 0:
            sample_nodes = list(graph.nodes())[:10]
            logger.info(f"Sample nodes: {sample_nodes}")
        
        # Normalize Romanian characters for better matching
        import unicodedata
        def normalize_romanian(text):
            """Normalize Romanian diacritics for matching."""
            replacements = {
                'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
                'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            # Also remove any remaining diacritics
            text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                          if unicodedata.category(c) != 'Mn')
            return text
        
        query_terms = config.crawler.query.split()
        # Create both original and normalized versions for matching
        query_lower = [term.lower() for term in query_terms]
        query_normalized = [normalize_romanian(term.lower()) for term in query_terms]
        full_query_lower = " ".join(query_terms).lower()
        full_query_normalized = normalize_romanian(full_query_lower)
        
        logger.info(f"Searching for query nodes matching: '{config.crawler.query}'")
        logger.info(f"Normalized form: '{full_query_normalized}'")
        
        # Find core query nodes in the visualization graph with multiple matching strategies
        core_nodes = set()
        for node in graph.nodes():
            node_lower = node.lower()
            node_normalized = normalize_romanian(node_lower)
            
            # Strategy 1: Exact match with full query (original or normalized)
            if node_lower == full_query_lower or node_normalized == full_query_normalized:
                core_nodes.add(node)
                logger.info(f"  Found core node (exact match): {node}")
            # Strategy 2: Exact match with any query term (original or normalized)
            elif any(term.lower() == node_lower or normalize_romanian(term.lower()) == node_normalized 
                    for term in query_terms):
                core_nodes.add(node)
                logger.info(f"  Found core node (term match): {node}")
            # Strategy 3: Node contains ALL query terms (more lenient for partial matches)
            elif all((norm_term in node_normalized) for norm_term in query_normalized if len(norm_term) > 2):
                core_nodes.add(node)
                logger.info(f"  Found core node (all terms in node): {node}")
            # Strategy 4: Node contains query term or vice versa (check normalized versions too)
            elif any((term in node_lower or node_lower in term or 
                     norm_term in node_normalized or node_normalized in norm_term)
                    for term, norm_term in zip(query_lower, query_normalized) if len(norm_term) > 2):
                core_nodes.add(node)
                logger.debug(f"  Found core node (partial match): {node}")
        
        if not core_nodes:
            logger.warning(f"WARNING: No core query nodes found for '{config.crawler.query}'!")
            logger.warning("This might mean the query entity was filtered out. Keeping all nodes.")
            # IMPORTANT: Don't remove anything if we can't find the query nodes
            # The issue is likely with query normalization, not the graph content
        else:
            logger.info(f"Found {len(core_nodes)} core query nodes")
            
            # Get all connected components
            connected_components = list(nx.connected_components(graph))
            logger.info(f"Graph has {len(connected_components)} connected component(s)")
            
            if len(connected_components) > 1:
                # Find the component(s) containing query nodes
                query_component = set()
                for component in connected_components:
                    if any(core_node in component for core_node in core_nodes):
                        query_component.update(component)
                        logger.info(f"  Query component size: {len(component)} nodes")
                
                if query_component:
                    # Remove nodes not in query component
                    nodes_to_remove = set(graph.nodes()) - query_component
                    if nodes_to_remove:
                        logger.info(f"  Removing {len(nodes_to_remove)} nodes in {len(connected_components) - 1} disconnected component(s)")
                        
                        # Log some examples of removed components
                        for component in connected_components:
                            if not any(node in query_component for node in component):
                                component_list = list(component)
                                logger.info(f"    - Disconnected component ({len(component)} nodes): {component_list[:3]}{'...' if len(component) > 3 else ''}")
                        
                        graph.remove_nodes_from(nodes_to_remove)
                        logger.info(f"  OK Removed all disconnected components")
                else:
                    logger.warning("  WARNING: Query component is empty after filtering!")
            else:
                logger.info("  OK Graph is already fully connected")
        
        logger.info("=" * 60)
        
        # Get final statistics
        stats = knowledge_graph.get_statistics()
        logger.info("=" * 60)
        logger.info("Knowledge Graph Statistics (after filtering):")
        logger.info(f"  Total nodes: {stats['nodes']}")
        logger.info(f"  Total edges: {stats['edges']}")
        logger.info(f"  Unique entities: {stats['unique_entities']}")
        logger.info(f"  Unique relations: {stats['unique_relations']}")
        logger.info(f"  Total pages processed: {knowledge_graph.metadata.get('total_pages_processed', len(visited_urls))}")
        logger.info(f"  Sources used: {', '.join(knowledge_graph.metadata.get('sources', {'wikipedia'}))}")
        
        # Show entity types
        entity_types = knowledge_graph.get_entity_types()
        logger.info("Entity types:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {entity_type}: {count}")
        logger.info("=" * 60)
        
        # Save the final knowledge graph if incremental building is enabled
        if config.graph.enable_incremental and config.graph.auto_save:
            logger.info("Saving knowledge graph for future incremental building...")
            knowledge_graph.metadata['total_pages_processed'] = len(visited_urls)
            knowledge_graph.save_to_file(config.graph.graph_file)
        
        # Shutdown NLP processor to ensure all background tasks complete
        logger.info("Waiting for all NLP processing to complete...")
        nlp_processor.shutdown(wait=True, timeout=30.0)
        logger.info("✓ All data processing complete")
        
        # Visualize the graph (check the FILTERED graph, not full knowledge graph)
        filtered_node_count = graph.number_of_nodes()
        if filtered_node_count > 0:
            logger.info(f"Creating visualization with {filtered_node_count} nodes...")
            visualizer.visualize(graph)
            logger.info("Visualization complete!")
            
            # Create interactive visualization if enabled
            if config.visualization.enable_interactive:
                try:
                    from vizualization.interactive_viz import create_interactive_visualization
                    
                    logger.info("Creating interactive visualization...")
                    create_interactive_visualization(
                        graph=graph,
                        output_file=config.visualization.interactive_file,
                        title=f"Knowledge Graph: {config.crawler.query}",
                        node_size_by=config.visualization.node_size_by,
                        color_by=config.visualization.color_by,
                        show_edge_labels=config.visualization.show_edge_labels
                    )
                    logger.info(f"✓ Interactive visualization saved to {config.visualization.interactive_file}")
                except ImportError:
                    logger.warning("Plotly not installed. Skipping interactive visualization.")
                    logger.warning("Install with: pip install plotly")
                except Exception as e:
                    logger.error(f"Failed to create interactive visualization: {e}", exc_info=True)
            
            # Print entity disambiguation statistics if enabled
            if config.graph.enable_disambiguation and knowledge_graph.disambiguator:
                logger.info("=" * 60)
                logger.info("Entity Disambiguation Statistics:")
                stats = knowledge_graph.disambiguator.get_statistics()
                logger.info(f"  Total entities processed: {stats['total_entities']}")
                logger.info(f"  Canonical entities: {stats['canonical_entities']}")
                logger.info(f"  Aliases merged: {stats['aliases_merged']}")
                logger.info(f"  Auto-discovered aliases: {stats['auto_discovered']}")
                logger.info("=" * 60)
                
                # Log some example aliases
                if stats['auto_discovered'] > 0:
                    aliases_export = knowledge_graph.disambiguator.export_aliases()
                    logger.info("Sample entity aliases discovered:")
                    for i, (canonical, aliases) in enumerate(list(aliases_export.items())[:5]):
                        if len(aliases) > 1:
                            logger.info(f"  {canonical}:")
                            for alias in list(aliases)[:3]:
                                if alias != canonical:
                                    logger.info(f"    → {alias}")
                    logger.info("=" * 60)
        else:
            logger.warning("No entities found in filtered graph. Skipping visualization.")
            logger.warning(f"Note: Full knowledge graph has {stats['nodes']} nodes, but all were filtered out.")
            logger.warning("This usually means:")
            logger.warning("  1. No nodes connected to the query entity")
            logger.warning("  2. All nodes filtered by degree/relevance settings")
            logger.warning("  3. Only isolated nodes remained after filtering")
            logger.warning("\nTry:")
            logger.warning("  - Reducing min_node_degree in config")
            logger.warning("  - Disabling filter_by_relevance")
            logger.warning("  - Checking if query entity was correctly extracted")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        # Clean up checkpoint file on successful completion
        if config.crawler.enable_checkpoints:
            checkpoint_path = Path(config.crawler.checkpoint_file)
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    logger.info("Checkpoint file cleaned up")
                except Exception as e:
                    logger.debug(f"Failed to clean up checkpoint: {e}")
        
        # Summary of where results were saved
        logger.info("=" * 60)
        logger.info("SCAN COMPLETE - Results saved to:")
        logger.info(f"  📁 Scan directory: {scan_paths['scan_dir']}")
        if scan_paths['graph_file'].exists():
            logger.info(f"  📊 Knowledge graph: {scan_paths['graph_file'].name}")
        if scan_paths['viz_file'].exists():
            logger.info(f"  📈 Static visualization: {scan_paths['viz_file'].name}")
        if scan_paths['interactive_viz_file'].exists():
            logger.info(f"  🎨 Interactive visualization: {scan_paths['interactive_viz_file'].name}")
        if scan_paths['cache_dir'].exists():
            cache_files = list(scan_paths['cache_dir'].iterdir())
            logger.info(f"  💾 Cache: {len(cache_files)} files in {scan_paths['cache_dir'].name}/")
        logger.info("=" * 60)
    
    logger.info("=" * 60)
    logger.info("Knowledge Graph Extractor finished successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
