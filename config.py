from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NLPConfig:
    """NLP processing configuration"""
    spacy_model: str = "en_core_web_lg"
    min_entity_length: int = 3  # Reduced for better entity capture after translation
    extract_comments: bool = False
    extract_tables: bool = True
    enable_translation: bool = True  # CRITICAL: Translate all non-English text to English
    target_language: str = "en"  # Translate to English for universal NER support
    translation_provider: str = "google"  # google, bing, baidu, etc.
    deduplicate_entities: bool = True  # Remove partial names when full names exist
    parallel_processing: bool = True  # Process multiple pages concurrently
    max_workers: int = 8  # Number of parallel NLP workers
    enable_disambiguation: bool = True  # Disambiguate entities using context
    enable_query_expansion: bool = True  # Expand query with synonyms
    enable_confidence_scoring: bool = True  # Score relationship confidence
    min_confidence: float = 0.3  # Minimum confidence threshold
    enable_temporal_extraction: bool = True  # Extract temporal relationships
    
    # Advanced NLP features
    enable_coreference: bool = True  # Resolve pronouns to entities (he/she/it -> actual names)
    enable_enhanced_relations: bool = True  # Use advanced dependency patterns for relation extraction
    enable_entity_linking: bool = True  # Link entities to Wikidata/DBpedia
    entity_linking_threshold: float = 0.85  # Minimum confidence for entity linking
    enable_relation_confidence: bool = True  # Compute confidence scores for each relation
    min_relation_confidence: float = 0.4  # Filter relations below this confidence


@dataclass
class CacheConfig:
    """Configuration for disk caching."""
    enabled: bool = True
    cache_dir: str = "cache"
    page_cache_ttl: int = 86400  # 24 hours in seconds
    nlp_cache_ttl: int = 604800  # 7 days in seconds
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB


@dataclass
class CrawlerConfig:
    """Configuration for the web crawler."""
    query: str = "Nvidia"
    start_url: str = ""
    max_pages: int = 10
    page_timeout: int = 20
    browser_headless: bool = False
    use_bfs: bool = True  # Use BFS for crawling
    capture_all_links: bool = True  # Capture all links, including those related to discovered entities
    concurrent_tabs: int = 15  # Number of tabs to open concurrently for parallel crawling
    enable_checkpoints: bool = True  # Save progress periodically
    checkpoint_interval: int = 5  # Save checkpoint every N pages
    checkpoint_file: str = "checkpoint.pkl"
    multi_source_discovery: bool = True  # Use multiple sources beyond Wikipedia
    sources: list[str] = field(default_factory=lambda: ["wikipedia"])  # Note: "wikidata" provides no crawlable content (entity linker handles it via API), "dbpedia" has limited text (mostly RDF data), "web" is too noisy
    enable_enriched_discovery: bool = True  # Enable entity-based enriched discovery
    enriched_discovery_max_entities: int = 3  # Max entities to discover (reduced from 15)
    enriched_discovery_relevance_threshold: float = 0.25  # Min relevance score for entity discovery (25%)
    enriched_discovery_max_pages: int = 5  # Max additional pages to crawl during enriched discovery
    # Web search configuration
    enable_web_search: bool = True  # Use web search to discover URLs instead of just Wikipedia
    web_search_max_results: int = 100  # Max search results to consider
    web_search_min_relevance: float = 0.30  # Min relevance score for search results
    ignore_start_url: bool = False  # If True, completely ignore start_url and only use search results
    # Phase-specific budgeting
    phase2_max_pages: int = 1000  # Max pages allowed for Phase 2 deep dive (set high or 0 for unlimited)
    enable_phase2: bool = True  # Enable Phase 2 deep-dive searches and crawling
    
    

    

@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    output_file: str = "focused_knowledge_graph.html"
    plot_title: str = "Knowledge Graph"
    x_range: tuple[float, float] = (-2.1, 2.1)
    y_range: tuple[float, float] = (-2.1, 2.1)
    scale: float = 2.0
    center: tuple[float, float] = (0.0, 0.0)
    remove_isolated_nodes: bool = True  # Remove nodes with no connections
    filter_by_relevance: bool = True  # Filter nodes not related to query
    min_node_degree: int = 1  # Minimum connections for a node to be shown
    show_confidence: bool = True  # Display confidence scores on edges
    max_nodes: Optional[int] = 200  # Limit the number of nodes for readability (None = no limit)
    min_edge_confidence: Optional[float] = None  # Hide edges below this confidence score
    layout_iterations: int = 80  # Iterations for the spring layout solver
    layout_force: Optional[float] = None  # Override spacing force (k) for spring layout
    layout_seed: int = 42  # Seed for reproducible layouts
    layout_spread: float = 2.5  # Multiply layout coordinates to add breathing room
    auto_range: bool = True  # Automatically resize plot ranges to fit layout
    node_size_range: tuple[int, int] = (12, 34)  # Min/max node size in screen units
    edge_width_range: tuple[float, float] = (1.0, 4.5)  # Min/max edge width based on confidence
    
    # Interactive Visualization
    enable_interactive: bool = True  # Generate interactive Plotly visualization
    interactive_file: str = "knowledge_graph_interactive.html"  # Output file for interactive viz
    node_size_by: str = "degree"  # Size nodes by: "degree", "uniform", "confidence"
    color_by: str = "label"  # Color nodes by: "label", "cluster", "query_distance"
    show_edge_labels: bool = False  # Show relationship text on edges (can be cluttered)


@dataclass
class GraphConfig:
    """Configuration for graph building."""
    enable_incremental: bool = True  # Enable incremental graph building
    graph_file: str = "knowledge_graph.pkl"  # File to save/load graph
    auto_save: bool = True  # Automatically save graph after building
    merge_strategy: str = "update"  # "update" or "replace" when loading existing
    
    # Entity Disambiguation
    enable_disambiguation: bool = True  # Enable entity name disambiguation
    similarity_threshold: float = 0.85  # Minimum similarity (0-1) for aliasing
    auto_discover_aliases: bool = True  # Automatically discover entity aliases
    
    # Distance Filtering
    max_query_distance: int | None = None  # Maximum hops from query (None = unlimited, 2-3 recommended for focused graphs)


@dataclass
class AppConfig:
    """Main application configuration."""
    cache: CacheConfig
    crawler: CrawlerConfig
    nlp: NLPConfig
    graph: GraphConfig
    visualization: VisualizationConfig
    
    @classmethod
    def default(cls) -> 'AppConfig':
        """Create default configuration."""
        return cls(
            cache=CacheConfig(),
            crawler=CrawlerConfig(),
            nlp=NLPConfig(),
            graph=GraphConfig(),
            visualization=VisualizationConfig()
        )
