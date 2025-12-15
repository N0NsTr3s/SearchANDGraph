# Knowledge Graph Extractor

A Python application that crawls web pages, extracts named entities and their relationships, and visualizes them as an interactive knowledge graph.

## Features

- **BFS Web Crawling**: Systematic breadth-first search crawling with comprehensive link capture
- **Web Crawling**: Automated web crawling using nodriver (undetected Chrome)
- **NLP Processing**: Entity and relation extraction using spaCy
- **Translation Support**: Automatic translation for multilingual content
- **Entity Deduplication**: Smart consolidation of partial names into full names
- **Graph Filtering**: Remove isolated nodes and irrelevant entities
- **Temporal Analysis**: Extract and track dates/timelines associated with relationships
- **Knowledge Graph**: Build relationships between entities
- **Interactive Visualization**: Bokeh-based interactive graph visualization with temporal information
- **Modular Design**: Clean, maintainable code structure
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Security Scanning (CI)

This repository includes a GitHub Actions security pipeline using **CodeQL** for code scanning and **Dependency Review** for pull requests.

- CodeQL runs on pushes/PRs to the default branches and on a weekly schedule.
- Dependency Review runs on pull requests to flag risky dependency changes.

## Project Structure

```
relations_extractor/
├── main.py                  # Original monolithic script
├── main_refactored.py       # New refactored entry point
├── config.py                # Configuration management
├── logger.py                # Logging setup
├── crawler.py               # Web crawling functionality
├── nlp_processor.py         # NLP and entity extraction
├── nlp_enhancements.py      # Advanced NLP (coreference, entity linking, temporal)
├── temporal_processor.py    # Temporal information extraction
├── graph_builder.py         # Knowledge graph construction
├── visualizer.py            # Bokeh-based graph visualization
├── interactive_viz.py       # Plotly-based graph visualization
├── entity_disambiguation.py # Entity deduplication and disambiguation
├── node_cleaner.py          # Node name normalization
├── test_temporal.py         # Test script for temporal extraction
├── __init__.py              # Package initialization
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download spaCy language model**:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

Run the refactored version:
```bash
python main_refactored.py
```

Or run the original version:
```bash
python main.py
```

### Customizing Configuration

Edit the configuration in `main_refactored.py`:

```python
config = AppConfig.default()

# Customize crawler settings
config.crawler.query = "Nvidia"
config.crawler.start_url = "https://en.wikipedia.org/wiki/Nvidia"
config.crawler.max_pages = 10

# Customize NLP settings
config.nlp.min_entity_length = 3

# Enable translation for non-English content
config.nlp.enable_translation = True
config.nlp.target_language = "en"
config.nlp.translation_provider = "google"

# Customize visualization settings
config.visualization.output_file = "my_knowledge_graph.html"
```

### Using as a Library

```python
from config import AppConfig
from crawler import WebCrawler
from nlp_processor import NLPProcessor
from graph_builder import KnowledgeGraph
from visualizer import GraphVisualizer

# Initialize components
config = AppConfig.default()
crawler = WebCrawler(config.crawler)
nlp_processor = NLPProcessor(config.nlp)
knowledge_graph = KnowledgeGraph()

# Crawl and process
crawled_contents, _ = await crawler.crawl(
    "https://example.com",
    "search query",
    max_pages=5
)

for content in crawled_contents:
    entities, relations = nlp_processor.extract_entities_and_relations(content)
    knowledge_graph.merge_entities_and_relations(entities, relations)

# Visualize
visualizer = GraphVisualizer(config.visualization)
visualizer.visualize(knowledge_graph.get_graph())
```

## Components

### 1. Configuration (`config.py`)
Manages all application settings using dataclasses:
- `CrawlerConfig`: Web crawler settings
- `NLPConfig`: NLP processing settings
- `VisualizationConfig`: Graph visualization settings
- `AppConfig`: Main configuration container

### 2. Logger (`logger.py`)
Sets up logging with:
- Console output with timestamps
- File logging to `logs/` directory
- Configurable log levels

### 3. Web Crawler (`crawler.py`)
Handles web page crawling:
- Async crawling with nodriver
- Content extraction with trafilatura
- Related link discovery
- URL validation and filtering

### 4. NLP Processor (`nlp_processor.py`)
Extracts entities and relationships:
- Named entity recognition with spaCy
- Relationship extraction between entities
- Configurable entity filtering

### 5. Knowledge Graph (`graph_builder.py`)
Builds and manages the graph:
- Entity and relation aggregation
- Graph statistics
- Graph filtering capabilities
- NetworkX integration

### 6. Visualizer (`visualizer.py`)
Creates interactive visualizations:
- Bokeh-based interactive plots
- Node and edge hover tooltips
- Customizable styling
- Searchable control panel for nodes and connections
- Pinned connections side panel for quick reference
- HTML output

#### Exploring the graph

The generated HTML includes a search bar on the right-hand panel:

- **Type any name, keyword, or QID** to highlight matching nodes and relationships in real time.
- **Results list** summarises matched entities and connections so you can jump between them quickly.
- **Clear search** resets the view to the original layout and styling.
- **Pinned connections** capture any edge you click, making it easy to keep important evidence in sight while you explore.

### 7. Temporal Processor (`temporal_processor.py`)
Extracts temporal information from text:
- Date extraction using spaCy and dateparser
- Date normalization to YYYY-MM-DD format
- Association of dates with relationships
- Support for date ranges (e.g., "2020-2021")
- Recognition of various date formats (full dates, years, months)

#### Temporal Analysis Features

The temporal processor automatically extracts and normalizes dates from text, associating them with relationships. For example:

```python
# Text: "Apple was founded by Steve Jobs in 1976"
# Result: Relationship with [dates:1976-01-01]

# Text: "Microsoft was founded on April 4, 1975"
# Result: Relationship with [dates:1975-04-04]
```

Dates are displayed in:
- **Hover tooltips**: Quick view of timeline information
- **Pinned connections**: Full temporal context with 📅 timeline indicators
- **Graph edges**: Stored in the `dates` attribute

To test temporal extraction:
```bash
python test_temporal.py
```

This feature enables answering "when" questions and tracking the evolution of relationships over time.

## Configuration Options

### Crawler Configuration
- `query`: Search query for finding related pages
- `start_url`: Starting URL for crawling
- `max_pages`: Maximum number of pages to crawl
- `page_timeout`: Timeout for page loading (seconds)
- `browser_headless`: Run browser in headless mode

### NLP Configuration
- `spacy_model`: spaCy model to use
- `min_entity_length`: Minimum length for entity text
- `extract_comments`: Include comments in extraction
- `extract_tables`: Include tables in extraction
- `enable_translation`: Enable automatic translation (default: True)
- `target_language`: Target language for translation (default: "en")
- `translation_provider`: Translation service provider (default: "google")

### Visualization Configuration
- `output_file`: Output HTML filename
- `plot_title`: Graph title
- `x_range`, `y_range`: Plot dimensions
- `scale`: Graph layout scale
- `center`: Graph center point
- `max_nodes`: Maximum nodes to display (default: 200) - filters by importance
- `min_edge_confidence`: Minimum confidence threshold for edges (optional)
- `layout_iterations`: Number of spring layout iterations (default: 80)
- `layout_spread`: Multiplier for node spacing (default: 2.5)
- `layout_force`: Force-directed layout strength (optional)
- `layout_seed`: Random seed for reproducible layouts (default: 42)
- `node_size_range`: Min and max node sizes (default: [12, 34])
- `edge_width_range`: Min and max edge widths (default: [1.0, 4.5])
- `auto_range`: Auto-adjust plot bounds (default: True)

## Output

The application generates:
1. **Log files**: In `logs/knowledge_graph.log`
2. **HTML visualization**: Interactive graph (default: `focused_knowledge_graph.html`)
3. **Console output**: Progress and statistics

## Requirements

- Python 3.8+
- nodriver
- spacy (with en_core_web_lg model recommended)
- networkx
- bokeh
- plotly (for interactive_viz.py)
- trafilatura
- translators (for multilingual support)
- dateparser (for temporal analysis)
- Other dependencies in `requirements.txt`

## Improvements Over Original

1. **Modularity**: Code split into logical components
2. **Configuration**: Centralized, type-safe configuration
3. **Logging**: Comprehensive logging throughout
4. **Error Handling**: Better error handling and reporting
5. **Type Hints**: Full type hints for better IDE support
6. **Documentation**: Extensive docstrings and comments
7. **Extensibility**: Easy to extend and modify
8. **Testing**: Structure supports unit testing
9. **Maintainability**: Cleaner, more readable code

## Future Enhancements

- [x] ~~Add translation support for multilingual content~~
- [x] ~~Entity deduplication~~
- [x] ~~BFS crawling strategy~~
- [x] ~~Graph filtering (isolated nodes, relevance)~~
- [x] ~~Temporal analysis with date extraction~~
- [ ] Advanced entity disambiguation using Wikidata QIDs
- [ ] Context vector-based disambiguation fallback
- [ ] Add unit tests
- [ ] Support for multiple NLP models
- [ ] Database persistence for large graphs
- [ ] REST API for graph queries
- [ ] Export to various formats (JSON, GraphML, etc.)
- [ ] Incremental crawling and updating
- [ ] Multi-language NLP models
- [ ] Temporal querying and timeline visualization

## Documentation

- [README.md](README.md) - Main documentation (this file)
- [TRANSLATION.md](TRANSLATION.md) - Translation features and usage
- [ENTITY_DEDUPLICATION.md](ENTITY_DEDUPLICATION.md) - Entity deduplication details
- [BFS_CRAWLING.md](BFS_CRAWLING.md) - BFS crawling and graph filtering
- See inline code documentation for detailed API information

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
