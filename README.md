# SearchANDGraph

SearchANDGraph is a Python application that discovers web sources for a query, crawls pages (and optionally documents), extracts entities/relations, and produces a knowledge graph with HTML visualizations.

This project is moving to a **desktop executable** workflow: run scans from a **PyQt UI** and view the generated HTML graph inside the app.

## Key Features

- **Crawl + extract**: Uses `nodriver` (Chrome automation) + `trafilatura` for webpage text extraction.
- **Web search seeding**: Discovers candidate URLs via `web_search.py` (DuckDuckGo/Bing scraping).
- **Documents**: Downloads PDFs/images and extracts text/tables where possible.
- **NLP pipeline**: Relevance scoring, entity extraction, relation extraction; optional translation to English.
- **Graph output**: Saves a `NetworkX` graph and generates HTML visualizations (static + interactive).
- **Scan management**: Query-safe scan directories and reproducible artifact paths.

## Quickstart

### 1) Install

Create/activate a virtualenv and install dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy model used by default in `config.py`:

```bash
python -m spacy download en_core_web_lg
```

### 2) Run a scan (script)

```bash
python main_refactored.py
```

Note: `main_refactored.py` currently runs with defaults unless you modify the call to `main(...)` or the defaults in `config.py`.

### 3) Run the desktop UI (PyQt)

```bash
python desktop_app.py
```

## Output Artifacts

Each query writes into a stable scan directory under `scans/<sanitized_query>/` (see `scan_manager.py`). Typical outputs:

- `knowledge_graph.pkl` (serialized graph)
- `knowledge_graph.html` (static visualization)
- `knowledge_graph_interactive.html` (interactive visualization)
- `cache/` (per-scan cache)
- `downloads/` (downloaded PDFs and other document artifacts)

## Configuration

Configuration is defined via dataclasses in `config.py`:

- `CrawlerConfig`: crawl/search settings (BFS, max pages, headless mode, web search, PDF downloads)
- `NLPConfig`: spaCy model, translation toggles/provider, parallelism, disambiguation options
- `VisualizationConfig`: graph size/limits/layout settings + interactive output
- `AppConfig`: container with `.default()` factory

## Optional System Dependencies

Some features require additional non-Python prerequisites:

- **Browser automation**: `nodriver` requires a working Chrome/Chromium install.
- **OCR** (images): `pytesseract` requires the Tesseract binary installed and available on `PATH`.
- **Advanced PDF tables** (optional): `tabula` requires Java (tabula-py).

If these aren’t available, the pipeline will still run, but document/OCR/table extraction may be reduced.

## Security Scanning (CI)

GitHub Actions workflow: [.github/workflows/codeql.yml](.github/workflows/codeql.yml)

- **CodeQL** runs on pushes and PRs to `main`/`master`, plus a weekly schedule (Mondays 02:56 UTC) and manual `workflow_dispatch`.
- **Dependency Review (PR diffs)** runs only on pull requests (it requires a PR diff context).
- **Dependency audit (pip-audit)** runs and reports findings in the workflow summary, but is configured to be **non-blocking** (does not fail the job).

## Project Layout (high level)

- `main_refactored.py`: main scan orchestration entry point
- `desktop_app.py`: PyQt desktop UI (runs scans and displays output)
- `crawler.py`: browser-based crawler + extraction + URL discovery
- `web_search.py`: search engine scraping + download helpers
- `document_extractor.py`: PDF/image download + text/table extraction
- `nlp_processor.py` / `nlp_enhancements.py`: entity/relation extraction and enhancements
- `graph_builder.py`: knowledge graph building/merging
- `visualizer.py` / `interactive_viz.py`: HTML graph visualizations
- `scan_manager.py`: standardized scan directory + artifact paths

