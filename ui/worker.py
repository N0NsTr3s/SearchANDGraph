from __future__ import annotations

import asyncio
from PyQt6.QtCore import QObject, pyqtSignal

from ui.models import ScanRequest
from ui.logging import _install_qt_log_handler


class ScanWorker(QObject):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    started = pyqtSignal(str)  # scan_dir
    finished = pyqtSignal(str, str)  # scan_dir, html_path
    failed = pyqtSignal(str)

    def __init__(self, request: ScanRequest):
        super().__init__()
        self._request = request

    def run(self) -> None:
        try:
            _install_qt_log_handler(lambda line: self.log.emit(line))

            self.status.emit("Running scan...")
            # Local imports to avoid circular import / heavy startup cost
            from scraper.scan_manager import get_scan_paths
            from main import main as run_main

            scan_paths = get_scan_paths(
                self._request.query,
                base_dir=self._request.base_dir,
                add_timestamp=False,
            )

            # Notify listeners that the scan is starting (with scan_dir)
            try:
                self.started.emit(str(scan_paths["scan_dir"].resolve()))
            except Exception:
                pass

            asyncio.run(
                run_main(
                    query=self._request.query,
                    start_url=getattr(self._request, "start_url", None),
                    max_pages=self._request.max_pages,
                    add_timestamp=False,
                    base_dir=self._request.base_dir,
                    ui_preview_dir=str((scan_paths["scan_dir"] / "_ui_previews").resolve())
                    if getattr(self._request, "preview_enabled", False)
                    else None,
                    browser_headless=self._request.headless,
                    enable_web_search=self._request.enable_web_search,
                    web_search_download_pdfs=self._request.download_pdfs,
                    preferred_sources=None if not self._request.preferred_sources else list(self._request.preferred_sources),
                    blacklisted_sources=None if not self._request.blacklisted_sources else list(self._request.blacklisted_sources),
                    viz_max_nodes=self._request.viz_max_nodes,
                    viz_min_edge_confidence=self._request.viz_min_edge_confidence,
                    viz_remove_isolated_nodes=self._request.viz_remove_isolated_nodes,
                    enable_phase2=self._request.enable_phase2,
                    phase2_max_pages=self._request.phase2_max_pages,
                    phase2_concurrent_tabs=self._request.phase2_concurrent_tabs,
                    document_min_relevance=self._request.document_min_relevance,
                    downloads_prune_irrelevant=self._request.downloads_prune_irrelevant,
                    downloads_prune_mode=self._request.downloads_prune_mode,
                    web_search_max_pdf_downloads=self._request.web_search_max_pdf_downloads,
                    web_search_min_relevance=self._request.web_search_min_relevance,
                    nlp_min_confidence=self._request.nlp_min_confidence,
                    nlp_min_relation_confidence=self._request.nlp_min_relation_confidence,
                )
            )

            html_path = ""
            if scan_paths["viz_file"].exists():
                html_path = str(scan_paths["viz_file"].resolve())
            elif scan_paths["interactive_viz_file"].exists():
                html_path = str(scan_paths["interactive_viz_file"].resolve())

            self.status.emit("Done")
            self.finished.emit(str(scan_paths["scan_dir"].resolve()), html_path)
        except Exception as e:
            self.status.emit("Error")
            self.failed.emit(str(e))
