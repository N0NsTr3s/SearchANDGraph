"""PyQt desktop UI for running SearchANDGraph scans.

Minimal UX:
- Enter query + basic crawl options
- Click Start to run a scan in the background
- Stream logs to the UI
- Display the generated HTML graph inside the app
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import CrawlerConfig, NLPConfig, VisualizationConfig
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QUrl, QTimer, QEvent
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

from single_instance import hold_app_mutex
from updater import check_for_updates, perform_update


def _prompt_update_and_install(release_data: dict) -> None:
    """Ask the user via a native Windows dialog, then run installer if accepted."""
    try:
        # Prefer Qt dialog when running inside the GUI
        if QApplication.instance() is not None:
            resp = QMessageBox.question(
                None,
                "Update Found",
                "A new update is available. Download and install now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp == QMessageBox.StandardButton.Yes:
                dlg = UpdateDialog(None, release_data)
                dlg.exec()
            return

        # Fallback to native Win32 dialog if Qt isn't available
        import ctypes

        # MB_YESNO = 0x04, IDYES = 6
        mbox = ctypes.windll.user32.MessageBoxW
        resp = mbox(0, "A new update is available. Download and install now?", "Update Found", 0x04)
        if int(resp) == 6:
            perform_update(release_data, silent=True)
            sys.exit(0)
    except Exception:
        # If anything goes wrong (no UI, etc.), skip updates quietly.
        return


class UpdateWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, release_data: dict):
        super().__init__()
        self._release = release_data

    def run(self) -> None:
        try:
            # Import here to avoid circular imports at module import time
            from updater import perform_update

            def cb(pct: int, msg: str) -> None:
                try:
                    self.progress.emit(int(pct), str(msg))
                except Exception:
                    pass

            result = perform_update(self._release, silent=True, progress_callback=cb)
            self.finished.emit(result or {})
        except Exception as e:
            self.error.emit(str(e))


class UpdateDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], release_data: dict):
        super().__init__(parent)
        self.setWindowTitle("Updating...")
        self.setModal(True)

        self._label = QLabel("Preparing update...")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        self._close_btn = QPushButton("Close")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self._on_close)

        self._view_log_btn = QPushButton("View Update Log")
        self._view_log_btn.setEnabled(False)
        self._view_log_btn.clicked.connect(self._on_view_log)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        layout.addWidget(self._view_log_btn)
        layout.addWidget(self._close_btn)

        # Worker/thread setup
        self._thread = QThread()
        self._worker = UpdateWorker(release_data)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Start
        self._thread.start()

        # polling timer for update status/log (populated after run finishes)
        self._result: dict = {}
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_update_status)

    def _on_progress(self, pct: int, msg: str) -> None:
        try:
            self._bar.setValue(int(pct))
            self._label.setText(str(msg))
        except Exception:
            pass

    def _on_finished(self, result: dict) -> None:
        self._result = result or {}
        action = self._result.get("action")
        if action == "installer":
            self._label.setText("Installer launched.")
            self._bar.setValue(100)
            self._close_btn.setEnabled(True)
            # Installer launched; safe to quit app
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:
                pass
            QApplication.quit()
            return

        # action == script (archive flow)
        self._label.setText("Update started in background. Waiting for completion...")
        self._bar.setValue(100)
        self._close_btn.setEnabled(True)

        # Enable view-log button immediately if the log already exists
        log = self._result.get("log")
        if log and os.path.exists(log):
            self._view_log_btn.setEnabled(True)

        # Start polling for a status file that the script will write when finished
        status = self._result.get("status")
        if status:
            self._status_path = status
            self._poll_timer.start()

        try:
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass

    def _on_error(self, err: str) -> None:
        self._label.setText(f"Update failed: {err}")
        self._close_btn.setEnabled(True)
        try:
            self._thread.quit()
            self._thread.wait(2000)
        except Exception:
            pass

    def _on_close(self) -> None:
        self.accept()

    def _on_view_log(self) -> None:
        try:
            log = self._result.get("log") if isinstance(self._result, dict) else None
            if log and os.path.exists(log):
                from PyQt6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(log)))
        except Exception:
            pass

    def _poll_update_status(self) -> None:
        try:
            status = getattr(self, "_status_path", None)
            if not status:
                return
            if os.path.exists(status):
                # Read status JSON
                try:
                    import json

                    with open(status, "r", encoding="utf-8") as sf:
                        data = json.load(sf)
                    exitcode = int(data.get("exit", -1))
                except Exception:
                    exitcode = -1

                log = self._result.get("log")
                if exitcode == 0:
                    self._label.setText("Update applied successfully.")
                else:
                    self._label.setText(f"Update failed (exit {exitcode}). Click 'View Update Log' to inspect.")
                    if log and os.path.exists(log):
                        self._view_log_btn.setEnabled(True)

                self._poll_timer.stop()
        except Exception:
            pass


class ExternalLinksPage(QWebEnginePage):
    def __init__(self, parent=None, emit_log=None):
        super().__init__(parent)
        self._emit_log = emit_log
        self._popup_page: QWebEnginePage | None = None

    def _open_external(self, url: QUrl) -> bool:
        """Open a URL via the OS.

        For file URLs on Windows, QDesktopServices is more reliable when given a
        local-file URL (QUrl.fromLocalFile) rather than a file:// URL string.
        """
        if not url or not url.isValid():
            return False
        try:
            scheme = (url.scheme() or "").lower()
            if scheme == "file":
                local_path = url.toLocalFile()
                if local_path:
                    return QDesktopServices.openUrl(QUrl.fromLocalFile(local_path))
        except Exception:
            pass
        return QDesktopServices.openUrl(url)

    def createWindow(self, _type):
        # Handle links with target="_blank" / window.open() by opening them externally.
        popup = QWebEnginePage(self.profile(), self)

        def _open_and_drop(url: QUrl):
            if url and url.isValid():
                ok = self._open_external(url)
                if self._emit_log:
                    if ok:
                        self._emit_log(f"Opened external link: {url.toString()}")
                    else:
                        self._emit_log(f"Failed to open external link: {url.toString()}")
            # Keep a reference briefly, then drop it.
            self._popup_page = None

        popup.urlChanged.connect(_open_and_drop)
        self._popup_page = popup
        return popup

    def acceptNavigationRequest(self, url: QUrl, nav_type, isMainFrame: bool) -> bool:  # type: ignore[override]
        # Some embedded HTML (Bokeh Div content) uses normal <a href> links.
        # Open those externally instead of navigating the embedded viewer away.
        try:
            scheme = (url.scheme() or "").lower()
            if scheme in {"http", "https", "file"}:
                # Always open external/file URLs outside the embedded view.
                # This intentionally covers both user clicks and JS-driven navigations
                # (e.g., shadow-dom-safe handlers using window.location.href).
                current = self.url()
                current_str = current.toString() if current and current.isValid() else ""
                target_str = url.toString() if url and url.isValid() else ""

                # Allow the initial/normal load of the current page.
                if isMainFrame and current_str and target_str and target_str == current_str:
                    return super().acceptNavigationRequest(url, nav_type, isMainFrame)

                # Don't hijack navigation to other local HTML pages (rare, but can be useful).
                if scheme == "file" and target_str.lower().endswith(".html"):
                    return super().acceptNavigationRequest(url, nav_type, isMainFrame)

                # Otherwise open externally and cancel navigation.
                ok = self._open_external(url)
                if self._emit_log:
                    if ok:
                        self._emit_log(f"Opened external link: {url.toString()}")
                    else:
                        self._emit_log(f"Failed to open external link: {url.toString()}")
                return False
        except Exception:
            pass
        return super().acceptNavigationRequest(url, nav_type, isMainFrame)

    def javaScriptConsoleMessage(self, level, message: str, lineNumber: int, sourceID: str) -> None:  # type: ignore[override]
        if self._emit_log:
            self._emit_log(f"JS[{lineNumber}]: {message}")
        super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)


@dataclass(frozen=True)
class ScanRequest:
    query: str
    max_pages: int
    headless: bool
    enable_web_search: bool
    download_pdfs: bool
    base_dir: str = "scans"
    preferred_sources: tuple[str, ...] = ()
    blacklisted_sources: tuple[str, ...] = ()
    viz_max_nodes: Optional[int] = None
    viz_min_edge_confidence: Optional[float] = None
    viz_remove_isolated_nodes: Optional[bool] = None

    enable_phase2: Optional[bool] = None
    phase2_max_pages: Optional[int] = None
    phase2_concurrent_tabs: Optional[int] = None

    document_min_relevance: Optional[float] = None
    downloads_prune_irrelevant: Optional[bool] = None
    downloads_prune_mode: Optional[str] = None
    web_search_max_pdf_downloads: Optional[int] = None
    web_search_min_relevance: Optional[float] = None

    nlp_min_confidence: Optional[float] = None
    nlp_min_relation_confidence: Optional[float] = None


@dataclass
class UserSettings:
    base_dir: str = "scans"
    preferred_sources: list[str] = None  # type: ignore[assignment]
    blacklisted_sources: list[str] = None  # type: ignore[assignment]
    recent_files: list[str] = None  # type: ignore[assignment]
    viz_max_nodes: Optional[int] = None
    viz_min_edge_confidence: Optional[float] = None
    viz_remove_isolated_nodes: Optional[bool] = None

    enable_phase2: Optional[bool] = None
    phase2_max_pages: Optional[int] = None
    phase2_concurrent_tabs: Optional[int] = None

    document_min_relevance: Optional[float] = None
    downloads_prune_irrelevant: Optional[bool] = None
    downloads_prune_mode: Optional[str] = None
    web_search_max_pdf_downloads: Optional[int] = None
    web_search_min_relevance: Optional[float] = None

    nlp_min_confidence: Optional[float] = None
    nlp_min_relation_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.preferred_sources is None:
            self.preferred_sources = []
        if self.blacklisted_sources is None:
            self.blacklisted_sources = []
        if self.recent_files is None:
            self.recent_files = []

    @staticmethod
    def from_dict(data: dict) -> "UserSettings":
        return UserSettings(
            base_dir=str(data.get("base_dir") or "scans"),
            preferred_sources=[str(s) for s in (data.get("preferred_sources") or [])],
            blacklisted_sources=[str(s) for s in (data.get("blacklisted_sources") or [])],
            recent_files=[str(s) for s in (data.get("recent_files") or [])],
            viz_max_nodes=data.get("viz_max_nodes"),
            viz_min_edge_confidence=data.get("viz_min_edge_confidence"),
            viz_remove_isolated_nodes=data.get("viz_remove_isolated_nodes"),
            enable_phase2=data.get("enable_phase2"),
            phase2_max_pages=data.get("phase2_max_pages"),
            phase2_concurrent_tabs=data.get("phase2_concurrent_tabs"),
            document_min_relevance=data.get("document_min_relevance"),
            downloads_prune_irrelevant=data.get("downloads_prune_irrelevant"),
            downloads_prune_mode=data.get("downloads_prune_mode"),
            web_search_max_pdf_downloads=data.get("web_search_max_pdf_downloads"),
            web_search_min_relevance=data.get("web_search_min_relevance"),
            nlp_min_confidence=data.get("nlp_min_confidence"),
            nlp_min_relation_confidence=data.get("nlp_min_relation_confidence"),
        )

    def to_dict(self) -> dict:
        return {
            "base_dir": self.base_dir,
            "preferred_sources": list(self.preferred_sources or []),
            "blacklisted_sources": list(self.blacklisted_sources or []),
            "recent_files": list(self.recent_files or []),
            "viz_max_nodes": self.viz_max_nodes,
            "viz_min_edge_confidence": self.viz_min_edge_confidence,
            "viz_remove_isolated_nodes": self.viz_remove_isolated_nodes,
            "enable_phase2": self.enable_phase2,
            "phase2_max_pages": self.phase2_max_pages,
            "phase2_concurrent_tabs": self.phase2_concurrent_tabs,
            "document_min_relevance": self.document_min_relevance,
            "downloads_prune_irrelevant": self.downloads_prune_irrelevant,
            "downloads_prune_mode": self.downloads_prune_mode,
            "web_search_max_pdf_downloads": self.web_search_max_pdf_downloads,
            "web_search_min_relevance": self.web_search_min_relevance,
            "nlp_min_confidence": self.nlp_min_confidence,
            "nlp_min_relation_confidence": self.nlp_min_relation_confidence,
        }


class OptionsDialog(QDialog):
    def __init__(self, parent: QWidget, settings: UserSettings):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self._settings = settings

        default_viz = VisualizationConfig()
        default_crawler = CrawlerConfig()
        default_nlp = NLPConfig()

        form = QFormLayout()

        # Output folder
        self.base_dir_edit = QLineEdit(settings.base_dir)
        self.base_dir_edit.setPlaceholderText("default: scans")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_base_dir)
        base_row = QWidget()
        base_row_l = QHBoxLayout(base_row)
        base_row_l.setContentsMargins(0, 0, 0, 0)
        base_row_l.addWidget(self.base_dir_edit, 1)
        base_row_l.addWidget(browse_btn)
        form.addRow("Output folder", base_row)

        # Preferred/blacklisted sources (comma-separated)
        self.preferred_sources_edit = QLineEdit(", ".join(settings.preferred_sources or [])
        )
        self.preferred_sources_edit.setPlaceholderText(
            f"default: {', '.join(default_crawler.sources or ['wikipedia'])} (comma-separated)"
        )
        form.addRow("Preferred sources", self.preferred_sources_edit)

        self.blacklisted_sources_edit = QLineEdit(", ".join(settings.blacklisted_sources or [])
        )
        self.blacklisted_sources_edit.setPlaceholderText("default: none (comma-separated)")
        form.addRow("Blacklisted sources", self.blacklisted_sources_edit)

        # Visualization overrides
        self.viz_max_nodes_edit = QLineEdit(
            "" if settings.viz_max_nodes is None else str(settings.viz_max_nodes)
        )
        self.viz_max_nodes_edit.setPlaceholderText(
            f"(empty = default: {default_viz.max_nodes}) range: 1 - 10000"
        )
        form.addRow("Max nodes", self.viz_max_nodes_edit)

        self.viz_min_edge_confidence_edit = QLineEdit(
            "" if settings.viz_min_edge_confidence is None else str(settings.viz_min_edge_confidence)
        )
        self.viz_min_edge_confidence_edit.setPlaceholderText(
            "(empty = default: no filter) range: 0.0 - 1.0"
        )
        form.addRow("Min edge confidence", self.viz_min_edge_confidence_edit)

        self.viz_remove_isolated_edit = QCheckBox("Remove isolated nodes")
        if settings.viz_remove_isolated_nodes is None:
            self.viz_remove_isolated_edit.setChecked(True)
        else:
            self.viz_remove_isolated_edit.setChecked(bool(settings.viz_remove_isolated_nodes))
        form.addRow("", self.viz_remove_isolated_edit)

        # Phase 2
        self.enable_phase2_edit = QCheckBox("Enable Phase 2")
        if settings.enable_phase2 is None:
            self.enable_phase2_edit.setChecked(True)
        else:
            self.enable_phase2_edit.setChecked(bool(settings.enable_phase2))
        form.addRow("", self.enable_phase2_edit)

        self.phase2_max_pages_edit = QLineEdit(
            "" if settings.phase2_max_pages is None else str(settings.phase2_max_pages)
        )
        self.phase2_max_pages_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.phase2_max_pages}) range: 0 - 5000"
        )
        form.addRow("Phase 2 max pages", self.phase2_max_pages_edit)

        self.phase2_concurrent_tabs_edit = QLineEdit(
            "" if settings.phase2_concurrent_tabs is None else str(settings.phase2_concurrent_tabs)
        )
        self.phase2_concurrent_tabs_edit.setPlaceholderText(
            "(empty = default: same as Phase 1) range: 1 - 50"
        )
        form.addRow("Phase 2 concurrent tabs", self.phase2_concurrent_tabs_edit)

        # Document handling
        self.document_min_relevance_edit = QLineEdit(
            "" if settings.document_min_relevance is None else str(settings.document_min_relevance)
        )
        self.document_min_relevance_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.document_min_relevance}) range: 0.0 - 1.0"
        )
        form.addRow("Document min relevance", self.document_min_relevance_edit)

        self.downloads_prune_irrelevant_edit = QCheckBox("Prune irrelevant downloads")
        if settings.downloads_prune_irrelevant is None:
            self.downloads_prune_irrelevant_edit.setChecked(True)
        else:
            self.downloads_prune_irrelevant_edit.setChecked(bool(settings.downloads_prune_irrelevant))
        form.addRow("", self.downloads_prune_irrelevant_edit)

        self.downloads_prune_mode_edit = QComboBox()
        self.downloads_prune_mode_edit.addItem("move")
        self.downloads_prune_mode_edit.addItem("delete")
        mode = (settings.downloads_prune_mode or "move").strip().lower()
        self.downloads_prune_mode_edit.setCurrentIndex(1 if mode == "delete" else 0)
        form.addRow("Prune mode", self.downloads_prune_mode_edit)

        self.web_search_max_pdf_downloads_edit = QLineEdit(
            "" if settings.web_search_max_pdf_downloads is None else str(settings.web_search_max_pdf_downloads)
        )
        self.web_search_max_pdf_downloads_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.web_search_max_pdf_downloads}) range: 0 - 1000"
        )
        form.addRow("Max PDF downloads", self.web_search_max_pdf_downloads_edit)

        self.web_search_min_relevance_edit = QLineEdit(
            "" if settings.web_search_min_relevance is None else str(settings.web_search_min_relevance)
        )
        self.web_search_min_relevance_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.web_search_min_relevance}) range: 0.0 - 1.0"
        )
        form.addRow("Web search min relevance", self.web_search_min_relevance_edit)

        # Confidence thresholds
        self.nlp_min_confidence_edit = QLineEdit(
            "" if settings.nlp_min_confidence is None else str(settings.nlp_min_confidence)
        )
        self.nlp_min_confidence_edit.setPlaceholderText(
            f"(empty = default: {default_nlp.min_confidence}) range: 0.0 - 1.0"
        )
        form.addRow("NLP min confidence", self.nlp_min_confidence_edit)

        self.nlp_min_relation_confidence_edit = QLineEdit(
            "" if settings.nlp_min_relation_confidence is None else str(settings.nlp_min_relation_confidence)
        )
        self.nlp_min_relation_confidence_edit.setPlaceholderText(
            f"(empty = default: {default_nlp.min_relation_confidence}) range: 0.0 - 1.0"
        )
        form.addRow("Min relation confidence", self.nlp_min_relation_confidence_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def _browse_base_dir(self) -> None:
        start = self.base_dir_edit.text().strip() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if path:
            self.base_dir_edit.setText(path)

    def result_settings(self) -> UserSettings:
        base_dir = self.base_dir_edit.text().strip() or "scans"

        def _split_csv(text: str) -> list[str]:
            parts = [p.strip() for p in (text or "").split(",")]
            return [p for p in parts if p]

        preferred = _split_csv(self.preferred_sources_edit.text())
        blacklisted = _split_csv(self.blacklisted_sources_edit.text())

        max_nodes_raw = (self.viz_max_nodes_edit.text() or "").strip()
        max_nodes: Optional[int]
        if not max_nodes_raw:
            max_nodes = None
        else:
            max_nodes = int(max_nodes_raw)

        min_edge_raw = (self.viz_min_edge_confidence_edit.text() or "").strip()
        min_edge: Optional[float]
        if not min_edge_raw:
            min_edge = None
        else:
            min_edge = float(min_edge_raw)

        remove_isolated = bool(self.viz_remove_isolated_edit.isChecked())

        enable_phase2 = bool(self.enable_phase2_edit.isChecked())

        phase2_max_pages_raw = (self.phase2_max_pages_edit.text() or "").strip()
        phase2_max_pages: Optional[int]
        phase2_max_pages = None if not phase2_max_pages_raw else int(phase2_max_pages_raw)

        phase2_tabs_raw = (self.phase2_concurrent_tabs_edit.text() or "").strip()
        phase2_concurrent_tabs: Optional[int]
        phase2_concurrent_tabs = None if not phase2_tabs_raw else int(phase2_tabs_raw)

        doc_min_rel_raw = (self.document_min_relevance_edit.text() or "").strip()
        document_min_relevance: Optional[float]
        document_min_relevance = None if not doc_min_rel_raw else float(doc_min_rel_raw)

        downloads_prune_irrelevant = bool(self.downloads_prune_irrelevant_edit.isChecked())
        downloads_prune_mode = str(self.downloads_prune_mode_edit.currentText() or "move").strip().lower()

        max_pdf_raw = (self.web_search_max_pdf_downloads_edit.text() or "").strip()
        web_search_max_pdf_downloads: Optional[int]
        web_search_max_pdf_downloads = None if not max_pdf_raw else int(max_pdf_raw)

        web_min_rel_raw = (self.web_search_min_relevance_edit.text() or "").strip()
        web_search_min_relevance: Optional[float]
        web_search_min_relevance = None if not web_min_rel_raw else float(web_min_rel_raw)

        nlp_min_conf_raw = (self.nlp_min_confidence_edit.text() or "").strip()
        nlp_min_confidence: Optional[float]
        nlp_min_confidence = None if not nlp_min_conf_raw else float(nlp_min_conf_raw)

        nlp_min_rel_conf_raw = (self.nlp_min_relation_confidence_edit.text() or "").strip()
        nlp_min_relation_confidence: Optional[float]
        nlp_min_relation_confidence = None if not nlp_min_rel_conf_raw else float(nlp_min_rel_conf_raw)

        return UserSettings(
            base_dir=base_dir,
            preferred_sources=preferred,
            blacklisted_sources=blacklisted,
            viz_max_nodes=max_nodes,
            viz_min_edge_confidence=min_edge,
            viz_remove_isolated_nodes=remove_isolated,
            enable_phase2=enable_phase2,
            phase2_max_pages=phase2_max_pages,
            phase2_concurrent_tabs=phase2_concurrent_tabs,
            document_min_relevance=document_min_relevance,
            downloads_prune_irrelevant=downloads_prune_irrelevant,
            downloads_prune_mode=downloads_prune_mode,
            web_search_max_pdf_downloads=web_search_max_pdf_downloads,
            web_search_min_relevance=web_search_min_relevance,
            nlp_min_confidence=nlp_min_confidence,
            nlp_min_relation_confidence=nlp_min_relation_confidence,
        )


class QtLogHandler(logging.Handler):
    def __init__(self, emit_line):
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emit_line(msg)
        except Exception:
            # Never crash logging.
            pass


def _install_qt_log_handler(emit_line) -> QtLogHandler:
    """Route logs to the UI and avoid noisy console handlers."""
    handler = QtLogHandler(emit_line)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Remove StreamHandlers from known loggers (console is not useful for a GUI).
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            root.removeHandler(h)

    for obj in logging.root.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            for h in list(obj.handlers):
                if isinstance(h, logging.StreamHandler):
                    obj.removeHandler(h)

    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler


class ScanWorker(QObject):
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    finished = pyqtSignal(str, str)  # scan_dir, html_path
    failed = pyqtSignal(str)

    def __init__(self, request: ScanRequest):
        super().__init__()
        self._request = request

    def run(self) -> None:
        try:
            _install_qt_log_handler(lambda line: self.log.emit(line))

            self.status.emit("Running scan...")

            from scan_manager import get_scan_paths
            from main_refactored import main as run_main

            scan_paths = get_scan_paths(
                self._request.query,
                base_dir=self._request.base_dir,
                add_timestamp=False,
            )

            # Run the async pipeline inside this worker thread.
            asyncio.run(
                run_main(
                    query=self._request.query,
                    max_pages=self._request.max_pages,
                    add_timestamp=False,
                    base_dir=self._request.base_dir,
                    browser_headless=self._request.headless,
                    enable_web_search=self._request.enable_web_search,
                    web_search_download_pdfs=self._request.download_pdfs,
                    preferred_sources=list(self._request.preferred_sources),
                    blacklisted_sources=list(self._request.blacklisted_sources),
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
            # Prefer the Bokeh visualization because it contains the full UI
            # (filters, panels, pinned items, etc.). The Plotly export is still
            # available via “Open graph…”.
            if scan_paths["viz_file"].exists():
                html_path = str(scan_paths["viz_file"].resolve())
            elif scan_paths["interactive_viz_file"].exists():
                html_path = str(scan_paths["interactive_viz_file"].resolve())

            self.status.emit("Done")
            self.finished.emit(str(scan_paths["scan_dir"].resolve()), html_path)
        except Exception as e:
            self.status.emit("Error")
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from version import CURRENT_VERSION
        self.setWindowTitle(f"SearchANDGraph - {CURRENT_VERSION}")
        self.resize(1200, 780)

        self._thread: Optional[QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._last_scan_dir: Optional[Path] = None
        self._settings = self._load_settings()
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._auto_fit_current_page)

        # Re-fit if the window is moved to a different screen (DPI/available geometry changes).
        self._screen_connected = False

        # Controls
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter a query (e.g., Nvidia)")

        self.max_pages = QSpinBox()
        self.max_pages.setRange(1, 5000)
        self.max_pages.setValue(10)

        self.headless = QCheckBox("Headless browser")
        self.headless.setChecked(True)

        self.enable_web_search = QCheckBox("Enable web search")
        self.enable_web_search.setChecked(True)

        self.download_pdfs = QCheckBox("Download PDFs from web search")
        self.download_pdfs.setChecked(True)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)

        self.scan_select = QComboBox()
        self.scan_select.setMinimumWidth(260)
        self.scan_select.currentIndexChanged.connect(self._on_scan_selected)
        self._scan_combo_updating = False

        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        
        # This is for testing must be removed in production
        self.status_label = QLabel("Idle")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Log panel
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        # Viewer
        self.web_view = QWebEngineView()
        self.web_view.setPage(ExternalLinksPage(self.web_view, emit_log=self._append_log))
        # Some local HTML (e.g., older Bokeh exports) loads JS/CSS from https:// URLs.
        # Qt WebEngine can block this by default for file:// content.
        self.web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            True,
        )
        self.web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls,
            True,
        )
        self.web_view.loadFinished.connect(self._on_web_load_finished)
        self.web_view.setHtml("<html><body><h3>Run a scan to view the graph.</h3></body></html>")

        # Layout
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(QLabel("Query:"))
        controls_layout.addWidget(self.query_input, 2)
        controls_layout.addWidget(QLabel("Max pages:"))
        controls_layout.addWidget(self.max_pages)
        controls_layout.addWidget(self.headless)
        controls_layout.addWidget(self.enable_web_search)
        controls_layout.addWidget(self.download_pdfs)
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(QLabel("Scans:"))
        controls_layout.addWidget(self.scan_select, 1)
        controls_layout.addWidget(self.open_folder_btn)

        self.bottom_split = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_split.addWidget(self.log_view)
        self.bottom_split.addWidget(self.web_view)
        self.bottom_split.setStretchFactor(0, 1)
        self.bottom_split.setStretchFactor(1, 5)
        self.bottom_split.setSizes([320, 880])
        # When the user drags the splitter handle, the main window isn't resized,
        # but the web view is. Trigger auto-fit on splitter moves.
        self.bottom_split.splitterMoved.connect(lambda *_: self._fit_timer.start(120))
        # Also react to internal widget resizes.
        self.bottom_split.installEventFilter(self)
        self.web_view.installEventFilter(self)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(controls)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.bottom_split, 1)

        self.setCentralWidget(root)

        # Menu (minimal)
        file_menu = self.menuBar().addMenu("File")

        options_action = QAction("Options…", self)
        options_action.triggered.connect(self._open_options)
        file_menu.addAction(options_action)

        self._recent_menu = file_menu.addMenu("Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Populate scan dropdown on startup.
        self._refresh_scan_dropdown()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Debounce auto-fit while resizing.
        self._fit_timer.start(150)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Connect once we have a native window handle.
        if not self._screen_connected:
            try:
                handle = self.windowHandle()
                if handle is not None:
                    handle.screenChanged.connect(lambda *_: self._fit_timer.start(200))
                    self._screen_connected = True
            except Exception:
                pass

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        # QSplitter moves / QWebEngineView resizes don't always trigger MainWindow.resizeEvent.
        if obj in (getattr(self, "bottom_split", None), getattr(self, "web_view", None)):
            if event.type() == QEvent.Type.Resize:
                self._fit_timer.start(120)
        return super().eventFilter(obj, event)

    def _on_web_load_finished(self, ok: bool) -> None:
            if not ok:
                    return

            self._install_embedded_click_workarounds()

            # Fit a few times after load so Bokeh has time to lay out widgets.
            self._fit_timer.start(150)
            QTimer.singleShot(650, self._auto_fit_current_page)
            QTimer.singleShot(1600, self._auto_fit_current_page)

    def _install_embedded_click_workarounds(self) -> None:
            """Install shadow-DOM-safe click handlers inside the embedded page.

            Bokeh widgets render in shadow roots; plain DOM delegation often misses anchors
            and remove buttons. This installs a capture listener and forces navigation for
            file/http(s) links so our QWebEnginePage can open them externally.
            """
            page = self.web_view.page()
            if page is None:
                    return

            js = r"""
(() => {
    if (window.__kgQtClickFixInstalled) return 'already';
    window.__kgQtClickFixInstalled = true;

    document.addEventListener('click', (e) => {
        const path = (e && e.composedPath) ? e.composedPath() : [];

        // 1) Remove pinned item
        for (const n of path) {
            try {
                if (n && n.getAttribute && n.getAttribute('data-action') === 'remove') {
                    const targetId = n.getAttribute('data-target');
                    console.log('[KG] remove click', targetId);
                    if (!targetId) return;
                    e.preventDefault();
                    e.stopPropagation();

                    // Prefer stateful removal (keeps the Bokeh Div model in sync)
                    // so removed pins do NOT reappear when pinning new connections.
                    try {
                        if (window.__kgPinnedRemoveById && typeof window.__kgPinnedRemoveById === 'function') {
                            window.__kgPinnedRemoveById(targetId);
                            console.log('[KG] stateful remove ok');
                            return;
                        }
                    } catch (_) {
                        // ignore
                    }

                    const root = (n.getRootNode && n.getRootNode()) ? n.getRootNode() : document;
                    let el = null;
                    if (root && root.querySelector) el = root.querySelector('#' + targetId);
                    if (!el) el = document.getElementById(targetId);
                    console.log('[KG] remove found', !!el);
                    if (el) el.remove();
                    return;
                }
            } catch (_) {
                // ignore
            }
        }

        // 2) Open external/file links (even inside shadow DOM)
        for (const n of path) {
            try {
                if (n && n.tagName && String(n.tagName).toLowerCase() === 'a' && n.href) {
                    const href = String(n.href || '');
                    if (!href) return;
                    if (!(href.startsWith('http://') || href.startsWith('https://') || href.startsWith('file://'))) return;
                    console.log('[KG] link click', href);
                    e.preventDefault();
                    e.stopPropagation();
                    // Force a main-frame navigation (ExternalLinksPage will open externally)
                    window.location.href = href;
                    return;
                }
            } catch (_) {
                // ignore
            }
        }
    }, true);

    return 'installed';
})()
"""

            page.runJavaScript(js, lambda r: self._append_log(f"Click fix: {r}"))

    def _auto_fit_current_page(self) -> None:
        """Scale the page to fit the available view (reduces scrolling)."""
        page = self.web_view.page()
        if page is None:
            return

        js = """
(() => {
    const de = document.documentElement;
    const body = document.body;
    const scrollWidth = Math.max(
        de ? de.scrollWidth : 0,
        body ? body.scrollWidth : 0,
        de ? de.offsetWidth : 0,
        body ? body.offsetWidth : 0,
    );
    const scrollHeight = Math.max(
        de ? de.scrollHeight : 0,
        body ? body.scrollHeight : 0,
        de ? de.offsetHeight : 0,
        body ? body.offsetHeight : 0,
    );
    return {scrollWidth, scrollHeight};
})()
"""

        def _apply_zoom(result) -> None:
            try:
                if not isinstance(result, dict):
                    return
                content_w = float(result.get("scrollWidth") or 0)
                content_h = float(result.get("scrollHeight") or 0)
                # If the page didn't report sensible size, don't change zoom.
                if content_w <= 0 or content_h <= 0:
                    return

                view_w = max(1.0, float(self.web_view.size().width()))
                view_h = max(1.0, float(self.web_view.size().height()))

                # Fit-to-view: show the whole UI without scrolling where possible.
                zoom_w = view_w / content_w
                zoom_h = view_h / content_h
                zoom = min(zoom_w, zoom_h)
                # Clamp: avoid unreadably small text or absurdly large zoom.
                zoom = max(0.25, min(1.8, zoom))

                if abs(self.web_view.zoomFactor() - zoom) >= 0.02:
                    self.web_view.setZoomFactor(zoom)
            except Exception:
                pass

        page.runJavaScript(js, _apply_zoom)

    def _append_log(self, line: str) -> None:
        self.log_view.append(line)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_start(self) -> None:
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Missing query", "Please enter a query.")
            return

        if self._thread is not None:
            QMessageBox.information(self, "Scan running", "A scan is already running.")
            return

        self.log_view.clear()
        self.open_folder_btn.setEnabled(False)
        self._last_scan_dir = None

        request = ScanRequest(
            query=query,
            max_pages=int(self.max_pages.value()),
            headless=bool(self.headless.isChecked()),
            enable_web_search=bool(self.enable_web_search.isChecked()),
            download_pdfs=bool(self.download_pdfs.isChecked()),
            base_dir=str(self._settings.base_dir or "scans"),
            preferred_sources=tuple(self._settings.preferred_sources or ()),
            blacklisted_sources=tuple(self._settings.blacklisted_sources or ()),
            viz_max_nodes=self._settings.viz_max_nodes,
            viz_min_edge_confidence=self._settings.viz_min_edge_confidence,
            viz_remove_isolated_nodes=self._settings.viz_remove_isolated_nodes,
            enable_phase2=self._settings.enable_phase2,
            phase2_max_pages=self._settings.phase2_max_pages,
            phase2_concurrent_tabs=self._settings.phase2_concurrent_tabs,
            document_min_relevance=self._settings.document_min_relevance,
            downloads_prune_irrelevant=self._settings.downloads_prune_irrelevant,
            downloads_prune_mode=self._settings.downloads_prune_mode,
            web_search_max_pdf_downloads=self._settings.web_search_max_pdf_downloads,
            web_search_min_relevance=self._settings.web_search_min_relevance,
            nlp_min_confidence=self._settings.nlp_min_confidence,
            nlp_min_relation_confidence=self._settings.nlp_min_relation_confidence,
        )

        self._set_status("Queued")

        self._thread = QThread()
        self._worker = ScanWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.status.connect(self._set_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._worker.finished.connect(self._cleanup_thread)
        self._worker.failed.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self) -> None:
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait(2000)
        self._thread = None
        self._worker = None

    def _settings_path(self) -> Path:
        # Keep it next to the app script for a simple, portable setup.
        return Path(__file__).resolve().parent / "user_settings.json"

    def _load_settings(self) -> UserSettings:
        path = self._settings_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return UserSettings.from_dict(data)
        except Exception:
            pass
        return UserSettings()

    def _save_settings(self) -> None:
        path = self._settings_path()
        try:
            path.write_text(json.dumps(self._settings.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Options", f"Failed to save settings: {e}")

    def _open_options(self) -> None:
        dlg = OptionsDialog(self, self._settings)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            updated = dlg.result_settings()
        except Exception as e:
            QMessageBox.warning(self, "Options", f"Invalid options: {e}")
            return

        self._settings = updated
        self._save_settings()
        self._refresh_scan_dropdown()
        self._append_log("Options saved")

    def _on_finished(self, scan_dir: str, html_path: str) -> None:
        self._last_scan_dir = Path(scan_dir)
        self.open_folder_btn.setEnabled(True)

        if html_path and Path(html_path).exists():
            self._log_html_hints(Path(html_path))
            url = QUrl.fromLocalFile(html_path)
            self.web_view.load(url)
            self._append_log(f"Loaded graph: {html_path}")
            self._add_recent_file(Path(html_path))
            self._refresh_scan_dropdown(select_path=Path(html_path))
        else:
            self.web_view.setHtml(
                "<html><body><h3>Scan finished, but no HTML graph was found.</h3></body></html>"
            )
            self._append_log("No HTML output found.")
            self._refresh_scan_dropdown()

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Scan failed", message)

    def _on_open_folder(self) -> None:
        if not self._last_scan_dir:
            return
        path = str(self._last_scan_dir)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f"open \"{path}\"")
            else:
                os.system(f"xdg-open \"{path}\"")
        except Exception as e:
            QMessageBox.warning(self, "Open folder failed", str(e))

    def _scan_html_files(self) -> list[Path]:
        base = Path(self._settings.base_dir or "scans")
        if not base.exists() or not base.is_dir():
            return []

        results: list[Path] = []
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if child.name == "_tmp":
                continue
            try:
                htmls = sorted(
                    [p for p in child.iterdir() if p.is_file() and p.suffix.lower() in {".html", ".htm"}],
                    key=lambda p: p.name.lower(),
                )
                results.extend(htmls)
            except Exception:
                continue
        return results

    def _refresh_scan_dropdown(self, select_path: Path | None = None) -> None:
        try:
            self._scan_combo_updating = True
            self.scan_select.blockSignals(True)
            self.scan_select.clear()
            self.scan_select.addItem("Select scan…", None)

            html_files = self._scan_html_files()
            for p in html_files:
                label = f"{p.parent.name} / {p.name}"
                self.scan_select.addItem(label, str(p.resolve()))

            # Preselect if requested (or try to keep current selection).
            target = None
            if select_path is not None:
                try:
                    target = str(select_path.resolve())
                except Exception:
                    target = str(select_path)

            if target:
                for i in range(self.scan_select.count()):
                    if self.scan_select.itemData(i) == target:
                        self.scan_select.setCurrentIndex(i)
                        break
        finally:
            self.scan_select.blockSignals(False)
            self._scan_combo_updating = False

    def _on_scan_selected(self, _index: int) -> None:
        if getattr(self, "_scan_combo_updating", False):
            return

        data = self.scan_select.currentData()
        if not data:
            return

        selected = Path(str(data))
        if not selected.exists():
            self._append_log(f"Scan HTML missing: {selected}")
            return

        self._last_scan_dir = selected.parent
        self.open_folder_btn.setEnabled(True)
        self._set_status("Viewing existing graph")
        self._log_html_hints(selected)
        self.web_view.load(QUrl.fromLocalFile(str(selected.resolve())))
        self._append_log(f"Loaded graph: {selected}")
        self._add_recent_file(selected)

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "_recent_menu") or self._recent_menu is None:
            return

        self._recent_menu.clear()
        items = list(self._settings.recent_files or [])
        # Keep only existing HTML files.
        pruned: list[str] = []
        for s in items:
            try:
                p = Path(s)
                if p.exists() and p.is_file() and p.suffix.lower() in {".html", ".htm"}:
                    pruned.append(str(p))
            except Exception:
                continue
        if pruned != items:
            self._settings.recent_files = pruned[:10]
            self._save_settings()

        if not pruned:
            empty = QAction("(empty)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return

        for s in pruned[:10]:
            p = Path(s)
            label = f"{p.parent.name} / {p.name}"
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, path=s: self._open_recent_path(path))
            self._recent_menu.addAction(act)

    def _add_recent_file(self, path: Path) -> None:
        try:
            p = path.resolve()
        except Exception:
            p = path

        if p.suffix.lower() not in {".html", ".htm"}:
            return
        s = str(p)

        items = list(self._settings.recent_files or [])
        # De-dupe while keeping newest at front.
        items = [x for x in items if x != s]
        items.insert(0, s)
        self._settings.recent_files = items[:10]
        self._save_settings()
        self._rebuild_recent_menu()

    def _open_recent_path(self, path_str: str) -> None:
        p = Path(path_str)
        if not p.exists():
            self._append_log(f"Recent file missing: {p}")
            # Rebuild to prune missing entries.
            self._rebuild_recent_menu()
            return

        self._last_scan_dir = p.parent
        self.open_folder_btn.setEnabled(True)
        self._set_status("Viewing existing graph")
        self._log_html_hints(p)
        self.web_view.load(QUrl.fromLocalFile(str(p.resolve())))
        self._append_log(f"Loaded graph: {p}")
        self._add_recent_file(p)
        self._refresh_scan_dropdown(select_path=p)

    def _log_html_hints(self, html_path: Path) -> None:
        """Add non-blocking hints for common embedded-viewer issues."""
        try:
            head = html_path.read_text(encoding="utf-8", errors="ignore")[:4096]
        except Exception:
            return

        if "cdn.bokeh.org" in head:
            self._append_log(
                "NOTE: This Bokeh HTML loads scripts from https://cdn.bokeh.org. "
                "If it fails to render, rerun the scan to regenerate with inline resources."
            )


def main() -> None:
    # Hold a named mutex for the lifetime of this process so Inno Setup (AppMutex)
    # can detect/rendezvous with a running app during updates.
    _mutex_handle = hold_app_mutex()

    app = QApplication(sys.argv)

    # Optional escape hatch for development: run update check after QApplication exists
    if not os.environ.get("SAG_DISABLE_UPDATES"):
        try:
            release = check_for_updates(timeout_s=4.0)
            if release:
                try:
                    resp = QMessageBox.question(
                        None,
                        "Update Found",
                        "A new update is available. Download and install now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if resp == QMessageBox.StandardButton.Yes:
                        dlg = UpdateDialog(None, release)
                        dlg.exec()
                        sys.exit(0)
                except Exception:
                    # Fallback: directly show UpdateDialog
                    dlg = UpdateDialog(None, release)
                    dlg.exec()
                    sys.exit(0)
        except Exception:
            pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
