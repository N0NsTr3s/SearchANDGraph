"""PyQt desktop UI for running SearchANDGraph scans.

Minimal UX:
- Enter query + basic crawl options
- Click Start to run a scan in the background
- Stream logs to the UI
- Display the generated HTML graph inside the app
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QUrl, QTimer, QEvent
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
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
)
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView


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

        self.setWindowTitle("SearchANDGraph")
        self.resize(1200, 780)

        self._thread: Optional[QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._last_scan_dir: Optional[Path] = None
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

        self.open_graph_btn = QPushButton("Open graph…")
        self.open_graph_btn.clicked.connect(self._on_open_graph)

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
        controls_layout.addWidget(self.open_graph_btn)
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
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

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

    def _on_finished(self, scan_dir: str, html_path: str) -> None:
        self._last_scan_dir = Path(scan_dir)
        self.open_folder_btn.setEnabled(True)

        if html_path and Path(html_path).exists():
            self._log_html_hints(Path(html_path))
            url = QUrl.fromLocalFile(html_path)
            self.web_view.load(url)
            self._append_log(f"Loaded graph: {html_path}")
        else:
            self.web_view.setHtml(
                "<html><body><h3>Scan finished, but no HTML graph was found.</h3></body></html>"
            )
            self._append_log("No HTML output found.")

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

    def _on_open_graph(self) -> None:
        start_dir = Path("scans")
        if self._last_scan_dir and self._last_scan_dir.exists():
            start_dir = self._last_scan_dir
        elif start_dir.exists():
            # Prefer the most recently modified scan directory.
            try:
                candidates = [p for p in start_dir.iterdir() if p.is_dir()]
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    start_dir = candidates[0]
            except Exception:
                pass

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open graph HTML",
            str(start_dir),
            "HTML files (*.html *.htm);;All files (*.*)",
        )
        if not file_path:
            return

        selected = Path(file_path)
        if not selected.exists():
            QMessageBox.warning(self, "File not found", str(selected))
            return

        self._last_scan_dir = selected.parent
        self.open_folder_btn.setEnabled(True)

        self._set_status("Viewing existing graph")
        self._log_html_hints(selected)
        self.web_view.load(QUrl.fromLocalFile(str(selected.resolve())))
        self._append_log(f"Opened graph: {selected}")

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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
