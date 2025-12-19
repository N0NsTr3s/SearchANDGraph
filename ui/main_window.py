from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QUrl, QTimer, QEvent
from PyQt6.QtGui import QAction, QDesktopServices, QIcon
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
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

from ui.models import ScanRequest, UserSettings
from ui.options import OptionsDialog
from ui.web import ExternalLinksPage
from ui.worker import ScanWorker
from ui.screenshot_carousel import ScreenshotCarousel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from version import CURRENT_VERSION

        self.setWindowTitle(f"SearchANDGraph - {CURRENT_VERSION}")

        try:
            logo_path = Path(__file__).resolve().parent / "assets" / "logoSAG32x32.ico"
            if logo_path.exists():
                self.setWindowIcon(QIcon(str(logo_path)))
        except Exception:
            pass

        self.resize(1280, 720)

        self._thread: Optional[QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._last_scan_dir: Optional[Path] = None
        self._settings = self._load_settings()
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._auto_fit_current_page)

        self._screen_connected = False

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter a query (e.g., Nvidia)")

        self.start_url_input = QLineEdit()
        self.start_url_input.setPlaceholderText("Optional start URL (e.g., https://example.com)")

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

        self.status_label = QLabel("Idle")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.web_view = QWebEngineView()
        self.web_view.setPage(ExternalLinksPage(self.web_view, emit_log=self._append_log))
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

        self._preview_dir: Path | None = None
        self._preview_running = False
        self.preview_carousel = ScreenshotCarousel()
        self.preview_carousel.hide()

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.preview_carousel, 0)
        right_layout.addWidget(self.web_view, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(QLabel("Query:"))
        controls_layout.addWidget(self.query_input, 2)
        controls_layout.addWidget(QLabel("Start URL:"))
        controls_layout.addWidget(self.start_url_input, 2)
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
        self.bottom_split.addWidget(right_pane)
        self.bottom_split.setStretchFactor(0, 1)
        self.bottom_split.setStretchFactor(1, 5)
        self.bottom_split.setSizes([320, 880])
        self.bottom_split.splitterMoved.connect(lambda *_: self._fit_timer.start(120))
        self.bottom_split.installEventFilter(self)
        self.web_view.installEventFilter(self)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(controls)
        root_layout.addWidget(self.status_label)
        root_layout.addWidget(self.bottom_split, 1)

        self.setCentralWidget(root)

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

        self._refresh_scan_dropdown()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_timer.start(150)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._screen_connected:
            try:
                handle = self.windowHandle()
                if handle is not None:
                    handle.screenChanged.connect(lambda *_: self._fit_timer.start(200))
                    self._screen_connected = True
            except Exception:
                pass

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj in (getattr(self, "bottom_split", None), getattr(self, "web_view", None)):
            if event.type() == QEvent.Type.Resize:
                self._fit_timer.start(120)
        return super().eventFilter(obj, event)

    def _on_web_load_finished(self, ok: bool) -> None:
        if not ok:
            return

        self._install_embedded_click_workarounds()

        self._fit_timer.start(150)
        QTimer.singleShot(650, self._auto_fit_current_page)
        QTimer.singleShot(1600, self._auto_fit_current_page)

    def _install_embedded_click_workarounds(self) -> None:
        page = self.web_view.page()
        if page is None:
            return

        js = r"""
(() => {
    if (window.__kgQtClickFixInstalled) return 'already';
    window.__kgQtClickFixInstalled = true;

    document.addEventListener('click', (e) => {
        const path = (e && e.composedPath) ? e.composedPath() : [];

        for (const n of path) {
            try {
                if (n && n.getAttribute && n.getAttribute('data-action') === 'remove') {
                    const targetId = n.getAttribute('data-target');
                    console.log('[KG] remove click', targetId);
                    if (!targetId) return;
                    e.preventDefault();
                    e.stopPropagation();

                    try {
                        if (window.__kgPinnedRemoveById && typeof window.__kgPinnedRemoveById === 'function') {
                            window.__kgPinnedRemoveById(targetId);
                            console.log('[KG] stateful remove ok');
                            return;
                        }
                    } catch (_) {
                    }

                    const root = (n && n.getRootNode && n.getRootNode()) ? n.getRootNode() : document;
                    let el = null;
                    if (root && root.querySelector) el = root.querySelector('#' + targetId);
                    if (!el) el = document.getElementById(targetId);
                    console.log('[KG] remove found', !!el);
                    if (el) el.remove();
                    return;
                }
            } catch (_) {
            }
        }

        for (const n of path) {
            try {
                if (n && n.tagName && String(n.tagName).toLowerCase() === 'a' && n.href) {
                    const href = String(n.href || '');
                    if (!href) return;
                    if (!(href.startsWith('http://') || href.startsWith('https://') || href.startsWith('file://'))) return;
                    console.log('[KG] link click', href);
                    e.preventDefault();
                    e.stopPropagation();
                    window.location.href = href;
                    return;
                }
            } catch (_) {
            }
        }
    }, true);

    return 'installed';
})()
"""

        page.runJavaScript(js, lambda r: self._append_log(f"Click fix: {r}"))

    def _auto_fit_current_page(self) -> None:
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
                if content_w <= 0 or content_h <= 0:
                    return

                view_w = max(1.0, float(self.web_view.size().width()))
                view_h = max(1.0, float(self.web_view.size().height()))

                zoom_w = view_w / content_w
                zoom_h = view_h / content_h
                zoom = min(zoom_w, zoom_h)
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

        start_url = self.start_url_input.text().strip()

        if self._thread is not None:
            QMessageBox.information(self, "Scan running", "A scan is already running.")
            return

        self.log_view.clear()
        self.open_folder_btn.setEnabled(False)
        self._last_scan_dir = None

        # Prepare preview directory under the scan folder and clear stale files.
        try:
            from scraper.scan_manager import get_scan_paths
            import shutil

            scan_paths = get_scan_paths(
                query=query,
                base_dir=str(self._settings.base_dir or "scans"),
                add_timestamp=False,
            )
            self._preview_dir = Path(scan_paths["scan_dir"]) / "_ui_previews"
            shutil.rmtree(self._preview_dir, ignore_errors=True)
            self._preview_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._preview_dir = None

        request = ScanRequest(
            query=query,
            start_url=start_url or None,
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
            preview_enabled=True,
            preview_interval_seconds=2,
        )

        self._set_status("Queued")

        self._thread = QThread()
        self._worker = ScanWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.status.connect(self._set_status)
        self._worker.started.connect(self._on_scan_started)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._worker.finished.connect(self._cleanup_thread)
        self._worker.failed.connect(self._cleanup_thread)

        self._thread.start()

    def _on_scan_started(self, scan_dir: str) -> None:
        # Show previews while the scan is running.
        try:
            self._preview_running = True
            self._preview_dir = Path(scan_dir) / "_ui_previews"
            self.preview_carousel.show()
            self.web_view.hide()
            self.preview_carousel.start(self._preview_dir, rotate_seconds=2)
        except Exception:
            pass

    def _cleanup_thread(self) -> None:
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait(2000)
        self._thread = None
        self._worker = None

    def _settings_path(self) -> Path:
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
        try:
            if self._preview_running:
                self._preview_running = False
                self.preview_carousel.stop()
                self.preview_carousel.hide()
                self.web_view.show()
        except Exception:
            pass

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
        try:
            if self._preview_running:
                self._preview_running = False
                self.preview_carousel.stop()
                self.preview_carousel.hide()
                self.web_view.show()
        except Exception:
            pass
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
        items = [x for x in items if x != s]
        items.insert(0, s)
        self._settings.recent_files = items[:10]
        self._save_settings()
        self._rebuild_recent_menu()

    def _open_recent_path(self, path_str: str) -> None:
        p = Path(path_str)
        if not p.exists():
            self._append_log(f"Recent file missing: {p}")
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
        try:
            head = html_path.read_text(encoding="utf-8", errors="ignore")[:4096]
        except Exception:
            return

        if "cdn.bokeh.org" in head:
            self._append_log(
                "NOTE: This Bokeh HTML loads scripts from https://cdn.bokeh.org. "
                "If it fails to render, rerun the scan to regenerate with inline resources."
            )
