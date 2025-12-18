from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtGui import QDesktopServices


class ExternalLinksPage(QWebEnginePage):
    def __init__(self, parent=None, emit_log=None):
        super().__init__(parent)
        self._emit_log = emit_log
        self._popup_page: QWebEnginePage | None = None

    def _open_external(self, url: QUrl) -> bool:
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
        popup = QWebEnginePage(self.profile(), self)

        def _open_and_drop(url: QUrl):
            if url and url.isValid():
                ok = self._open_external(url)
                if self._emit_log:
                    if ok:
                        self._emit_log(f"Opened external link: {url.toString()}")
                    else:
                        self._emit_log(f"Failed to open external link: {url.toString()}")
            self._popup_page = None

        popup.urlChanged.connect(_open_and_drop)
        self._popup_page = popup
        return popup

    def acceptNavigationRequest(self, url: QUrl, nav_type, isMainFrame: bool) -> bool:  # type: ignore[override]
        try:
            scheme = (url.scheme() or "").lower()
            if scheme in {"http", "https", "file"}:
                current = self.url()
                current_str = current.toString() if current and current.isValid() else ""
                target_str = url.toString() if url and url.isValid() else ""

                if isMainFrame and current_str and target_str and target_str == current_str:
                    return super().acceptNavigationRequest(url, nav_type, isMainFrame)

                if scheme == "file" and target_str.lower().endswith(".html"):
                    return super().acceptNavigationRequest(url, nav_type, isMainFrame)

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
