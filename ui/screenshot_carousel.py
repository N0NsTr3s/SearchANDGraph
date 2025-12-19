from __future__ import annotations

from collections import deque
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ScreenshotCarousel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._watch_dir: Path | None = None
        self._pixmaps: deque[QPixmap] = deque()
        self._current_index: int = -1

        self._label = QLabel("Capturing previews…")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumHeight(220)
        self._label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(450)
        self._poll_timer.timeout.connect(self._poll_dir)

        self._rotate_timer = QTimer(self)
        self._rotate_timer.timeout.connect(self._next)

    def start(self, watch_dir: Path, rotate_seconds: int = 2) -> None:
        self._watch_dir = Path(watch_dir)
        self._pixmaps.clear()
        self._current_index = -1
        self._label.setText("Capturing previews…")
        self._label.setPixmap(QPixmap())

        # Start timers
        self._poll_timer.start()
        self._rotate_timer.setInterval(max(250, int(rotate_seconds) * 1000))
        self._rotate_timer.start()

        self._poll_dir()
        self._next()

    def stop(self) -> None:
        self._poll_timer.stop()
        self._rotate_timer.stop()
        self._watch_dir = None
        self._pixmaps.clear()
        self._current_index = -1
        self._label.setPixmap(QPixmap())
        self._label.setText("")

    def _poll_dir(self) -> None:
        watch_dir = self._watch_dir
        if watch_dir is None:
            return

        try:
            if not watch_dir.exists() or not watch_dir.is_dir():
                return

            # Load any new screenshots and delete files immediately after loading.
            paths = sorted(
                [p for p in watch_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                key=lambda p: (p.stat().st_mtime, p.name.lower()),
            )

            for p in paths:
                pm = QPixmap(str(p))
                try:
                    # Delete as soon as it is created/loaded.
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

                if pm.isNull():
                    continue

                self._pixmaps.append(pm)

            # Keep memory bounded.
            while len(self._pixmaps) > 120:
                self._pixmaps.popleft()

        except Exception:
            # UI widget: best-effort.
            return

    def _next(self) -> None:
        if not self._pixmaps:
            self._label.setText("Capturing previews…")
            self._label.setPixmap(QPixmap())
            return

        self._label.setText("")
        self._current_index = (self._current_index + 1) % len(self._pixmaps)
        self._render_current()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_current()

    def _render_current(self) -> None:
        if not self._pixmaps or self._current_index < 0:
            return

        pm = self._pixmaps[self._current_index]
        if pm.isNull():
            return

        scaled = pm.scaled(self._label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(scaled)
