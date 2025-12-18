from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QMessageBox


def _prompt_update_and_install(release_data: dict) -> None:
    try:
        # Prefer Qt dialog when running inside the GUI
        if QDialog is not None and QMessageBox is not None:
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

        mbox = ctypes.windll.user32.MessageBoxW
        resp = mbox(0, "A new update is available. Download and install now?", "Update Found", 0x04)
        if int(resp) == 6:
            from updater import perform_update

            perform_update(release_data, silent=True)
            sys.exit(0)
    except Exception:
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
            from updater import perform_update

            def cb(pct: int, msg: str) -> None:
                try:
                    self.progress.emit(int(pct), str(msg))
                except Exception:
                    pass

            result = perform_update(self._release, silent=False, progress_callback=cb)
            self.finished.emit(result or {})
        except Exception as e:
            self.error.emit(str(e))


class UpdateDialog(QDialog):
    def __init__(self, parent: Optional[QDialog], release_data: dict):
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

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        layout.addWidget(self._close_btn)

        self._thread = QThread()
        self._worker = UpdateWorker(release_data)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._thread.start()

        self._result: dict = {}

    def _on_progress(self, pct: int, msg: str) -> None:
        try:
            self._bar.setValue(int(pct))
            self._label.setText(str(msg))
        except Exception:
            pass

    def _on_finished(self, result: dict) -> None:
        self._result = result or {}
        action = self._result.get("action")
        if action in {"installer_launched", "installer"}:
            self._label.setText("Installer launched.\nPlease close this window and\nFollow the installer to complete the update.")
            self._bar.setValue(100)
            self._close_btn.setEnabled(True)
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:
                pass
            return

        self._label.setText("Update finished.")
        self._bar.setValue(100)
        self._close_btn.setEnabled(True)

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
