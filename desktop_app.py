"""Entrypoint module for the desktop UI.

This file is intentionally small — most UI code has been moved into the `ui` package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from single_instance import hold_app_mutex
from updater import check_for_updates
from ui.updates import UpdateDialog
from ui.main_window import MainWindow


def main() -> None:
    _mutex_handle = hold_app_mutex()

    app = QApplication(sys.argv)
    try:
        ico_path = Path(__file__).resolve().parent / "assets" / "logoSAG32x32.ico"
        if ico_path.exists():
            app.setWindowIcon(QIcon(str(ico_path)))
    except Exception:
        pass

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
