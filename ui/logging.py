from __future__ import annotations

import logging
from typing import Callable


class QtLogHandler(logging.Handler):
    def __init__(self, emit_line: Callable[[str], None]):
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emit_line(msg)
        except Exception:
            pass


def _install_qt_log_handler(emit_line) -> QtLogHandler:
    handler = QtLogHandler(emit_line)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

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
