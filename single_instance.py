"""Single-instance / installer coordination helpers.

Inno Setup can be configured with AppMutex=... to detect a running app.
Holding the named mutex for the lifetime of the process lets the installer wait
until the app exits before overwriting files.
"""

from __future__ import annotations

import ctypes


APP_MUTEX_NAME = "SearchANDGraphMutex"


def hold_app_mutex(name: str = APP_MUTEX_NAME) -> int:
    """Create (or open) a named mutex and return its handle.

    Keep the returned handle referenced for the lifetime of the app.
    """

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, False, name)
    return int(handle)


def mutex_already_exists() -> bool:
    """Return True if the last mutex creation found an existing mutex."""

    kernel32 = ctypes.windll.kernel32
    # ERROR_ALREADY_EXISTS == 183
    return int(kernel32.GetLastError()) == 183
