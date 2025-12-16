"""GitHub Releases-based update checker that handles .exe installers and
archives containing a PyInstaller "onedir" build.

Behavior:
- Prefer a .exe installer asset (Inno Setup) and run it silently.
- If only an archive (.zip/.tar.gz) is available, download and extract it,
  then spawn a PowerShell script that uses robocopy to mirror files into
  the installation directory after the app exits.

This module is Windows-focused (PowerShell + robocopy)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import shutil
import zipfile
from typing import Any, Optional, Callable

import requests
from packaging import version as pkg_version

from version import CURRENT_VERSION, REPO_NAME, REPO_OWNER


def check_for_updates(timeout_s: float = 5.0) -> Optional[dict[str, Any]]:
    """Return release JSON if a newer version exists; otherwise None."""

    try:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        headers = {"User-Agent": f"{REPO_NAME}"}
        response = requests.get(url, headers=headers, timeout=timeout_s)
        if response.status_code != 200:
            print(f"Update check failed: HTTP {response.status_code}")
            return None

        data = response.json()
        latest_tag = str(data.get("tag_name") or "").strip()
        if not latest_tag:
            print("Update check failed: no tag_name in response")
            return None

        latest_clean = latest_tag.lstrip("v").strip()
        if pkg_version.parse(latest_clean) > pkg_version.parse(CURRENT_VERSION):
            print(f"Update available: {CURRENT_VERSION} -> {latest_clean}")
            return data
    except Exception:
        return None

    return None


def _pick_asset_url(release_data: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return (url, kind) where kind is 'exe' or 'zip'. Prefer .exe, then .zip."""
    assets = release_data.get("assets") or []
    if not isinstance(assets, list):
        return None

    exe_url = None
    archive_url = None
    exact_archive_url = None
    archive_exts = (".zip", ".tar.gz", ".tgz", ".tar")

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").lower()
        url = asset.get("browser_download_url")
        if not url:
            continue
        if name.endswith(".exe"):
            exe_url = url
            break
        if any(name.endswith(ext) for ext in archive_exts):
            archive_url = archive_url or url
            # Prefer an exact-named archive SearchANDGraph.zip (case-insensitive)
            if name == f"{REPO_NAME.lower()}.zip":
                exact_archive_url = url

    if exe_url:
        return exe_url, "exe"
    if exact_archive_url:
        return exact_archive_url, "zip"
    if archive_url:
        return archive_url, "zip"
    return None


def perform_update(
    release_data: dict[str, Any], *, silent: bool = True, progress_callback: Optional[Callable[[int, str], None]] = None, exit_after_launch: bool = False
) -> dict[str, Any]:
    """Download asset, handle .exe installer or archive containing onedir files.

    For archives: extract to temp and launch a PowerShell script that copies files
    into the install dir (uses robocopy) after the app exits.

    When provided, `progress_callback` is called as `progress_callback(percent, message)`.
    The function returns a dict describing the action taken instead of exiting the process.
    """

    picked = _pick_asset_url(release_data)
    if not picked:
        raise RuntimeError("No suitable installer or archive asset found in the latest release.")

    download_url, kind = picked
    temp_dir = tempfile.gettempdir()

    headers = {"User-Agent": f"{REPO_NAME}"}

    # Prepare Windows-specific startup flags to avoid showing a console window
    win_startupinfo = None
    win_creationflags = 0
    if os.name == "nt":
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # 0 == SW_HIDE
            si.wShowWindow = 0
            win_startupinfo = si
            win_creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            win_startupinfo = None
            win_creationflags = 0

    if kind == "exe":
        installer_path = os.path.join(temp_dir, f"{REPO_NAME}_Update.exe")
        r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
        r.raise_for_status()

        total = int(r.headers.get("content-length") or 0)
        written = 0
        if progress_callback:
            progress_callback(0, "Downloading installer...")
        with open(installer_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total and progress_callback:
                        pct = int(written * 100 / total)
                        progress_callback(min(pct, 99), "Downloading installer...")

        args = [installer_path]
        if silent:
            args += ["/SILENT", "/NORESTART"]

        subprocess.Popen(args, close_fds=True, startupinfo=win_startupinfo, creationflags=win_creationflags)
        if progress_callback:
            progress_callback(100, "Installer launched")
        if exit_after_launch:
            # Ensure the whole process exits (not just the worker thread)
            try:
                import threading

                if threading.current_thread() is threading.main_thread():
                    raise SystemExit(0)
            except SystemExit:
                raise
            except Exception:
                pass
            os._exit(0)
        return {"action": "installer", "path": installer_path}

    # kind == "zip" (archive)
    archive_path = os.path.join(temp_dir, f"{REPO_NAME}_Update.zip")
    r = requests.get(download_url, headers=headers, stream=True, timeout=30.0)
    r.raise_for_status()

    total = int(r.headers.get("content-length") or 0)
    written = 0
    if progress_callback:
        progress_callback(0, "Downloading update archive...")
    with open(archive_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                written += len(chunk)
                if total and progress_callback:
                    pct = int(written * 100 / total)
                    progress_callback(min(pct, 80), "Downloading update archive...")

    extract_dir = os.path.join(temp_dir, f"{REPO_NAME}_Update_Extract")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    # Try to unpack archive
    try:
        shutil.unpack_archive(archive_path, extract_dir)
    except Exception:
        # Fallback to zipfile
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)

    # Determine install directory
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        # In development, assume current working dir is project root; prefer installation dir if known
        app_dir = os.path.abspath(os.getcwd())

    if progress_callback:
        progress_callback(85, "Preparing update script...")

    # Create PowerShell script to robocopy extracted files into app_dir
    # Prepare PowerShell apply script and make updater wait for the app to exit
    current_pid = os.getpid()
    ps_script = os.path.join(temp_dir, f"{REPO_NAME}_apply_update.ps1")
    src = extract_dir.replace('"', '""')
    dest = app_dir.replace('"', '""')
    exe_name = os.path.basename(sys.executable) if getattr(sys, "frozen", False) else "SearchANDGraph.exe"
    log_path = os.path.join(temp_dir, f"{REPO_NAME}_update_log.txt").replace('"', '""')
    status_path = os.path.join(temp_dir, f"{REPO_NAME}_update_status.json").replace('"', '""')

    # Use a robust rename-first PowerShell flow to avoid overwriting locked files
    ps_contents = f'''$appDir = "{dest}"
$oldDir = "$appDir`_old"
$tempSrc = "{src}"

# 1. Force kill any zombie browser/drivers that commonly hold locks
Get-Process chrome, msedge, chromedriver, msedgedriver -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 2. Wait for main EXE to release (try for ~5 seconds)
$waitCount = 0
while ((Test-Path "$appDir\\{exe_name}") -and $waitCount -lt 10) {{
    try {{ [IO.File]::OpenWrite("$appDir\\{exe_name}").Close(); break }}
    catch {{ Start-Sleep -Milliseconds 500; $waitCount++ }}
}}

# 3. Move the current installation to a temporary 'old' name (unplugs running app)
if (Test-Path $oldDir) {{ Remove-Item $oldDir -Recurse -Force -ErrorAction SilentlyContinue }}
Rename-Item -Path $appDir -NewName (Split-Path $oldDir -Leaf)

# 4. Copy the NEW files to the original path (preserve user data folder 'scans')
robocopy "$tempSrc" "$appDir" /MIR /R:2 /W:2 /XD "scans"

# 5. Relaunch and cleanup
Start-Process "$appDir\\{exe_name}"
Remove-Item $oldDir -Recurse -Force -ErrorAction SilentlyContinue
'''

    with open(ps_script, "w", encoding="utf-8") as f:
        f.write(ps_contents)

    if progress_callback:
        progress_callback(95, "Applying update...")

    # Detect whether the destination is writable; if not, we'll request elevation
    need_elevation = False
    try:
        test_path = os.path.join(app_dir, f".{REPO_NAME}_write_test")
        with open(test_path, "w") as tf:
            tf.write("test")
        os.remove(test_path)
    except Exception:
        need_elevation = True

    # Launch PowerShell script. If elevation is required, use Start-Process -Verb RunAs
    if os.name == "nt":
        if need_elevation:
            if progress_callback:
                progress_callback(96, "Requesting elevation to apply update...")
            # Use PowerShell to start an elevated PowerShell which runs the script
            elevate_cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                (
                    "Start-Process -Verb RunAs -FilePath 'powershell' "
                    f"-ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-WindowStyle','Hidden','-File','{ps_script}')"
                ),
            ]
            # Don't hide this bootstrapper too aggressively; UAC prompt must appear reliably.
            subprocess.Popen(elevate_cmd, close_fds=True)
        else:
            ps_args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", ps_script]
            subprocess.Popen(ps_args, close_fds=True, startupinfo=win_startupinfo, creationflags=win_creationflags)
    else:
        # Non-Windows fallback: run the script with the default shell (best-effort)
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script])
        except Exception:
            pass

    # Optionally spawn a small monitor script (Python) that can watch the status file
    monitor_script = os.path.join(temp_dir, f"{REPO_NAME}_update_monitor.py")
    monitor_contents = f"""
import time, json, sys, os
from pathlib import Path
status = Path(sys.argv[1])
log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
poll = 0.5
while True:
    if status.exists():
        try:
            data = json.loads(status.read_text(encoding='utf-8'))
            exitcode = int(data.get('exit', -1))
        except Exception:
            exitcode = -1
        # If failure, try to open the log with the default app
        if exitcode != 0 and log and log.exists():
            try:
                if os.name == 'nt':
                    os.startfile(str(log))
                else:
                    import webbrowser
                    webbrowser.open(str(log))
            except Exception:
                pass
        break
    time.sleep(poll)
"""

    try:
        with open(monitor_script, 'w', encoding='utf-8') as mf:
            mf.write(monitor_contents)
        # Launch detached monitor so it can run after this process exits
        try:
            if os.name == 'nt':
                DETACHED = 0x00000008
                subprocess.Popen([sys.executable, monitor_script, status_path, log_path], close_fds=True, startupinfo=win_startupinfo, creationflags=win_creationflags | DETACHED)
            else:
                subprocess.Popen([sys.executable, monitor_script, status_path, log_path], close_fds=True)
        except Exception:
            pass
    except Exception:
        pass

    if progress_callback:
        progress_callback(100, "Update applied (background)")
    result = {"action": "script", "script": ps_script, "elevated": need_elevation, "log": log_path, "status": status_path}
    if exit_after_launch:
        # Ensure the whole process exits (not just the worker thread)
        try:
            import threading

            if threading.current_thread() is threading.main_thread():
                raise SystemExit(0)
        except SystemExit:
            raise
        except Exception:
            pass
        os._exit(0)
    return result
