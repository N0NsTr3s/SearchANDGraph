"""FastAPI server exposing the crawler/graph pipeline.

Goals (minimal UX):
- Public webpage with a query box.
- If a graph for a query already exists, display it instead of re-running.
- If it doesn't exist, start a background run and show the graph when ready.

This server serves generated scan artifacts under /scans/.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from scan_manager import get_scan_paths, sanitize_query_for_path

app = FastAPI(title="SearchANDGraph API")

WORKSPACE_ROOT = Path(__file__).resolve().parent
SCANS_DIR = WORKSPACE_ROOT / "scans"
SCANS_DIR.mkdir(parents=True, exist_ok=True)

# Serve scan artifacts publicly
app.mount("/scans", StaticFiles(directory=str(SCANS_DIR)), name="scans")

# In-memory job tracking (best-effort). Status is also written to disk.
_tasks: dict[str, asyncio.Task] = {}
_locks: dict[str, asyncio.Lock] = {}

# Basic public-safety controls (in-memory; resets on restart).
# For true public deployments, move these to Redis (or similar) and add auth.
MAX_REQUESTS_PER_MINUTE_PER_IP = 120
MAX_RUN_REQUESTS_PER_MINUTE_PER_IP = 10
MAX_CONCURRENT_RUNS = 1

_recent_requests: dict[str, list[float]] = {}
_recent_run_requests: dict[str, list[float]] = {}
_run_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)


def _lock_for(query_key: str) -> asyncio.Lock:
    lock = _locks.get(query_key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[query_key] = lock
    return lock


def _client_ip(request: Request) -> str:
    # If you later deploy behind a reverse proxy, configure trusted headers and
    # use X-Forwarded-For safely. For now, keep it simple.
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _prune_and_check(bucket: dict[str, list[float]], key: str, limit: int, window_s: float) -> bool:
    """Return True if allowed, False if rate-limited."""
    now = asyncio.get_running_loop().time()
    times = bucket.get(key)
    if times is None:
        times = []
        bucket[key] = times

    cutoff = now - window_s
    # prune
    i = 0
    for t in times:
        if t >= cutoff:
            break
        i += 1
    if i:
        del times[:i]

    if len(times) >= limit:
        return False

    times.append(now)
    return True


def _enforce_rate_limits(request: Request, *, is_run: bool = False) -> None:
    ip = _client_ip(request)
    if not _prune_and_check(_recent_requests, ip, MAX_REQUESTS_PER_MINUTE_PER_IP, 60.0):
        raise HTTPException(status_code=429, detail="Too many requests")
    if is_run and not _prune_and_check(_recent_run_requests, ip, MAX_RUN_REQUESTS_PER_MINUTE_PER_IP, 60.0):
        raise HTTPException(status_code=429, detail="Too many run requests")


def _status_path(scan_dir: Path) -> Path:
    return scan_dir / "status.json"


def _write_status(scan_dir: Path, status: str, detail: Optional[str] = None) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "status": status,
    }
    if detail:
        payload["detail"] = detail
    _status_path(scan_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_status(scan_dir: Path) -> Dict[str, Any]:
    path = _status_path(scan_dir)
    if not path.exists():
        return {"status": "missing"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "unknown"}


def _graph_url_for_query(query: str) -> Optional[str]:
    """Return the public URL to an existing graph HTML for the query."""
    scan_paths = get_scan_paths(query, base_dir=str(SCANS_DIR), add_timestamp=False)

    # Prefer interactive, then static
    interactive = scan_paths["interactive_viz_file"]
    static = scan_paths["viz_file"]

    if interactive.exists():
        rel = interactive.relative_to(SCANS_DIR).as_posix()
        return f"/scans/{rel}"
    if static.exists():
        rel = static.relative_to(SCANS_DIR).as_posix()
        return f"/scans/{rel}"

    return None


async def _run_scan(query: str) -> None:
    """Run a scan and write status updates to disk."""
    scan_paths = get_scan_paths(query, base_dir=str(SCANS_DIR), add_timestamp=False)
    scan_dir: Path = scan_paths["scan_dir"]

    _write_status(scan_dir, "queued")

    async with _run_semaphore:
        _write_status(scan_dir, "running")

    try:
        # Import here so the server can start even if heavy deps load slowly.
        from main_refactored import main as run_main

        await run_main(query=query, add_timestamp=False, base_dir=str(SCANS_DIR))

        # Mark done if graph exists
        if _graph_url_for_query(query):
            _write_status(scan_dir, "ready")
        else:
            _write_status(scan_dir, "error", "Scan finished but graph HTML not found")

    except Exception as e:
        _write_status(scan_dir, "error", str(e))
        raise


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Minimal public page to query + display graphs."""
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>SearchANDGraph</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .row { display:flex; gap: 8px; align-items:center; }
    input { flex: 1; padding: 10px; font-size: 16px; }
    button { padding: 10px 14px; font-size: 16px; cursor: pointer; }
    #status { margin-top: 12px; white-space: pre-wrap; }
    iframe { width: 100%; height: 78vh; border: 1px solid #ddd; margin-top: 14px; }
  </style>
</head>
<body>
  <h2>SearchANDGraph</h2>
  <div class=\"row\">
    <input id=\"q\" placeholder=\"Enter a query (e.g., Nvidia)\" />
    <button id=\"go\">Search</button>
  </div>
  <div id=\"status\"></div>
  <iframe id=\"frame\" src=\"about:blank\"></iframe>

<script>
const q = document.getElementById('q');
const statusEl = document.getElementById('status');
const frame = document.getElementById('frame');
const btn = document.getElementById('go');

function setStatus(msg) { statusEl.textContent = msg; }

async function getExisting(query) {
  const r = await fetch(`/api/graphs?query=${encodeURIComponent(query)}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function startRun(query) {
  const r = await fetch(`/api/run?query=${encodeURIComponent(query)}`, { method: 'POST' });
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function getStatus(query) {
  const r = await fetch(`/api/status?query=${encodeURIComponent(query)}`);
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function pollUntilReady(query) {
  for (;;) {
    const s = await getStatus(query);
    if (s.status === 'ready' && s.url) return s;
    if (s.status === 'error') throw new Error(s.detail || 'error');
    setStatus(`Status: ${s.status}...`);
    await new Promise(res => setTimeout(res, 2500));
  }
}

btn.addEventListener('click', async () => {
  const query = q.value.trim();
  if (!query) return;

  frame.src = 'about:blank';
  setStatus('Checking for existing graph...');

  try {
    const existing = await getExisting(query);
    if (existing.status === 'ready' && existing.url) {
      setStatus('Found existing graph.');
      frame.src = existing.url;
      return;
    }

    setStatus('No existing graph. Starting a new run...');
    await startRun(query);

    const done = await pollUntilReady(query);
    setStatus('Graph ready.');
    frame.src = done.url;

  } catch (e) {
    setStatus(`Error: ${e.message || e}`);
  }
});
</script>
</body>
</html>"""


@app.get("/api/graphs")
async def get_graph(request: Request, query: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Return existing graph URL if available (no rerun)."""
    _enforce_rate_limits(request)
    url = _graph_url_for_query(query)
    if url:
        return {
            "status": "ready",
            "url": url,
            "query_key": sanitize_query_for_path(query),
            "match": "exact_normalized",
        }

    scan_paths = get_scan_paths(query, base_dir=str(SCANS_DIR), add_timestamp=False)
    scan_dir: Path = scan_paths["scan_dir"]
    status = _read_status(scan_dir)

    # If a job is running, surface that; otherwise missing.
    if status.get("status") in {"running", "queued"}:
        return {
            "status": status.get("status"),
            "query_key": sanitize_query_for_path(query),
            "match": "exact_normalized",
        }

    return {"status": "missing", "query_key": sanitize_query_for_path(query), "match": "exact_normalized"}


@app.get("/api/status")
async def get_run_status(request: Request, query: str = Query(..., min_length=1)) -> Dict[str, Any]:
    _enforce_rate_limits(request)
    scan_paths = get_scan_paths(query, base_dir=str(SCANS_DIR), add_timestamp=False)
    scan_dir: Path = scan_paths["scan_dir"]

    url = _graph_url_for_query(query)
    if url:
        return {
            "status": "ready",
            "url": url,
            "query_key": sanitize_query_for_path(query),
            "match": "exact_normalized",
        }

    status = _read_status(scan_dir)
    status["query_key"] = sanitize_query_for_path(query)
    status["match"] = "exact_normalized"
    return status


@app.post("/api/run")
async def run_query(request: Request, query: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Start a background run unless a graph already exists."""
    _enforce_rate_limits(request, is_run=True)
    # If already exists, don't rerun
    existing = _graph_url_for_query(query)
    if existing:
        return {
            "status": "ready",
            "url": existing,
            "query_key": sanitize_query_for_path(query),
            "match": "exact_normalized",
        }

    query_key = sanitize_query_for_path(query)
    scan_paths = get_scan_paths(query, base_dir=str(SCANS_DIR), add_timestamp=False)
    scan_dir: Path = scan_paths["scan_dir"]

    lock = _lock_for(query_key)
    async with lock:
        # Re-check under lock
        existing = _graph_url_for_query(query)
        if existing:
            return {"status": "ready", "url": existing, "query_key": query_key}

        task = _tasks.get(query_key)
        if task and not task.done():
            _write_status(scan_dir, "running")
            return {"status": "running", "query_key": query_key, "match": "exact_normalized"}

        _write_status(scan_dir, "queued")

        async def runner() -> None:
            await _run_scan(query)

        _tasks[query_key] = asyncio.create_task(runner())
        return {"status": "queued", "query_key": query_key, "match": "exact_normalized"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
