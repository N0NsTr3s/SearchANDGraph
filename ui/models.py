from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ScanRequest:
    query: str
    max_pages: int
    headless: bool
    enable_web_search: bool
    download_pdfs: bool
    start_url: Optional[str] = None
    base_dir: str = "scans"
    preferred_sources: tuple[str, ...] | None = None
    blacklisted_sources: tuple[str, ...] | None = None
    viz_max_nodes: Optional[int] = None
    viz_min_edge_confidence: Optional[float] = None
    viz_remove_isolated_nodes: Optional[bool] = None

    enable_phase2: Optional[bool] = None
    phase2_max_pages: Optional[int] = None
    phase2_concurrent_tabs: Optional[int] = None

    document_min_relevance: Optional[float] = None
    downloads_prune_irrelevant: Optional[bool] = None
    downloads_prune_mode: Optional[str] = None
    web_search_max_pdf_downloads: Optional[int] = None
    web_search_min_relevance: Optional[float] = None

    nlp_min_confidence: Optional[float] = None
    nlp_min_relation_confidence: Optional[float] = None
    # Preview settings
    preview_enabled: bool = False
    preview_interval_seconds: int = 2


@dataclass
class UserSettings:
    base_dir: str = "scans"
    preferred_sources: list[str] | None = None  # type: ignore[assignment]
    blacklisted_sources: list[str] | None = None  # type: ignore[assignment]
    source_priority: dict[str, int] | None = None
    recent_files: list[str] = None  # type: ignore[assignment]
    viz_max_nodes: Optional[int] = None
    viz_min_edge_confidence: Optional[float] = None
    viz_remove_isolated_nodes: Optional[bool] = None

    # New persisted defaults moved to Options dialog
    headless: bool = True
    enable_web_search: bool = True
    download_pdfs: bool = True
    # Timestamp (epoch seconds) when logs were last cleared by the auto-clean task
    last_logs_cleared: Optional[int] = None

    enable_phase2: Optional[bool] = None
    phase2_max_pages: Optional[int] = None
    phase2_concurrent_tabs: Optional[int] = None

    document_min_relevance: Optional[float] = None
    downloads_prune_irrelevant: Optional[bool] = None
    downloads_prune_mode: Optional[str] = None
    web_search_max_pdf_downloads: Optional[int] = None
    web_search_min_relevance: Optional[float] = None

    nlp_min_confidence: Optional[float] = None
    nlp_min_relation_confidence: Optional[float] = None

    preview_enabled: bool = False
    preview_interval_seconds: int = 2

    def __post_init__(self) -> None:
        if self.recent_files is None:
            self.recent_files = []

    @staticmethod
    def from_dict(data: dict) -> "UserSettings":
        return UserSettings(
            base_dir=str(data.get("base_dir") or "scans"),
            preferred_sources=(None if data.get("preferred_sources") is None else [str(s) for s in data.get("preferred_sources")]),
            blacklisted_sources=(None if data.get("blacklisted_sources") is None else [str(s) for s in data.get("blacklisted_sources")]),
            recent_files=[str(s) for s in (data.get("recent_files") or [])],
            viz_max_nodes=data.get("viz_max_nodes"),
            viz_min_edge_confidence=data.get("viz_min_edge_confidence"),
            viz_remove_isolated_nodes=data.get("viz_remove_isolated_nodes"),
            enable_phase2=data.get("enable_phase2"),
            phase2_max_pages=data.get("phase2_max_pages"),
            phase2_concurrent_tabs=data.get("phase2_concurrent_tabs"),
            document_min_relevance=data.get("document_min_relevance"),
            downloads_prune_irrelevant=data.get("downloads_prune_irrelevant"),
            downloads_prune_mode=data.get("downloads_prune_mode"),
            web_search_max_pdf_downloads=data.get("web_search_max_pdf_downloads"),
            web_search_min_relevance=data.get("web_search_min_relevance"),
            nlp_min_confidence=data.get("nlp_min_confidence"),
            nlp_min_relation_confidence=data.get("nlp_min_relation_confidence"),
            preview_enabled=bool(data.get("preview_enabled", False)),
            preview_interval_seconds=int(data.get("preview_interval_seconds", 2) or 2),
            # source_priority may be a mapping or None
            source_priority=(None if data.get("source_priority") is None else {str(k): int(v) for k, v in (data.get("source_priority") or {}).items()}),
            headless=bool(data.get("headless", True)),
            enable_web_search=bool(data.get("enable_web_search", True)),
            download_pdfs=bool(data.get("download_pdfs", True)),
            last_logs_cleared=(None if data.get("last_logs_cleared") is None else int(data.get("last_logs_cleared"))),
        )

    def to_dict(self) -> dict:
        return {
            "base_dir": self.base_dir,
            "preferred_sources": None if self.preferred_sources is None else list(self.preferred_sources),
            "blacklisted_sources": None if self.blacklisted_sources is None else list(self.blacklisted_sources),
            "recent_files": list(self.recent_files or []),
            "viz_max_nodes": self.viz_max_nodes,
            "viz_min_edge_confidence": self.viz_min_edge_confidence,
            "viz_remove_isolated_nodes": self.viz_remove_isolated_nodes,
            "enable_phase2": self.enable_phase2,
            "phase2_max_pages": self.phase2_max_pages,
            "phase2_concurrent_tabs": self.phase2_concurrent_tabs,
            "document_min_relevance": self.document_min_relevance,
            "downloads_prune_irrelevant": self.downloads_prune_irrelevant,
            "downloads_prune_mode": self.downloads_prune_mode,
            "web_search_max_pdf_downloads": self.web_search_max_pdf_downloads,
            "web_search_min_relevance": self.web_search_min_relevance,
            "nlp_min_confidence": self.nlp_min_confidence,
            "nlp_min_relation_confidence": self.nlp_min_relation_confidence,
            "preview_enabled": bool(self.preview_enabled),
            "preview_interval_seconds": int(self.preview_interval_seconds or 2),
            "source_priority": None if self.source_priority is None else dict(self.source_priority),
            "headless": bool(self.headless),
            "enable_web_search": bool(self.enable_web_search),
            "download_pdfs": bool(self.download_pdfs),
            "last_logs_cleared": None if self.last_logs_cleared is None else int(self.last_logs_cleared),
        }
