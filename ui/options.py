from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QVBoxLayout,
    QDialogButtonBox,
    QFileDialog,
    QWidget,
)

from utils.config import CrawlerConfig, NLPConfig, VisualizationConfig
from ui.models import UserSettings


class OptionsDialog(QDialog):
    def __init__(self, parent: QWidget, settings: UserSettings):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self._settings = settings

        default_viz = VisualizationConfig()
        default_crawler = CrawlerConfig()
        default_nlp = NLPConfig()

        form = QFormLayout()

        self.base_dir_edit = QLineEdit(settings.base_dir)
        self.base_dir_edit.setPlaceholderText("default: scans")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_base_dir)
        base_row = QWidget()
        base_row_l = QHBoxLayout(base_row)
        base_row_l.setContentsMargins(0, 0, 0, 0)
        base_row_l.addWidget(self.base_dir_edit, 1)
        base_row_l.addWidget(browse_btn)
        form.addRow("Output folder", base_row)

        self.preferred_sources_edit = QLineEdit(
            ", ".join(settings.preferred_sources or [])
        )
        self.preferred_sources_edit.setPlaceholderText(
            f"default: {', '.join(default_crawler.sources or ['wikipedia'])} (comma-separated)"
        )
        form.addRow("Preferred sources", self.preferred_sources_edit)

        self.blacklisted_sources_edit = QLineEdit(
            ", ".join(settings.blacklisted_sources or [])
        )
        self.blacklisted_sources_edit.setPlaceholderText("default: none (comma-separated)")
        form.addRow("Blacklisted sources", self.blacklisted_sources_edit)

        self.viz_max_nodes_edit = QLineEdit(
            "" if settings.viz_max_nodes is None else str(settings.viz_max_nodes)
        )
        self.viz_max_nodes_edit.setPlaceholderText(
            f"(empty = default: {default_viz.max_nodes}) range: 1 - 10000"
        )
        form.addRow("Max nodes", self.viz_max_nodes_edit)

        self.viz_min_edge_confidence_edit = QLineEdit(
            "" if settings.viz_min_edge_confidence is None else str(settings.viz_min_edge_confidence)
        )
        self.viz_min_edge_confidence_edit.setPlaceholderText(
            "(empty = default: no filter) range: 0.0 - 1.0"
        )
        form.addRow("Min edge confidence", self.viz_min_edge_confidence_edit)

        self.viz_remove_isolated_edit = QCheckBox("Remove isolated nodes")
        if settings.viz_remove_isolated_nodes is None:
            self.viz_remove_isolated_edit.setChecked(True)
        else:
            self.viz_remove_isolated_edit.setChecked(bool(settings.viz_remove_isolated_nodes))
        form.addRow("", self.viz_remove_isolated_edit)

        self.enable_phase2_edit = QCheckBox("Enable Phase 2")
        if settings.enable_phase2 is None:
            self.enable_phase2_edit.setChecked(True)
        else:
            self.enable_phase2_edit.setChecked(bool(settings.enable_phase2))
        form.addRow("", self.enable_phase2_edit)

        self.phase2_max_pages_edit = QLineEdit(
            "" if settings.phase2_max_pages is None else str(settings.phase2_max_pages)
        )
        self.phase2_max_pages_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.phase2_max_pages}) range: 0 - 5000"
        )
        form.addRow("Phase 2 max pages", self.phase2_max_pages_edit)

        self.phase2_concurrent_tabs_edit = QLineEdit(
            "" if settings.phase2_concurrent_tabs is None else str(settings.phase2_concurrent_tabs)
        )
        self.phase2_concurrent_tabs_edit.setPlaceholderText(
            "(empty = default: same as Phase 1) range: 1 - 50"
        )
        form.addRow("Phase 2 concurrent tabs", self.phase2_concurrent_tabs_edit)

        self.document_min_relevance_edit = QLineEdit(
            "" if settings.document_min_relevance is None else str(settings.document_min_relevance)
        )
        self.document_min_relevance_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.document_min_relevance}) range: 0.0 - 1.0"
        )
        form.addRow("Document min relevance", self.document_min_relevance_edit)

        self.downloads_prune_irrelevant_edit = QCheckBox("Prune irrelevant downloads")
        if settings.downloads_prune_irrelevant is None:
            self.downloads_prune_irrelevant_edit.setChecked(True)
        else:
            self.downloads_prune_irrelevant_edit.setChecked(bool(settings.downloads_prune_irrelevant))
        form.addRow("", self.downloads_prune_irrelevant_edit)

        self.downloads_prune_mode_edit = QComboBox()
        self.downloads_prune_mode_edit.addItem("move")
        self.downloads_prune_mode_edit.addItem("delete")
        mode = (settings.downloads_prune_mode or "move").strip().lower()
        self.downloads_prune_mode_edit.setCurrentIndex(1 if mode == "delete" else 0)
        form.addRow("Prune mode", self.downloads_prune_mode_edit)

        self.web_search_max_pdf_downloads_edit = QLineEdit(
            "" if settings.web_search_max_pdf_downloads is None else str(settings.web_search_max_pdf_downloads)
        )
        self.web_search_max_pdf_downloads_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.web_search_max_pdf_downloads}) range: 0 - 1000"
        )
        form.addRow("Max PDF downloads", self.web_search_max_pdf_downloads_edit)

        self.web_search_min_relevance_edit = QLineEdit(
            "" if settings.web_search_min_relevance is None else str(settings.web_search_min_relevance)
        )
        self.web_search_min_relevance_edit.setPlaceholderText(
            f"(empty = default: {default_crawler.web_search_min_relevance}) range: 0.0 - 1.0"
        )
        form.addRow("Web search min relevance", self.web_search_min_relevance_edit)

        # Moved from main UI: default toggles for crawler behavior
        self.headless_edit = QCheckBox("Headless browser")
        self.headless_edit.setChecked(True if settings.headless is None else bool(settings.headless))
        form.addRow("", self.headless_edit)

        self.enable_web_search_edit = QCheckBox("Enable web search")
        self.enable_web_search_edit.setChecked(True if settings.enable_web_search is None else bool(settings.enable_web_search))
        form.addRow("", self.enable_web_search_edit)

        self.download_pdfs_edit = QCheckBox("Download PDFs from web search")
        self.download_pdfs_edit.setChecked(True if settings.download_pdfs is None else bool(settings.download_pdfs))
        form.addRow("", self.download_pdfs_edit)

        self.nlp_min_confidence_edit = QLineEdit(
            "" if settings.nlp_min_confidence is None else str(settings.nlp_min_confidence)
        )
        self.nlp_min_confidence_edit.setPlaceholderText(
            f"(empty = default: {default_nlp.min_confidence}) range: 0.0 - 1.0"
        )
        form.addRow("NLP min confidence", self.nlp_min_confidence_edit)

        self.nlp_min_relation_confidence_edit = QLineEdit(
            "" if settings.nlp_min_relation_confidence is None else str(settings.nlp_min_relation_confidence)
        )
        self.nlp_min_relation_confidence_edit.setPlaceholderText(
            f"(empty = default: {default_nlp.min_relation_confidence}) range: 0.0 - 1.0"
        )
        form.addRow("Min relation confidence", self.nlp_min_relation_confidence_edit)

        # Translator region selection (affects `translators` package region)
        self.translator_region_edit = QComboBox()
        for code in ("EN", "RO", "CN", "US", "DE", "FR", "RU"):
            self.translator_region_edit.addItem(code)
        try:
            current_region = (settings.translator_region or "EN").upper()
        except Exception:
            current_region = "EN"
        idx = self.translator_region_edit.findText(current_region)
        if idx >= 0:
            self.translator_region_edit.setCurrentIndex(idx)
        form.addRow("Translator region", self.translator_region_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def _browse_base_dir(self) -> None:
        start = self.base_dir_edit.text().strip() or str(Path.cwd())
        path = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if path:
            self.base_dir_edit.setText(path)

    def result_settings(self) -> UserSettings:
        base_dir = self.base_dir_edit.text().strip() or "scans"

        def _split_csv(text: str) -> list[str]:
            parts = [p.strip() for p in (text or "").split(",")]
            return [p for p in parts if p]

        preferred = _split_csv(self.preferred_sources_edit.text())
        preferred = preferred if preferred else None

        blacklisted = _split_csv(self.blacklisted_sources_edit.text())
        blacklisted = blacklisted if blacklisted else None

        max_nodes_raw = (self.viz_max_nodes_edit.text() or "").strip()
        max_nodes: Optional[int]
        if not max_nodes_raw:
            max_nodes = None
        else:
            max_nodes = int(max_nodes_raw)

        min_edge_raw = (self.viz_min_edge_confidence_edit.text() or "").strip()
        min_edge: Optional[float]
        if not min_edge_raw:
            min_edge = None
        else:
            min_edge = float(min_edge_raw)

        remove_isolated = bool(self.viz_remove_isolated_edit.isChecked())

        enable_phase2 = bool(self.enable_phase2_edit.isChecked())

        phase2_max_pages_raw = (self.phase2_max_pages_edit.text() or "").strip()
        phase2_max_pages: Optional[int]
        phase2_max_pages = None if not phase2_max_pages_raw else int(phase2_max_pages_raw)

        phase2_tabs_raw = (self.phase2_concurrent_tabs_edit.text() or "").strip()
        phase2_concurrent_tabs: Optional[int]
        phase2_concurrent_tabs = None if not phase2_tabs_raw else int(phase2_tabs_raw)

        doc_min_rel_raw = (self.document_min_relevance_edit.text() or "").strip()
        document_min_relevance: Optional[float]
        document_min_relevance = None if not doc_min_rel_raw else float(doc_min_rel_raw)

        downloads_prune_irrelevant = bool(self.downloads_prune_irrelevant_edit.isChecked())
        downloads_prune_mode = str(self.downloads_prune_mode_edit.currentText() or "move").strip().lower()

        max_pdf_raw = (self.web_search_max_pdf_downloads_edit.text() or "").strip()
        web_search_max_pdf_downloads: Optional[int]
        web_search_max_pdf_downloads = None if not max_pdf_raw else int(max_pdf_raw)

        web_min_rel_raw = (self.web_search_min_relevance_edit.text() or "").strip()
        web_search_min_relevance: Optional[float]
        web_search_min_relevance = None if not web_min_rel_raw else float(web_min_rel_raw)

        nlp_min_conf_raw = (self.nlp_min_confidence_edit.text() or "").strip()
        nlp_min_confidence: Optional[float]
        nlp_min_confidence = None if not nlp_min_conf_raw else float(nlp_min_conf_raw)

        nlp_min_rel_conf_raw = (self.nlp_min_relation_confidence_edit.text() or "").strip()
        nlp_min_relation_confidence: Optional[float]
        nlp_min_relation_confidence = None if not nlp_min_rel_conf_raw else float(nlp_min_rel_conf_raw)
        headless = bool(self.headless_edit.isChecked())
        enable_web_search = bool(self.enable_web_search_edit.isChecked())
        download_pdfs = bool(self.download_pdfs_edit.isChecked())

        return UserSettings(
            base_dir=base_dir,
            preferred_sources=preferred,
            blacklisted_sources=blacklisted,
            viz_max_nodes=max_nodes,
            viz_min_edge_confidence=min_edge,
            viz_remove_isolated_nodes=remove_isolated,
            enable_phase2=enable_phase2,
            phase2_max_pages=phase2_max_pages,
            phase2_concurrent_tabs=phase2_concurrent_tabs,
            document_min_relevance=document_min_relevance,
            downloads_prune_irrelevant=downloads_prune_irrelevant,
            downloads_prune_mode=downloads_prune_mode,
            web_search_max_pdf_downloads=web_search_max_pdf_downloads,
            web_search_min_relevance=web_search_min_relevance,
            nlp_min_confidence=nlp_min_confidence,
            nlp_min_relation_confidence=nlp_min_relation_confidence,
            headless=headless,
            enable_web_search=enable_web_search,
            download_pdfs=download_pdfs,
            translator_region=(self.translator_region_edit.currentText() or "EN").strip().upper(),
        )
