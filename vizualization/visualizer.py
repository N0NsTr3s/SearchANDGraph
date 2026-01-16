"""
Graph visualization module using Bokeh.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import json
from pathlib import Path
from urllib.parse import urlparse

from bokeh.layouts import column, row
from bokeh.io import save
try:
    from bokeh.events import DocumentReady, Tap
except Exception:  # pragma: no cover
    DocumentReady = None  # type: ignore[assignment]
    Tap = None  # type: ignore[assignment]
from bokeh.models import (
    Button, CheckboxGroup, ColumnDataSource, CustomJS, Div, 
    HoverTool, Range1d, RangeSlider, Select, Slider, TapTool, TextInput
)
from bokeh.plotting import figure, from_networkx, output_file
try:
    from ..utils.config import VisualizationConfig
    from ..utils.logger import setup_logger
except:
    from utils.config import VisualizationConfig
    from utils.logger import setup_logger
    
logger = setup_logger(__name__)
class GraphVisualizer:
    """Handles graph visualization using Bokeh."""

    NODE_COLORS = {
        'PERSON': '#FF6B6B',
        'ORG': '#4ECDC4',
        'GPE': '#45B7D1',
        'LOC': '#96CEB4',
        'DATE': '#FFEAA7',
        'EVENT': '#A29BFE',
        'LAW': '#F8C291',
        'PRODUCT': '#81ECEC',
        'DEFAULT': '#74B9FF'
    }

    def __init__(self, config: VisualizationConfig):
        """
        Initialize the graph visualizer.
        
        Args:
            config: Visualization configuration settings
        """
        self.config = config
    
    def visualize(self, graph: nx.Graph):
        """
        Create an interactive visualization of the knowledge graph.
        
        Args:
            graph: NetworkX Graph to visualize
        """
        if not graph.nodes():
            logger.warning("Graph is empty. Nothing to visualize.")
            return

        prepared_graph = self._prepare_graph(graph)

        if not prepared_graph.nodes():
            logger.warning("Graph is empty after applying visualization filters. Nothing to visualize.")
            return

        logger.info(
            "Creating visualization with %s nodes and %s edges (rendering %s nodes / %s edges after filtering)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            prepared_graph.number_of_nodes(),
            prepared_graph.number_of_edges()
        )

        if self.config.auto_range:
            plot = figure(
                title=self.config.plot_title,
                tools="pan,wheel_zoom,box_zoom,save,reset",
                active_scroll="wheel_zoom",
                width=1000,
                height=800,
                tooltips=None
            )
        else:
            plot = figure(
                title=self.config.plot_title,
                x_range=Range1d(*self.config.x_range),
                y_range=Range1d(*self.config.y_range),
                tools="pan,wheel_zoom,box_zoom,save,reset",
                active_scroll="wheel_zoom",
                width=1000,
                height=800,
                tooltips=None
            )

        # Apply subtle styling tweaks for better UX
        plot.toolbar.autohide = True
        plot.toolbar_location = "above"
        plot.background_fill_color = "#fdfdfd"
        plot.border_fill_color = "#ffffff"
        plot.outline_line_color = None
        plot.min_border_left = 20
        plot.min_border_right = 20
        plot.min_border_top = 10
        plot.min_border_bottom = 10

        positions = self._compute_layout(prepared_graph)

        graph_renderer = from_networkx(prepared_graph, positions)  # type: ignore[arg-type]
        plot.renderers.append(graph_renderer)

        self._style_nodes(graph_renderer, prepared_graph)
        self._style_edges(graph_renderer, prepared_graph)

        if self.config.auto_range:
            self._auto_adjust_ranges(plot, positions)

        # Capture current plot ranges if available (Range1d exposes start/end); guard with hasattr to satisfy static checkers
        if hasattr(plot.x_range, "start") and hasattr(plot.x_range, "end") and hasattr(plot.y_range, "start") and hasattr(plot.y_range, "end"):
            initial_ranges = {
                'x_start': getattr(plot.x_range, 'start'),
                'x_end': getattr(plot.x_range, 'end'),
                'y_start': getattr(plot.y_range, 'start'),
                'y_end': getattr(plot.y_range, 'end')
            }
        else:
            initial_ranges = {}

        node_details = Div(
            text=(
                "<div style='padding: 10px 12px; color: #555; font-size: 0.9em; background-color: #ffffff; "
                "border: 1px solid #dcdfe6; border-radius: 6px; max-height: 160px; overflow-y: auto;'>"
                "Click a node to see details here."
                "</div>"
            ),
            width=350,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0)
        )

        # Add node interaction (hover + click)
        self._add_node_interaction(plot, prepared_graph, graph_renderer, node_details)
        
        # Add edge interaction (hover and click with pinning)
        pinned_panel, documents_index = self._add_edge_interaction(plot, prepared_graph, graph_renderer)

        # Also include locally downloaded PDFs (even if they aren't linked to a specific edge yet).
        try:
            scan_dir = Path(self.config.output_file).resolve().parent
            downloads_dir = scan_dir / "downloads"
            if downloads_dir.exists():
                for pdf_path in downloads_dir.rglob("*.pdf"):
                    try:
                        file_url = pdf_path.resolve().as_uri()
                    except Exception:
                        continue
                    if file_url not in documents_index:
                        documents_index[file_url] = {
                            'connections': [],
                            'nodes': [],
                        }
        except Exception:
            pass

        # Build search experience (no intro tip)
        search_input = TextInput(
            title="Search nodes",
            placeholder="e.g. Maria Popescu, acquisition, Q359129",
            width=350
        )

        search_results_default = (
            "<div style='padding: 10px 12px; color: #555; font-size: 0.9em; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px; max-height: 120px; overflow-y: auto;'>"
            "Start typing to highlight matching nodes."
            "</div>"
        )

        search_results = Div(
            text=search_results_default,
            width=350,
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0)
        )

        clear_button = Button(label="Clear search", button_type="default", width=130)

        # === NEW: Dynamic Filtering Controls ===
        # Get unique entity types from graph
        entity_types = sorted(set(
            prepared_graph.nodes[node].get('label', 'UNKNOWN') 
            for node in prepared_graph.nodes()
        ))
        
        filters_intro = Div(
            text="""
            <div style='padding: 10px 12px; background-color: #fff3cd; border-radius: 6px; border-left: 4px solid #ffc107; margin-bottom: 10px;'>
                <p style='margin: 0; font-size: 0.85em; color: #856404;'>
                    <strong>🎛️ Filters:</strong> Dynamically filter the graph to focus on what matters.
                </p>
            </div>
            """,
            width=350
        )

        # Entity type filter checkboxes
        entity_type_labels = [f"{etype} ({sum(1 for n in prepared_graph.nodes() if prepared_graph.nodes[n].get('label') == etype)})" for etype in entity_types]
        entity_type_filter = CheckboxGroup(
            labels=entity_type_labels,
            active=list(range(len(entity_types))),  # All checked by default
            width=350
        )
        
        entity_filter_label = Div(
            text="<strong style='font-size: 0.9em;'>Entity Types:</strong>",
            width=350
        )

        # Edge confidence slider
        max_confidence = max(
            (data.get('confidence', 0.0) for _, _, data in prepared_graph.edges(data=True)),
            default=1.0
        )
        min_confidence = min(
            (data.get('confidence', 0.0) for _, _, data in prepared_graph.edges(data=True) if data.get('confidence') is not None),
            default=0.0
        )
        
        confidence_slider = Slider(
            start=0.0,
            end=1.0,
            value=min_confidence,
            step=0.05,
            title="Minimum Edge Confidence",
            width=350
        )

        # Node degree (connections) slider
        degree_list = [prepared_graph.degree(node) for node in prepared_graph.nodes()]  # type: ignore[misc]
        max_degree = max(degree_list) if degree_list else 1
        degree_slider = Slider(
            start=0,
            end=max_degree,
            value=0,
            step=1,
            title="Minimum Node Connections",
            width=350
        )

        # Reset filters button
        reset_filters_button = Button(
            label="Reset All Filters",
            button_type="warning",
            width=350
        )

        # === NEW: Display Options ===
        display_intro = Div(
            text="""
            <div style='padding: 10px 12px; background-color: #d1ecf1; border-radius: 6px; border-left: 4px solid #17a2b8; margin: 15px 0 10px 0;'>
                <p style='margin: 0; font-size: 0.85em; color: #0c5460;'>
                    <strong>👁️ Display:</strong> Customize how the graph appears.
                </p>
            </div>
            """,
            width=350
        )

        # Toggle labels checkbox
        show_labels_checkbox = CheckboxGroup(
            labels=["Show node labels", "Show edge labels"],
            active=[0],  # Node labels shown by default
            width=350
        )

        # Color scheme selector
        color_scheme_select = Select(
            title="Color nodes by:",
            value="type",
            options=["type", "degree", "community"],
            width=350
        )

        # Size scheme selector
        size_scheme_select = Select(
            title="Size nodes by:",
            value="degree",
            options=["degree", "uniform"],
            width=350
        )

        # Documents dropdown (PDFs linked from connections + locally downloaded PDFs)
        def _doc_label(doc_key: str) -> str:
            try:
                if doc_key.startswith('file:'):
                    parsed = urlparse(doc_key)
                    name = Path(parsed.path).name
                    return name or doc_key
                parsed = urlparse(doc_key)
                name = Path(parsed.path).name
                return name or doc_key
            except Exception:
                return doc_key

        doc_option_pairs = sorted(
            [(k, _doc_label(k)) for k in documents_index.keys()],
            key=lambda pair: (pair[1].lower(), pair[0].lower())
        )

        documents_select = Select(
            title="Documents",
            value=doc_option_pairs[0][0] if doc_option_pairs else "",
            options=doc_option_pairs,
            width=350
        )
        documents_details = Div(
            text=(
                "<div style='padding: 10px 12px; color: #555; font-size: 0.9em; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px;'>"
                "No documents detected yet. If PDFs were downloaded, re-run the scan/graph generation so they get indexed here."
                "</div>"
            ) if not doc_option_pairs else "",
            width=350,
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0)
        )

        doc_index_json = json.dumps(documents_index)
        documents_select.js_on_change(
            "value",
            CustomJS(
                args=dict(details_div=documents_details, doc_index_json=doc_index_json),
                code="""
                    function ensureGlobalDelegation() {
                        // Attach once: handles <a> clicks inside Bokeh shadow DOM and pinned remove buttons.
                        if (document.__kgGlobalDelegationAttached) return;
                        document.__kgGlobalDelegationAttached = true;

                        document.addEventListener('click', function (e) {
                            const path = (e && e.composedPath) ? e.composedPath() : [];

                            // 1) Pinned remove buttons (inside shadow DOM)
                            let removeBtn = null;
                            for (const n of path) {
                                if (n && n.getAttribute && n.getAttribute('data-action') === 'remove') {
                                    removeBtn = n;
                                    break;
                                }
                            }
                            if (removeBtn) {
                                const targetId = removeBtn.getAttribute('data-target');
                                if (targetId) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    try {
                                        if (window.__kgPinnedRemoveById && typeof window.__kgPinnedRemoveById === 'function') {
                                            // Prefer stateful removal so deleted pins don't come back on re-render.
                                            window.__kgPinnedRemoveById(targetId);
                                        } else {
                                            const root = (removeBtn.getRootNode && removeBtn.getRootNode()) ? removeBtn.getRootNode() : document;
                                            let el = null;
                                            if (root && root.querySelector) {
                                                el = root.querySelector('#' + targetId);
                                            }
                                            if (!el) {
                                                el = document.getElementById(targetId);
                                            }
                                            if (el) el.remove();
                                        }
                                    } catch (err) {
                                        // no-op
                                    }
                                }
                                return;
                            }

                            // 2) External links inside Bokeh widgets (shadow DOM retargeting)
                            let anchor = null;
                            for (const n of path) {
                                if (n && n.tagName && String(n.tagName).toLowerCase() === 'a' && n.href) {
                                    anchor = n;
                                    break;
                                }
                            }
                            if (!anchor) return;
                            const href = String(anchor.href || '');
                            if (!href) return;
                            // Only intercept real external/file links.
                            if (!(href.startsWith('http://') || href.startsWith('https://') || href.startsWith('file://'))) return;
                            // Try to open in a new window/tab (Qt WebEngine createWindow will forward externally).
                            try {
                                const w = window.open(href, anchor.target || '_blank');
                                if (w) {
                                    e.preventDefault();
                                    e.stopPropagation();
                                }
                            } catch (err) {
                                // If window.open fails, fall back to default browser behavior.
                            }
                        }, true);
                    }

                    const v = cb_obj.value;
                    const index = JSON.parse(doc_index_json || '{}');
                    const item = index[v];

                    if (!v || !item) {
                        details_div.text = "<div style='padding: 12px; color: #555; font-size: 0.9em; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px;'>No document selected.</div>";
                        return;
                    }

                    const connections = (item.connections || []).slice(0, 25);
                    const nodes = (item.nodes || []).slice(0, 25);

                    let html = "<div style='padding:12px; background-color:#ffffff; border:1px solid #dcdfe6; border-radius:6px;'>";
                    html += `<div style='margin-bottom:8px;'><a href='${v}' target='_blank' style='color:#0066cc; text-decoration:none;'>📄 Open document</a></div>`;
                    if (connections.length) {
                        html += "<div style='font-size:0.9em; color:#12325b; margin-bottom:6px;'><strong>Linked connections</strong></div>";
                        html += "<ul style='margin:0 0 10px 18px; padding:0; font-size:0.9em; color:#333;'>";
                        for (const c of connections) {
                            html += `<li>${c}</li>`;
                        }
                        html += "</ul>";
                    }
                    if (nodes.length) {
                        html += "<div style='font-size:0.9em; color:#12325b; margin-bottom:6px;'><strong>Related nodes</strong></div>";
                        html += "<ul style='margin:0 0 0 18px; padding:0; font-size:0.9em; color:#333;'>";
                        for (const n of nodes) {
                            html += `<li>${n}</li>`;
                        }
                        html += "</ul>";
                    }
                    html += "</div>";
                    details_div.text = html;

                    // Make sure external link clicks work inside embedded shadow DOM.
                    ensureGlobalDelegation();
                """
            )
        )

        # Initialize documents panel content for the first item.
        if doc_option_pairs:
            first = doc_option_pairs[0][0]
            first_item = documents_index.get(first, {})
            connections = (first_item.get('connections') or [])[:25]
            nodes = (first_item.get('nodes') or [])[:25]
            html = "<div style='padding:12px; background-color:#ffffff; border:1px solid #dcdfe6; border-radius:6px;'>"
            html += f"<div style='margin-bottom:8px;'><a href='{first}' target='_blank' style='color:#0066cc; text-decoration:none;'>📄 Open document</a></div>"
            if connections:
                html += "<div style='font-size:0.9em; color:#12325b; margin-bottom:6px;'><strong>Linked connections</strong></div>"
                html += "<ul style='margin:0 0 10px 18px; padding:0; font-size:0.9em; color:#333;'>" + "".join(f"<li>{c}</li>" for c in connections) + "</ul>"
            if nodes:
                html += "<div style='font-size:0.9em; color:#12325b; margin-bottom:6px;'><strong>Related nodes</strong></div>"
                html += "<ul style='margin:0 0 0 18px; padding:0; font-size:0.9em; color:#333;'>" + "".join(f"<li>{n}</li>" for n in nodes) + "</ul>"
            html += "</div>"
            documents_details.text = html

        documents_intro = Div(
            text=(
                "<div style='padding: 10px 12px; background-color: #e8f4f8; border-radius: 6px; border-left: 4px solid #0066cc; margin: 15px 0 10px 0;'>"
                "<p style='margin: 0; font-size: 0.85em; color: #0c5460;'>"
                "<strong>📄 Documents:</strong> PDFs linked as evidence for connections."
                "</p></div>"
            ),
            width=350
        )

        # Right panel: Search, Node details, Documents, and Filters
        right_panel = column(
            search_input,
            clear_button,
            search_results,
            node_details,
            documents_intro,
            documents_select,
            documents_details,
            filters_intro,
            entity_filter_label,
            entity_type_filter,
            confidence_slider,
            degree_slider,
            reset_filters_button,
            display_intro,
            show_labels_checkbox,
            color_scheme_select,
            size_scheme_select,
            sizing_mode="stretch_height",
            width=370
        )

        # Keep the right panel non-scrollable; show controls in full.

        # Left panel: Pinned connections only
        left_panel = column(
            pinned_panel,
            sizing_mode="stretch_height",
            width=370
        )

        search_callback = CustomJS(
            args=dict(
                search_input=search_input,
                node_source=graph_renderer.node_renderer.data_source,
                edge_source=graph_renderer.edge_renderer.data_source,
                results_div=search_results,
                graph_renderer=graph_renderer,
                x_range=plot.x_range,
                y_range=plot.y_range,
                initial_ranges=initial_ranges,
                default_message=search_results_default
            ),
            code="""
                const query = (search_input.value || '').trim().toLowerCase();
                const nodeData = node_source.data;
                const edgeData = edge_source.data;

                const baseNodeColors = nodeData['color_base'] || [];
                const baseNodeSizes = nodeData['size_base'] || [];
                const baseNodeAlphas = nodeData['alpha_base'] || [];
                const nodeColors = nodeData['color'] || [];
                const nodeSizes = nodeData['size'] || [];
                const nodeAlphas = nodeData['alpha'] || [];
                const nodeIds = nodeData['node_id'] || nodeData['index'] || [];
                const displayNames = nodeData['display_name'] || [];
                const nodeQids = nodeData['node_qid'] || [];
                const nodeLabels = nodeData['label'] || [];
                const degrees = nodeData['degree'] || [];

                const baseEdgeColors = edgeData['edge_color_base'] || [];
                const baseEdgeAlphas = edgeData['edge_alpha_base'] || [];
                const baseEdgeWidths = edgeData['edge_width_base'] || [];
                const edgeColors = edgeData['edge_color'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const edgeWidths = edgeData['edge_width'] || [];
                const startIds = edgeData['start_id'] || edgeData['start'] || [];
                const endIds = edgeData['end_id'] || edgeData['end'] || [];

                const layout = graph_renderer.layout_provider.graph_layout;

                function resetView() {
                    for (let i = 0; i < nodeColors.length; i++) {
                        nodeColors[i] = baseNodeColors[i] || nodeColors[i];
                        nodeSizes[i] = baseNodeSizes[i] || nodeSizes[i];
                        nodeAlphas[i] = baseNodeAlphas[i] || 0.95;
                    }
                    for (let j = 0; j < edgeColors.length; j++) {
                        edgeColors[j] = baseEdgeColors[j] || '#8A9DBF';
                        edgeAlphas[j] = baseEdgeAlphas[j] || 0.45;
                        edgeWidths[j] = baseEdgeWidths[j] || edgeWidths[j];
                    }
                    if (initial_ranges && initial_ranges.x_start !== undefined) {
                        x_range.start = initial_ranges.x_start;
                        x_range.end = initial_ranges.x_end;
                        y_range.start = initial_ranges.y_start;
                        y_range.end = initial_ranges.y_end;
                    }
                    results_div.text = default_message;
                    node_source.change.emit();
                    edge_source.change.emit();
                }

                if (!query) {
                    resetView();
                    return;
                }

                const matchedNodeIndices = [];
                const matchedNodeSummaries = [];
                const matchedNodeIds = new Set();
                let highlightedEdgeCount = 0;

                for (let i = 0; i < nodeIds.length; i++) {
                    const displayValue = (displayNames[i] || '').toLowerCase();
                    const nodeIdValue = (nodeIds[i] || '').toLowerCase();
                    const qidValue = (nodeQids[i] || '').toLowerCase();

                    const matches = (
                        (displayValue && displayValue.includes(query)) ||
                        (qidValue && qidValue.includes(query)) ||
                        (nodeIdValue && nodeIdValue.includes(query))
                    );

                    if (matches) {
                        matchedNodeIndices.push(i);
                        matchedNodeIds.add(nodeIds[i]);
                        matchedNodeSummaries.push(`<li><strong>${displayNames[i]}</strong> <span style='color:#577399;'>(${nodeLabels[i] || 'N/A'})</span> — degree ${degrees[i] || 0}</li>`);
                        nodeColors[i] = '#f1c40f';
                        nodeSizes[i] = (baseNodeSizes[i] || nodeSizes[i]) * 1.25;
                        nodeAlphas[i] = 1.0;
                    } else {
                        nodeColors[i] = baseNodeColors[i] || nodeColors[i];
                        nodeSizes[i] = (baseNodeSizes[i] || nodeSizes[i]) * 0.82;
                        nodeAlphas[i] = 0.18;
                    }
                }

                for (let j = 0; j < edgeColors.length; j++) {
                    const startId = startIds[j];
                    const endId = endIds[j];
                    const touchesMatched = matchedNodeIds.has(startId) || matchedNodeIds.has(endId);
                    if (touchesMatched && matchedNodeIds.size > 0) {
                        highlightedEdgeCount += 1;
                        edgeColors[j] = baseEdgeColors[j] || edgeColors[j] || '#8A9DBF';
                        edgeAlphas[j] = Math.max(baseEdgeAlphas[j] || 0.45, 0.6);
                        edgeWidths[j] = baseEdgeWidths[j] || edgeWidths[j];
                    } else {
                        edgeColors[j] = baseEdgeColors[j] || '#8A9DBF';
                        edgeAlphas[j] = Math.min(baseEdgeAlphas[j] || 0.45, 0.12);
                        edgeWidths[j] = (baseEdgeWidths[j] || edgeWidths[j]) * 0.85;
                    }
                }

                if (matchedNodeIndices.length === 0) {
                    results_div.text = `<div style='padding:12px; color:#a94442; background-color:#f8d7da; border:1px solid #f5c6cb; border-radius:6px;'>No nodes matched "${search_input.value}".</div>`;
                    node_source.change.emit();
                    edge_source.change.emit();
                    return;
                }

                const xs = [];
                const ys = [];
                matchedNodeIds.forEach(id => {
                    const coords = layout[id];
                    if (coords) {
                        xs.push(coords[0]);
                        ys.push(coords[1]);
                    }
                });

                if (xs.length > 0) {
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    const padX = (maxX - minX || 1) * 0.4;
                    const padY = (maxY - minY || 1) * 0.4;
                    x_range.start = minX - padX;
                    x_range.end = maxX + padX;
                    y_range.start = minY - padY;
                    y_range.end = maxY + padY;
                }

                const nodeList = matchedNodeSummaries.slice(0, 8).join('');
                const nodeCount = matchedNodeIndices.length;

                let summaryHtml = `<div style='padding:10px 12px; background-color:#ffffff; border:1px solid #dcdfe6; border-radius:6px; max-height:140px; overflow-y:auto;'>`;
                summaryHtml += `<div style='color:#1b4b8f; font-size:0.95em; margin-bottom:8px;'>Found <strong>${nodeCount}</strong> matching node${nodeCount === 1 ? '' : 's'}.</div>`;
                if (highlightedEdgeCount > 0) {
                    summaryHtml += `<div style='color:#577399; font-size:0.85em; margin-bottom:10px;'>Highlighted ${highlightedEdgeCount} connecting edge${highlightedEdgeCount === 1 ? '' : 's'} for context.</div>`;
                }
                if (nodeList) {
                    summaryHtml += `<div style='margin-bottom:12px;'><h4 style='margin:0 0 6px 0; color:#12325b; font-size:0.95em;'>Nodes</h4><ul style='margin:0 0 0 18px; font-size:0.9em; color:#333;'>${nodeList}${matchedNodeSummaries.length > 8 ? '<li>…</li>' : ''}</ul></div>`;
                }
                summaryHtml += `</div>`;

                results_div.text = summaryHtml;

                node_source.change.emit();
                edge_source.change.emit();
            """
        )

        search_input.js_on_change("value", search_callback)
        clear_button.js_on_event("button_click", CustomJS(args=dict(search_input=search_input), code="search_input.value = '';"))

        # === Node Click Callback - Show node details in the right panel ===
        node_tap_callback = CustomJS(
            args=dict(
                node_source=graph_renderer.node_renderer.data_source,
                details_div=node_details
            ),
            code="""
                const selected = node_source.selected.indices;
                if (selected.length === 0) {
                    return;
                }
                
                const idx = selected[0];
                const nodeData = node_source.data;
                const nodeId = nodeData['node_id'][idx];
                const displayName = nodeData['display_name'][idx];
                const qid = nodeData['node_qid'][idx] || '';
                const label = nodeData['label'][idx] || 'UNKNOWN';
                const degree = nodeData['degree'][idx] || 0;
                
                let html = `
                    <div style='padding: 10px 12px; background-color: #ffffff; border-radius: 6px; border: 1px solid #dcdfe6; max-height: 160px; overflow-y: auto;'>
                        <div style='margin: 0 0 8px 0; font-size: 0.95em; color: #0066cc; font-weight: 700;'>🔍 Node: ${displayName}</div>
                        <div style='margin-bottom: 8px; padding: 8px; background-color: #e8f4f8; border-radius: 4px;'>
                            <p style='margin: 0 0 6px 0; font-size: 0.85em; color: #666;'>
                                <strong>Type:</strong> <span style='padding: 2px 8px; background-color: #0066cc; color: white; border-radius: 3px; font-size: 0.8em;'>${label}</span>
                            </p>
                `;
                
                if (qid) {
                    html += `
                        <p style='margin: 0 0 8px 0; font-size: 0.85em;'>
                            <strong>Wikidata:</strong> <a href='https://www.wikidata.org/wiki/${qid}' target='_blank' style='color: #0066cc;'>${qid}</a>
                        </p>
                    `;
                }
                
                html += `
                        <p style='margin: 0; font-size: 0.85em; color: #666;'>
                            <strong>Connections:</strong> ${degree}
                        </p>
                    </div>
                    <div style='font-size: 0.85em; color: #666;'>
                        <p style='margin: 0;'><em>Click an edge to see relationship details.</em></p>
                    </div>
                </div>`;
                
                details_div.text = html;
            """
        )
        
        # Add TapTool for node selection
        node_tap_tool = TapTool(renderers=[graph_renderer.node_renderer], callback=node_tap_callback)
        plot.add_tools(node_tap_tool)

        # === NEW: Entity Type Filter Callback ===
        entity_filter_callback = CustomJS(
            args=dict(
                checkbox=entity_type_filter,
                node_source=graph_renderer.node_renderer.data_source,
                edge_source=graph_renderer.edge_renderer.data_source,
                entity_types=entity_types
            ),
            code="""
                const active = checkbox.active;
                const activeTypes = active.map(i => entity_types[i]);
                const nodeData = node_source.data;
                const edgeData = edge_source.data;
                
                const labels = nodeData['label'] || [];
                const nodeAlphas = nodeData['alpha'] || [];
                const baseAlphas = nodeData['alpha_base'] || [];
                const nodeIds = nodeData['node_id'] || [];
                
                // Create a set of visible node IDs
                const visibleNodeIds = new Set();
                
                for (let i = 0; i < labels.length; i++) {
                    if (activeTypes.includes(labels[i])) {
                        nodeAlphas[i] = baseAlphas[i] || 0.95;
                        visibleNodeIds.add(nodeIds[i]);
                    } else {
                        nodeAlphas[i] = 0.0;  // Hide node
                    }
                }
                
                // Hide edges where either endpoint is hidden
                const startIds = edgeData['start_id'] || [];
                const endIds = edgeData['end_id'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const baseEdgeAlphas = edgeData['edge_alpha_base'] || [];
                
                for (let i = 0; i < startIds.length; i++) {
                    if (visibleNodeIds.has(startIds[i]) && visibleNodeIds.has(endIds[i])) {
                        edgeAlphas[i] = baseEdgeAlphas[i] || 0.45;
                    } else {
                        edgeAlphas[i] = 0.0;  // Hide edge
                    }
                }
                
                node_source.change.emit();
                edge_source.change.emit();
            """
        )
        
        entity_type_filter.js_on_change("active", entity_filter_callback)

        # === NEW: Confidence Filter Callback ===
        confidence_filter_callback = CustomJS(
            args=dict(
                slider=confidence_slider,
                edge_source=graph_renderer.edge_renderer.data_source
            ),
            code="""
                const minConfidence = slider.value;
                const edgeData = edge_source.data;
                const confidences = edgeData['confidence'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const baseAlphas = edgeData['edge_alpha_base'] || [];
                
                for (let i = 0; i < confidences.length; i++) {
                    const conf = confidences[i];
                    if (conf === null || conf === undefined || conf >= minConfidence) {
                        edgeAlphas[i] = baseAlphas[i] || 0.45;
                    } else {
                        edgeAlphas[i] = 0.0;  // Hide edge
                    }
                }
                
                edge_source.change.emit();
            """
        )
        
        confidence_slider.js_on_change("value", confidence_filter_callback)

        # === NEW: Degree Filter Callback ===
        degree_filter_callback = CustomJS(
            args=dict(
                slider=degree_slider,
                node_source=graph_renderer.node_renderer.data_source,
                edge_source=graph_renderer.edge_renderer.data_source
            ),
            code="""
                const minDegree = slider.value;
                const nodeData = node_source.data;
                const edgeData = edge_source.data;
                
                const degrees = nodeData['degree'] || [];
                const nodeAlphas = nodeData['alpha'] || [];
                const baseAlphas = nodeData['alpha_base'] || [];
                const nodeIds = nodeData['node_id'] || [];
                
                // Create a set of visible node IDs
                const visibleNodeIds = new Set();
                
                for (let i = 0; i < degrees.length; i++) {
                    if (degrees[i] >= minDegree) {
                        nodeAlphas[i] = baseAlphas[i] || 0.95;
                        visibleNodeIds.add(nodeIds[i]);
                    } else {
                        nodeAlphas[i] = 0.0;  // Hide node
                    }
                }
                
                // Hide edges where either endpoint is hidden
                const startIds = edgeData['start_id'] || [];
                const endIds = edgeData['end_id'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const baseEdgeAlphas = edgeData['edge_alpha_base'] || [];
                
                for (let i = 0; i < startIds.length; i++) {
                    if (visibleNodeIds.has(startIds[i]) && visibleNodeIds.has(endIds[i])) {
                        edgeAlphas[i] = baseEdgeAlphas[i] || 0.45;
                    } else {
                        edgeAlphas[i] = 0.0;  // Hide edge
                    }
                }
                
                node_source.change.emit();
                edge_source.change.emit();
            """
        )
        
        degree_slider.js_on_change("value", degree_filter_callback)

        # === NEW: Reset Filters Callback ===
        reset_filters_callback = CustomJS(
            args=dict(
                entity_checkbox=entity_type_filter,
                confidence_slider=confidence_slider,
                degree_slider=degree_slider,
                node_source=graph_renderer.node_renderer.data_source,
                edge_source=graph_renderer.edge_renderer.data_source,
                num_types=len(entity_types)
            ),
            code="""
                // Reset all filters to default
                entity_checkbox.active = Array.from({length: num_types}, (_, i) => i);
                confidence_slider.value = 0.0;
                degree_slider.value = 0;
                
                // Reset all visibilities
                const nodeData = node_source.data;
                const edgeData = edge_source.data;
                
                const nodeAlphas = nodeData['alpha'] || [];
                const baseAlphas = nodeData['alpha_base'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const baseEdgeAlphas = edgeData['edge_alpha_base'] || [];
                
                for (let i = 0; i < nodeAlphas.length; i++) {
                    nodeAlphas[i] = baseAlphas[i] || 0.95;
                }
                
                for (let i = 0; i < edgeAlphas.length; i++) {
                    edgeAlphas[i] = baseEdgeAlphas[i] || 0.45;
                }
                
                node_source.change.emit();
                edge_source.change.emit();
            """
        )
        
        reset_filters_button.js_on_event("button_click", reset_filters_callback)

        # === NEW: Label Toggle Callback ===
        label_toggle_callback = CustomJS(
            args=dict(
                checkbox=show_labels_checkbox,
                node_renderer=graph_renderer.node_renderer
            ),
            code="""
                const active = checkbox.active;
                const showNodeLabels = active.includes(0);
                
                // Toggle node label visibility
                // Note: Bokeh doesn't directly support text labels on graph nodes
                // This is a placeholder for future enhancement
                console.log("Node labels:", showNodeLabels ? "shown" : "hidden");
                console.log("Edge labels:", active.includes(1) ? "shown" : "hidden");
            """
        )
        
        show_labels_checkbox.js_on_change("active", label_toggle_callback)

        # === NEW: Color Scheme Selector Callback ===
        color_scheme_callback = CustomJS(
            args=dict(
                select=color_scheme_select,
                node_source=graph_renderer.node_renderer.data_source,
                node_colors=self.NODE_COLORS
            ),
            code="""
                const scheme = select.value;
                const nodeData = node_source.data;
                const labels = nodeData['label'] || [];
                const degrees = nodeData['degree'] || [];
                const colors = nodeData['color'] || [];
                const baseColors = nodeData['color_base'] || [];
                
                // Define color mappings
                const typeColors = {
                    'PERSON': '#FF6B6B',
                    'ORG': '#4ECDC4',
                    'GPE': '#45B7D1',
                    'LOC': '#96CEB4',
                    'DATE': '#FFEAA7',
                    'EVENT': '#A29BFE',
                    'LAW': '#F8C291',
                    'PRODUCT': '#81ECEC',
                    'DEFAULT': '#74B9FF'
                };
                
                if (scheme === 'type') {
                    // Color by entity type
                    for (let i = 0; i < labels.length; i++) {
                        colors[i] = typeColors[labels[i]] || typeColors['DEFAULT'];
                        baseColors[i] = colors[i];
                    }
                } else if (scheme === 'degree') {
                    // Color by degree (connection count)
                    const maxDeg = Math.max(...degrees, 1);
                    for (let i = 0; i < degrees.length; i++) {
                        const intensity = degrees[i] / maxDeg;
                        // Gradient from light blue to dark blue
                        const r = Math.floor(116 + (27 - 116) * intensity);
                        const g = Math.floor(185 + (75 - 185) * intensity);
                        const b = Math.floor(255 + (143 - 255) * intensity);
                        colors[i] = `rgb(${r},${g},${b})`;
                        baseColors[i] = colors[i];
                    }
                } else if (scheme === 'community') {
                    // Placeholder for community detection
                    // For now, just use a random color per node
                    const communityColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#A29BFE'];
                    for (let i = 0; i < labels.length; i++) {
                        colors[i] = communityColors[i % communityColors.length];
                        baseColors[i] = colors[i];
                    }
                }
                
                node_source.change.emit();
            """
        )
        
        color_scheme_select.js_on_change("value", color_scheme_callback)

        # === NEW: Size Scheme Selector Callback ===
        size_scheme_callback = CustomJS(
            args=dict(
                select=size_scheme_select,
                node_source=graph_renderer.node_renderer.data_source
            ),
            code="""
                const scheme = select.value;
                const nodeData = node_source.data;
                const degrees = nodeData['degree'] || [];
                const sizes = nodeData['size'] || [];
                const baseSizes = nodeData['size_base'] || [];
                
                if (scheme === 'degree') {
                    // Size by degree (already computed in _compute_node_sizes)
                    for (let i = 0; i < baseSizes.length; i++) {
                        sizes[i] = baseSizes[i];
                    }
                } else if (scheme === 'uniform') {
                    // All nodes same size
                    const uniformSize = 15;
                    for (let i = 0; i < sizes.length; i++) {
                        sizes[i] = uniformSize;
                    }
                }
                
                node_source.change.emit();
            """
        )
        
        size_scheme_select.js_on_change("value", size_scheme_callback)

        # Set output file (inline resources so HTML works offline / inside embedded viewers)
        output_file(self.config.output_file, mode="inline")
        logger.info(f"Visualization saved to: {self.config.output_file}")
        
        # Layout: Left panel (pinned), center (plot), right panel (search + filters)
        main_layout = row(left_panel, plot, right_panel, sizing_mode="stretch_height")

        # Ensure clicks inside Bokeh shadow DOM behave in embedded viewers (Qt WebEngine):
        # - Open file/http links reliably
        # - Allow pinned-item removal buttons to work even though they live in a shadow root
        if DocumentReady is not None:
            plot.js_on_event(
                DocumentReady,
                CustomJS(
                    code="""
                        (function installKgDelegation() {
                            if (document.__kgGlobalDelegationAttached) return;
                            document.__kgGlobalDelegationAttached = true;

                            document.addEventListener('click', function (e) {
                                const path = (e && e.composedPath) ? e.composedPath() : [];

                                // 1) Pinned remove buttons
                                for (const n of path) {
                                    if (n && n.getAttribute && n.getAttribute('data-action') === 'remove') {
                                        const targetId = n.getAttribute('data-target');
                                        if (targetId) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            try {
                                                if (window.__kgPinnedRemoveById && typeof window.__kgPinnedRemoveById === 'function') {
                                                    window.__kgPinnedRemoveById(targetId);
                                                } else {
                                                    const root = (n.getRootNode && n.getRootNode()) ? n.getRootNode() : document;
                                                    let el = null;
                                                    if (root && root.querySelector) {
                                                        el = root.querySelector('#' + targetId);
                                                    }
                                                    if (!el) {
                                                        el = document.getElementById(targetId);
                                                    }
                                                    if (el) el.remove();
                                                }
                                            } catch (err) {
                                                // no-op
                                            }
                                        }
                                        return;
                                    }
                                }

                                // 2) External/file links inside shadow DOM
                                for (const n of path) {
                                    if (n && n.tagName && String(n.tagName).toLowerCase() === 'a' && n.href) {
                                        const href = String(n.href || '');
                                        if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('file://')) {
                                            // Force a navigation request (Qt will intercept and open externally)
                                            e.preventDefault();
                                            e.stopPropagation();
                                            try {
                                                window.location.href = href;
                                            } catch (err) {
                                                // no-op
                                            }
                                            return;
                                        }
                                    }
                                }
                            }, true);
                        })();
                    """
                ),
            )
        # Save without launching an external browser (desktop UI loads the file itself)
        save(main_layout)

    def _prepare_graph(self, graph: nx.Graph) -> nx.Graph:
        """Apply readability filters before rendering the graph."""
        working_graph = graph.copy()

        removed_edges = 0
        if self.config.min_edge_confidence is not None:
            low_conf_edges = [
                (u, v)
                for u, v, data in working_graph.edges(data=True)
                if data.get('confidence', 0.0) < self.config.min_edge_confidence
            ]
            removed_edges = len(low_conf_edges)
            if low_conf_edges:
                working_graph.remove_edges_from(low_conf_edges)
                logger.info(
                    "Removed %s low-confidence edges (threshold=%.2f)",
                    removed_edges,
                    self.config.min_edge_confidence
                )

        if self.config.remove_isolated_nodes:
            isolates = list(nx.isolates(working_graph))
            if isolates:
                working_graph.remove_nodes_from(isolates)
                logger.info("Removed %s isolated nodes", len(isolates))

        max_nodes = self.config.max_nodes
        if max_nodes and working_graph.number_of_nodes() > max_nodes:
            # Keep the most connected nodes to maintain context
            degree_pairs = list(working_graph.degree)  # type: ignore[misc]
            degree_sorted = sorted(degree_pairs, key=lambda item: item[1], reverse=True)
            keep_nodes = {node for node, _ in degree_sorted[:max_nodes]}
            drop_nodes = [node for node in working_graph.nodes if node not in keep_nodes]
            working_graph.remove_nodes_from(drop_nodes)
            logger.info(
                "Clamped graph to top %s nodes by degree (removed %s nodes)",
                max_nodes,
                len(drop_nodes)
            )

            if self.config.remove_isolated_nodes:
                isolates = list(nx.isolates(working_graph))
                if isolates:
                    working_graph.remove_nodes_from(isolates)
                    logger.info("Removed %s newly isolated nodes after clamping", len(isolates))

        return working_graph

    def _compute_layout(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """Compute a spaced layout using configurable spring parameters."""
        if graph.number_of_nodes() == 1:
            node = next(iter(graph.nodes()))
            return {node: (self.config.center[0], self.config.center[1])}

        k = self.config.layout_force
        if k is None:
            k = 1.0 / math.sqrt(max(graph.number_of_nodes(), 1))

        layout_raw = nx.spring_layout(
            graph,
            k=k,
            iterations=self.config.layout_iterations,
            seed=self.config.layout_seed,
            scale=self.config.scale,
            center=self.config.center
        )

        spread = self.config.layout_spread or 1.0

        # Convert numpy arrays returned by networkx into tuples of floats to satisfy type hints
        layout: Dict[str, Tuple[float, float]] = {}
        for node, coords in layout_raw.items():
            x, y = coords[0], coords[1]
            if spread != 1.0:
                layout[node] = (float(x * spread), float(y * spread))
            else:
                layout[node] = (float(x), float(y))

        return layout

    def _style_nodes(self, graph_renderer, graph: nx.Graph) -> None:
        """Apply dynamic node sizing and coloring."""
        node_source = graph_renderer.node_renderer.data_source
        node_order = list(graph.nodes())
        labels = [graph.nodes[node].get('label', 'UNKNOWN') for node in node_order]
        
        # Get display names (use display_name if available, otherwise use node ID)
        display_names = [graph.nodes[node].get('display_name', node) for node in node_order]

        colors = [self.NODE_COLORS.get(label, self.NODE_COLORS['DEFAULT']) for label in labels]
        sizes = self._compute_node_sizes(graph, node_order)
        degree_map = {node: int(deg) for node, deg in graph.degree}  # type: ignore[misc]
        degrees = [degree_map.get(node, 0) for node in node_order]

        # Build search text for each node (includes labels, aliases, IDs, etc.)
        search_texts: List[str] = []
        node_qids: List[str] = []
        for node, label, name in zip(node_order, labels, display_names):
            node_data = graph.nodes[node]
            qid = node_data.get('qid') or node_data.get('wikidata_id') or ''
            node_qids.append(str(qid) if qid else '')

            primary_tokens = [str(name), str(node)]
            if qid:
                primary_tokens.append(str(qid))
            search_texts.append(' '.join(part.lower() for part in primary_tokens if part))

        # Alpha values for visibility control
        alphas = [0.95 for _ in node_order]

        node_source.data['label'] = labels
        node_source.data['color'] = list(colors)
        node_source.data['size'] = list(sizes)
        node_source.data['degree'] = degrees
        node_source.data['display_name'] = display_names
        node_source.data['node_id'] = node_order
        node_source.data['search_text'] = search_texts
        node_source.data['node_qid'] = node_qids
        node_source.data['color_base'] = list(colors)
        node_source.data['size_base'] = list(sizes)
        node_source.data['alpha'] = list(alphas)
        node_source.data['alpha_base'] = list(alphas)

        glyph = graph_renderer.node_renderer.glyph
        glyph.size = 'size'
        glyph.fill_color = 'color'
        glyph.line_color = '#1B4B8F'
        glyph.line_width = 1.6
        glyph.fill_alpha = 'alpha'
        glyph.line_alpha = 'alpha'

    def _style_edges(self, graph_renderer, graph: nx.Graph) -> None:
        """Apply edge width scaling based on confidence."""
        edge_source = graph_renderer.edge_renderer.data_source
        edges = list(graph.edges(data=True))

        widths, confidences = self._compute_edge_widths(edges)

        # Metadata for search highlighting
        start_ids: List[str] = []
        end_ids: List[str] = []
        start_display: List[str] = []
        end_display: List[str] = []
        search_texts: List[str] = []
        reason_previews: List[str] = []
        relation_types: List[str] = []  # Store relation types

        default_edge_color = '#8A9DBF'
        edge_colors = [default_edge_color for _ in edges]
        edge_alphas = [0.45 for _ in edges]

        for u, v, data in edges:
            start_ids.append(str(u))
            end_ids.append(str(v))
            start_name = graph.nodes[u].get('display_name', u)
            end_name = graph.nodes[v].get('display_name', v)
            start_display.append(str(start_name))
            end_display.append(str(end_name))

            reasons = data.get('reasons', []) or []
            if reasons:
                preview = reasons[0].split('|||')[0] if isinstance(reasons[0], str) else str(reasons[0])
            else:
                preview = ''
            reason_previews.append(preview)
            
            # Extract relation type
            relation_type = data.get('relation_type', '')
            relation_types.append(relation_type if relation_type else '')

            relation = data.get('relation') or data.get('type') or ''
            confidence = data.get('confidence')
            confidence_text = f"{confidence:.2f}" if isinstance(confidence, (float, int)) else ''
            reason_text = ' '.join(str(item) for item in reasons if item)
            search_parts = [str(u), str(v), str(start_name), str(end_name), relation, reason_text, confidence_text, relation_type]
            search_texts.append(' '.join(part.lower() for part in search_parts if part))

        edge_labels = [f"{s} → {e}" for s, e in zip(start_display, end_display)]

        edge_source.data['edge_width'] = list(widths)
        edge_source.data['edge_width_base'] = list(widths)
        edge_source.data['confidence'] = confidences
        edge_source.data['edge_color'] = list(edge_colors)
        edge_source.data['edge_color_base'] = list(edge_colors)
        edge_source.data['edge_alpha'] = list(edge_alphas)
        edge_source.data['edge_alpha_base'] = list(edge_alphas)
        edge_source.data['start_id'] = start_ids
        edge_source.data['end_id'] = end_ids
        edge_source.data['start_display'] = start_display
        edge_source.data['end_display'] = end_display
        edge_source.data['edge_label'] = edge_labels
        edge_source.data['reason_preview'] = reason_previews
        edge_source.data['search_text'] = search_texts
        edge_source.data['relation_type'] = relation_types  # Add relation types to data source

        glyph = graph_renderer.edge_renderer.glyph
        glyph.line_width = 'edge_width'
        glyph.line_alpha = 'edge_alpha'
        glyph.line_color = 'edge_color'

    def _auto_adjust_ranges(self, plot, positions: Dict[str, Tuple[float, float]]) -> None:
        """Automatically adjust the plot ranges to fit the layout."""
        if not positions:
            return

        xs = [coord[0] for coord in positions.values()]
        ys = [coord[1] for coord in positions.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span = max(max_x - min_x, max_y - min_y)
        padding = span * 0.2 if span else 1.0

        plot.x_range = Range1d(min_x - padding, max_x + padding)
        plot.y_range = Range1d(min_y - padding, max_y + padding)

    def _compute_node_sizes(self, graph: nx.Graph, node_order: List[str]) -> List[float]:
        """Scale node sizes between configured bounds using degree centrality."""
        min_size, max_size = self.config.node_size_range
        degrees = {node: int(deg) for node, deg in graph.degree}  # type: ignore[misc]

        if not degrees:
            return [min_size for _ in node_order]

        min_degree = min(degrees.values())
        max_degree = max(degrees.values())

        if min_degree == max_degree:
            uniform_size = (min_size + max_size) / 2
            return [uniform_size for _ in node_order]

        size_range = max_size - min_size
        return [
            min_size + (degrees[node] - min_degree) / (max_degree - min_degree) * size_range
            for node in node_order
        ]

    def _compute_edge_widths(self, edges: List[Tuple[str, str, Dict]]) -> Tuple[List[float], List[Optional[float]]]:
        """Convert edge confidences into visible widths."""
        min_width, max_width = self.config.edge_width_range
        width_range = max_width - min_width

        widths: List[float] = []
        confidences: List[Optional[float]] = []

        for _, _, data in edges:
            confidence = data.get('confidence')
            confidences.append(confidence)

            if confidence is None:
                widths.append(min_width)
            else:
                clamped = max(0.0, min(confidence, 1.0))
                widths.append(min_width + clamped * width_range)

        return widths, confidences
    
    def _add_node_interaction(self, plot, graph, graph_renderer, node_details_div):
        """
        Add hover tooltips and click interaction for nodes.
        
        Args:
            plot: Bokeh plot object
            graph: NetworkX graph
            graph_renderer: Bokeh graph renderer
            node_details_div: Div element to update with node details
        """
        # Add node labels
        graph_renderer.node_renderer.data_source.data['label'] = [
            graph.nodes[node].get('label', 'N/A') for node in graph.nodes()
        ]
        
        # Create hover tool for nodes
        node_hover = HoverTool(
            tooltips=[
                ("Entity", "@display_name"),
                ("Type", "@label"),
                ("Degree", "@degree")
            ],
            renderers=[graph_renderer.node_renderer]
        )
        plot.add_tools(node_hover)

        # Add click interaction
        node_source = graph_renderer.node_renderer.data_source
        edge_source = graph_renderer.edge_renderer.data_source
        
        # Callback for node selection
        callback = CustomJS(
            args=dict(
                node_source=node_source,
                edge_source=edge_source,
                details_div=node_details_div,
                graph_renderer=graph_renderer
            ),
            code="""
                const selected_indices = node_source.selected.indices;
                const nodeData = node_source.data;
                const edgeData = edge_source.data;
                
                const baseNodeColors = nodeData['color_base'] || [];
                const baseNodeSizes = nodeData['size_base'] || [];
                const baseNodeAlphas = nodeData['alpha_base'] || [];
                const nodeColors = nodeData['color'] || [];
                const nodeSizes = nodeData['size'] || [];
                const nodeAlphas = nodeData['alpha'] || [];
                const nodeIds = nodeData['node_id'] || [];
                const displayNames = nodeData['display_name'] || [];
                const nodeLabels = nodeData['label'] || [];
                const degrees = nodeData['degree'] || [];
                
                const baseEdgeColors = edgeData['edge_color_base'] || [];
                const baseEdgeAlphas = edgeData['edge_alpha_base'] || [];
                const edgeColors = edgeData['edge_color'] || [];
                const edgeAlphas = edgeData['edge_alpha'] || [];
                const startIds = edgeData['start_id'] || [];
                const endIds = edgeData['end_id'] || [];
                
                // Reset function
                function resetView() {
                    for (let i = 0; i < nodeColors.length; i++) {
                        nodeColors[i] = baseNodeColors[i] || nodeColors[i];
                        nodeSizes[i] = baseNodeSizes[i] || nodeSizes[i];
                        nodeAlphas[i] = baseNodeAlphas[i] || 0.95;
                    }
                    for (let j = 0; j < edgeColors.length; j++) {
                        edgeColors[j] = baseEdgeColors[j] || '#8A9DBF';
                        edgeAlphas[j] = baseEdgeAlphas[j] || 0.45;
                    }
                    details_div.text = "<div style='padding: 10px 12px; color: #555; font-size: 0.9em; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px; max-height: 160px; overflow-y: auto;'>Click a node to see details here.</div>";
                }

                if (selected_indices.length === 0) {
                    resetView();
                    node_source.change.emit();
                    edge_source.change.emit();
                    return;
                }
                
                const idx = selected_indices[0];
                const clickedNodeId = nodeIds[idx];
                const clickedDisplayName = displayNames[idx];
                
                // Find neighbors
                const neighborIds = new Set();
                const connectedEdgeIndices = new Set();
                
                for (let i = 0; i < startIds.length; i++) {
                    if (startIds[i] === clickedNodeId) {
                        neighborIds.add(endIds[i]);
                        connectedEdgeIndices.add(i);
                    } else if (endIds[i] === clickedNodeId) {
                        neighborIds.add(startIds[i]);
                        connectedEdgeIndices.add(i);
                    }
                }
                
                // Update Nodes
                for (let i = 0; i < nodeIds.length; i++) {
                    if (i === idx) {
                        // Clicked node
                        nodeColors[i] = '#f1c40f'; // Highlight color
                        nodeSizes[i] = (baseNodeSizes[i] || nodeSizes[i]) * 1.25;
                        nodeAlphas[i] = 1.0;
                    } else if (neighborIds.has(nodeIds[i])) {
                        // Neighbor node
                        nodeColors[i] = baseNodeColors[i]; // Keep original color
                        nodeSizes[i] = baseNodeSizes[i];
                        nodeAlphas[i] = 1.0;
                    } else {
                        // Unrelated node
                        nodeColors[i] = baseNodeColors[i];
                        nodeSizes[i] = baseNodeSizes[i];
                        nodeAlphas[i] = 0.1; // Dimmed
                    }
                }
                
                // Update Edges
                for (let i = 0; i < startIds.length; i++) {
                    if (connectedEdgeIndices.has(i)) {
                        edgeColors[i] = '#555555'; // Darker for visibility
                        edgeAlphas[i] = 1.0;
                    } else {
                        edgeColors[i] = baseEdgeColors[i];
                        edgeAlphas[i] = 0.05; // Very dim
                    }
                }
                
                // Update Details Div
                const label = nodeLabels[idx] || 'Unknown';
                const degree = degrees[idx] || 0;
                
                let html = `<div style='padding: 10px 12px; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px; max-height: 200px; overflow-y: auto;'>`;
                html += `<h4 style='margin: 0 0 8px 0; color: #2c3e50;'>${clickedDisplayName}</h4>`;
                html += `<div style='font-size: 0.9em; margin-bottom: 4px;'><strong>Type:</strong> <span style='background-color: #eee; padding: 2px 6px; border-radius: 4px;'>${label}</span></div>`;
                html += `<div style='font-size: 0.9em; margin-bottom: 8px;'><strong>Connections:</strong> ${degree}</div>`;
                
                if (neighborIds.size > 0) {
                    html += `<div style='font-size: 0.9em; color: #555; margin-bottom: 4px;'><strong>Directly connected to:</strong></div>`;
                    html += `<ul style='margin: 0 0 0 18px; padding: 0; font-size: 0.85em; color: #333;'>`;
                    
                    // Collect neighbor names for display
                    const neighbors = [];
                    for(let i=0; i<nodeIds.length; i++) {
                        if(neighborIds.has(nodeIds[i])) {
                            neighbors.push(displayNames[i]);
                        }
                    }
                    // Sort and limit
                    neighbors.sort();
                    const limit = 20;
                    for(let i=0; i<Math.min(neighbors.length, limit); i++) {
                        html += `<li>${neighbors[i]}</li>`;
                    }
                    if (neighbors.length > limit) {
                        html += `<li><em>...and ${neighbors.length - limit} more</em></li>`;
                    }
                    html += `</ul>`;
                } else {
                    html += `<div style='font-size: 0.9em; color: #777; font-style: italic;'>No direct connections visible.</div>`;
                }
                
                html += `</div>`;
                details_div.text = html;
                
                node_source.change.emit();
                edge_source.change.emit();
            """
        )
        
        node_source.selected.js_on_change('indices', callback)
        
        # Add TapTool for nodes
        node_tap = TapTool(renderers=[graph_renderer.node_renderer])
        plot.add_tools(node_tap)
    
    def _add_edge_interaction(self, plot, graph, graph_renderer):
        """
        Add interactive edge tooltips with click-to-pin functionality.
        Shows hover tooltip AND allows pinning multiple connections on the left.
        
        Args:
            plot: Bokeh plot object
            graph: NetworkX graph
            graph_renderer: Bokeh graph renderer
            
        Returns:
            Div element for displaying pinned connections
        """
        edge_start = []
        edge_end = []
        edge_reasons = []
        edge_reasons_compact = []  # For hover tooltips
        
        # Collect edge data and ensure no duplicate reasons
        documents_index: Dict[str, Dict[str, List[str]]] = {}

        def is_pdf_url(candidate: str) -> bool:
            try:
                parsed = urlparse(candidate)
                suffix = Path((parsed.path or '').lower()).suffix
                if suffix == '.pdf':
                    return True
                # Some PDFs are served via query parameters; be permissive.
                q = (parsed.query or '').lower()
                return 'pdf' in q and (candidate.startswith('http://') or candidate.startswith('https://'))
            except Exception:
                return False
        for u, v, data in graph.edges(data=True):
            edge_start.append(u)
            edge_end.append(v)
            
            # Get provenance data (contains text + source_url)
            provenance_list = data.get('provenance', [])
            dates = data.get('dates', [])
            confidence = data.get('confidence')
            relation_type = data.get('relation_type', '')  # Get relation type
            
            # Rebuild reasons with URLs from provenance data
            # Format: "text|||url" for each provenance entry
            reasons = []
            if provenance_list:
                for prov in provenance_list:
                    if isinstance(prov, dict):
                        text = prov.get('text', '')
                        source_url = prov.get('source_url', '')
                        if text:
                            if source_url and source_url.lower() != 'unknown':
                                reasons.append(f"{text}|||{source_url}")
                            else:
                                reasons.append(text)
            else:
                # Fallback to legacy 'reasons' field if no provenance
                reasons = data.get('reasons', [])
            
            # Deduplicate reasons
            unique_reasons = []
            seen_reasons = set()
            
            for r in reasons:
                # Normalize reason for comparison (remove extra whitespace from text part only)
                if '|||' in r:
                    text_part = r.split('|||')[0]
                    normalized = ' '.join(text_part.split())
                else:
                    normalized = ' '.join(r.split())
                    
                if normalized not in seen_reasons:
                    seen_reasons.add(normalized)
                    unique_reasons.append(r)
            
            # Update the graph data with deduplicated reasons
            data['reasons'] = unique_reasons
            
            # Format reasons for pinned panel (full detail)
            if unique_reasons:
                reasons_html = '<div style="max-width: 100%; font-family: Arial, sans-serif;">'
                
                # Add relation type if available
                if relation_type:
                    reasons_html += f'<div style="margin-bottom: 12px; padding: 10px; background-color: #e8f4f8; border-left: 4px solid #0066cc; border-radius: 3px;">'
                    reasons_html += f'<p style="margin: 0; font-size: 0.95em;"><strong>🔗 Relation:</strong> {relation_type}</p>'
                    reasons_html += '</div>'
                
                # Add temporal info header if dates exist
                if dates:
                    dates_str = ', '.join(dates)
                    reasons_html += f'<div style="margin-bottom: 12px; padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 3px;">'
                    reasons_html += f'<p style="margin: 0; font-size: 0.95em;"><strong>📅 Timeline:</strong> {dates_str}</p>'
                    reasons_html += '</div>'
                
                # Add confidence score if available
                if confidence is not None:
                    conf_color = '#28a745' if confidence > 0.7 else '#ffc107' if confidence > 0.4 else '#dc3545'
                    reasons_html += f'<div style="margin-bottom: 12px; padding: 8px; background-color: #e9ecef; border-left: 4px solid {conf_color}; border-radius: 3px;">'
                    reasons_html += f'<p style="margin: 0; font-size: 0.9em;"><strong>Confidence:</strong> {confidence:.2%}</p>'
                    reasons_html += '</div>'
                
                for idx, r in enumerate(unique_reasons, 1):
                    # Split reason and URL(s) if present
                    if '|||' in r:
                        parts = r.split('|||')
                        sentence = parts[0].strip()
                        urls = [url.strip() for url in parts[1:] if url.strip()]
                    else:
                        sentence = r
                        urls = []

                    # Collect provenance URLs from edge data (prefer these if reason string lacks reliable link)
                    prov_urls = []
                    for p in data.get('provenance', []):
                        try:
                            # provenance stored as dicts (provenances_to_dicts)
                            if isinstance(p, dict):
                                su = p.get('source_url') or p.get('source') or ''
                            else:
                                # fallback if object
                                su = getattr(p, 'source_url', '') or getattr(p, 'source', '') or ''
                        except Exception:
                            su = ''
                        if su and su.lower() != 'unknown' and (su.startswith('http://') or su.startswith('https://')):
                            if su not in prov_urls:
                                prov_urls.append(su)

                    # Filter out unreliable URLs (unknown, empty, or not proper URLs) from reason string first
                    parsed_urls = [
                        url for url in urls
                        if url and url.lower() != 'unknown' and (url.startswith('http://') or url.startswith('https://'))
                    ]

                    # Prefer parsed_urls (explicit in reason) but fall back to provenance URLs
                    reliable_urls = parsed_urls[:]  # copy
                    for pu in prov_urls:
                        if pu not in reliable_urls:
                            reliable_urls.append(pu)

                    reasons_html += f'<div style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-left: 3px solid #0066cc; border-radius: 3px;">'

                    # Make the sentence itself clickable if there's a reliable URL
                    if reliable_urls:
                        primary_url = reliable_urls[0]
                        reasons_html += f'<p style="margin: 0; white-space: normal; word-wrap: break-word; line-height: 1.5; font-size: 0.9em;">'
                        reasons_html += f'<strong style="color: #0066cc;">{idx}.</strong> '
                        reasons_html += f'<a href="{primary_url}" target="_blank" style="color: #333; text-decoration: none; border-bottom: 1px dotted #0066cc;" title="Click to view source">{sentence}</a></p>'
                    else:
                        reasons_html += f'<p style="margin: 0; white-space: normal; word-wrap: break-word; line-height: 1.5; font-size: 0.9em;">'
                        reasons_html += f'<strong style="color: #0066cc;">{idx}.</strong> {sentence}</p>'

                    # Display all reliable source URLs (from parsed reason or provenance)
                    if reliable_urls:
                        if len(reliable_urls) == 1:
                            url = reliable_urls[0]
                            display_url = url if len(url) <= 50 else url[:47] + '...'
                            reasons_html += f'<p style="margin: 5px 0 0 0; font-size: 0.85em;"><a href="{url}" target="_blank" style="color: #0066cc; text-decoration: none;">📎 {display_url}</a></p>'
                        else:
                            # Multiple sources - display as a list
                            reasons_html += f'<p style="margin: 5px 0 2px 0; font-size: 0.85em; color: #666;"><strong>Sources ({len(reliable_urls)}):</strong></p>'
                            for url_idx, url in enumerate(reliable_urls, 1):
                                display_url = url if len(url) <= 45 else url[:42] + '...'
                                reasons_html += f'<p style="margin: 2px 0 2px 15px; font-size: 0.8em;"><a href="{url}" target="_blank" style="color: #0066cc; text-decoration: none;">📎 {url_idx}. {display_url}</a></p>'
                    elif urls and not reliable_urls:
                        # Show a note if there were URLs in original reason but none were reliable and none in provenance
                        reasons_html += f'<p style="margin: 5px 0 0 0; font-size: 0.8em; color: #999; font-style: italic;">⚠️ Source URL not available</p>'

                    reasons_html += '</div>'

                # Collect document URLs for this connection and build a global index for the dropdown.
                doc_urls: List[str] = []
                # Prefer provenance URLs, then parsed URLs.
                for p in data.get('provenance', []) or []:
                    if isinstance(p, dict):
                        su = str(p.get('source_url') or '')
                        if su and is_pdf_url(su):
                            doc_urls.append(su)
                for r in unique_reasons:
                    if isinstance(r, str) and '|||' in r:
                        maybe_url = r.rsplit('|||', 1)[-1].strip()
                        if maybe_url and is_pdf_url(maybe_url):
                            doc_urls.append(maybe_url)

                # De-dupe while preserving order
                seen_doc = set()
                doc_urls_unique: List[str] = []
                for du in doc_urls:
                    key = du.lower().rstrip('/')
                    if key in seen_doc:
                        continue
                    seen_doc.add(key)
                    doc_urls_unique.append(du)

                if doc_urls_unique:
                    # Inline dropdown in the pinned connection panel.
                    reasons_html += '<details style="margin-top: 12px; padding: 10px; background-color: #ffffff; border: 1px solid #dcdfe6; border-radius: 6px;">'
                    reasons_html += f'<summary style="cursor: pointer; color: #0066cc; font-weight: bold;">📄 Documents ({len(doc_urls_unique)})</summary>'
                    reasons_html += '<ul style="margin: 10px 0 0 18px; padding: 0;">'
                    for du in doc_urls_unique:
                        reasons_html += f'<li style="margin: 6px 0;"><a href="{du}" target="_blank" style="color: #0066cc; text-decoration: none;">{du}</a></li>'
                    reasons_html += '</ul>'
                    reasons_html += '</details>'

                    connection_label = f"{graph.nodes[u].get('display_name', u)} → {graph.nodes[v].get('display_name', v)}"
                    for du in doc_urls_unique:
                        entry = documents_index.get(du)
                        if not entry:
                            entry = {'connections': [], 'nodes': []}
                            documents_index[du] = entry
                        if connection_label not in entry['connections']:
                            entry['connections'].append(connection_label)
                        for node_name in (graph.nodes[u].get('display_name', u), graph.nodes[v].get('display_name', v)):
                            if node_name not in entry['nodes']:
                                entry['nodes'].append(str(node_name))
                
                reasons_html += '</div>'
                
                # Compact version for hover tooltip
                reasons_compact = ''
                
                # Add relation type to hover if present
                if relation_type:
                    reasons_compact += f'<strong>🔗 {relation_type}</strong><br>'
                
                # Add dates to hover if present
                if dates:
                    dates_str = ', '.join(dates[:3])  # Show first 3 dates
                    if len(dates) > 3:
                        dates_str += f' (+{len(dates)-3} more)'
                    reasons_compact += f'<strong>📅 {dates_str}</strong><br>'
                
                # Add confidence to hover
                if confidence is not None:
                    reasons_compact += f'<strong>Confidence: {confidence:.1%}</strong><br>'
                
                reasons_compact += f'<strong>{len(unique_reasons)} connection(s) found</strong><br>'
                for idx, r in enumerate(unique_reasons[:2], 1):
                    if '|||' in r:
                        sentence = r.split('|||')[0]
                    else:
                        sentence = r
                    # Truncate long sentences
                    display_sentence = sentence if len(sentence) <= 100 else sentence[:97] + '...'
                    reasons_compact += f'{idx}. {display_sentence}<br>'
                if len(unique_reasons) > 2:
                    reasons_compact += f'<em>... and {len(unique_reasons) - 2} more</em>'
            else:
                reasons_html = '<p style="color: #999;">No specific reason found</p>'
                reasons_compact = '<em>No specific reason found</em>'
            
            edge_reasons.append(reasons_html)
            edge_reasons_compact.append(reasons_compact)
        
        # Calculate edge midpoints for interaction targets
        edge_pos = graph_renderer.layout_provider.graph_layout
        edge_x = [
            (edge_pos[start][0] + edge_pos[end][0]) / 2 
            for start, end in zip(edge_start, edge_end)
        ]
        edge_y = [
            (edge_pos[start][1] + edge_pos[end][1]) / 2 
            for start, end in zip(edge_start, edge_end)
        ]
        
        # Create data source for edge interaction (include human-readable display names)
        edge_source = ColumnDataSource(data=dict(
            x=edge_x,
            y=edge_y,
            start=edge_start,
            end=edge_end,
            start_display=[graph.nodes[n].get('display_name', n) for n in edge_start],
            end_display=[graph.nodes[n].get('display_name', n) for n in edge_end],
            reasons=edge_reasons,
            reasons_compact=edge_reasons_compact
        ))
        
        # Create Div for displaying pinned connections on the left side
        pinned_div = Div(
            text="""
            <div style='padding: 20px; background-color: #f8f9fa; border-right: 2px solid #dee2e6; height: 100%; overflow-y: auto;'>
                <h3 style='margin-top: 0; color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;'>
                    📌 Pinned Connections
                </h3>
                <p style='color: #666; font-size: 0.9em; line-height: 1.5;'>
                    Click on connections (edges) to pin them here. Each pinned item can be removed individually.
                </p>
                <div id='pinned-items' style='margin-top: 15px;'></div>
            </div>
            """,
            width=350,
            height=520,
            sizing_mode="fixed"
        )

        # Create invisible circles at edge midpoints for interaction
        edge_circles = plot.circle(
            'x', 'y',
            source=edge_source,
            size=12,
            alpha=0,
            hover_alpha=0.4,
            hover_color="orange",
            selection_alpha=0.6,
            selection_color="red",
            nonselection_alpha=0
        )
        
        # Add hover tool with compact information (show display names rather than raw node ids)
        hover = HoverTool(
            renderers=[edge_circles],
            tooltips="""
            <div style="max-width: 400px; padding: 12px; background-color: white; border: 2px solid #0066cc; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-size: 13px; font-weight: bold; margin-bottom: 8px; color: #0066cc;">
                    @start_display → @end_display
                </div>
                <div style="font-size: 11px; color: #333; line-height: 1.4;">
                    @reasons_compact{safe}
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 10px; color: #666;">
                    💡 <em>Click to pin this connection on the right panel</em>
                </div>
            </div>
            """
        )
        plot.add_tools(hover)
        
        # Add tap/click functionality with JavaScript callback to pin items
        # Use Tap event (cb_obj.x/y) and hit-test nearest edge midpoint
        tap_callback = CustomJS(args=dict(source=edge_source, div=pinned_div, x_range=plot.x_range, y_range=plot.y_range), code="""
            if (!cb_obj || cb_obj.x === undefined || cb_obj.y === undefined) {
                return;
            }

            const tapX = cb_obj.x;
            const tapY = cb_obj.y;
            const xs = source.data['x'] || [];
            const ys = source.data['y'] || [];

            let minDist = Infinity;
            let minIdx = -1;
            for (let i = 0; i < xs.length; i++) {
                const dx = tapX - xs[i];
                const dy = tapY - ys[i];
                const d2 = dx * dx + dy * dy;
                if (d2 < minDist) {
                    minDist = d2;
                    minIdx = i;
                }
            }

            const spanX = Math.abs((x_range.end || 0) - (x_range.start || 0)) || 1.0;
            const spanY = Math.abs((y_range.end || 0) - (y_range.start || 0)) || 1.0;
            const threshold = 0.03 * Math.max(spanX, spanY);

            if (minIdx < 0 || minDist > (threshold * threshold)) {
                return;
            }

            const idx = minIdx;
            const start = source.data['start'][idx];
            const end = source.data['end'][idx];
            const startDisplay = source.data['start_display'][idx] || start;
            const endDisplay = source.data['end_display'][idx] || end;
            const reasons = source.data['reasons'][idx];

                // Create a stable key so repeated clicks don't create duplicates.
                const rawKey = [String(start), String(end)].sort().join('||');
                // Keep IDs simple/safe (avoid relying on CSS.escape for selectors)
                const edgeKey = rawKey.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 120);
                const connectionId = 'conn_' + edgeKey;

                // Base HTML shell (matches the Python Div's initial content).
                const PINNED_BASE_START = `
                    <div style='padding: 20px; background-color: #f8f9fa; border-right: 2px solid #dee2e6; height: 100%; overflow-y: auto;'>
                        <h3 style='margin-top: 0; color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;'>
                            📌 Pinned Connections
                        </h3>
                        <p style='color: #666; font-size: 0.9em; line-height: 1.5;'>
                            Click on connections (edges) to pin them here. Each pinned item can be removed individually.
                        </p>
                        <div id='pinned-items' style='margin-top: 15px;'>`;
                const PINNED_BASE_END = `</div></div>`;

                function pinnedItemHTML(item) {
                    return `
                        <div id="${item.id}" style="margin-bottom: 15px; padding: 12px; background-color: white; border: 1px solid #0066cc; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                                <h4 style="margin: 0; color: #0066cc; font-size: 1em; flex-grow: 1;">
                                    🔗 ${item.startDisplay} → ${item.endDisplay}
                                </h4>
                                <button data-action="remove" data-target="${item.id}"
                                        style="background-color: #dc3545; color: white; border: none; border-radius: 3px; padding: 4px 8px; cursor: pointer; font-size: 0.85em; margin-left: 8px; flex-shrink: 0;">
                                    ✕
                                </button>
                            </div>
                            <div style="font-size: 0.85em;">${item.reasons}</div>
                        </div>
                    `;
                }

                function ensurePinnedState() {
                    if (!window.__kgPinnedState) {
                        window.__kgPinnedState = { items: [] };
                    }
                    // Keep stable references so other global handlers can persist removals.
                    window.__kgPinnedDivModel = div;

                    window.__kgPinnedRender = function () {
                        const st = window.__kgPinnedState || { items: [] };
                        const html = PINNED_BASE_START + (st.items || []).map(pinnedItemHTML).join('') + PINNED_BASE_END;
                        // Keep Bokeh model in sync so removed pins don't reappear.
                        window.__kgPinnedDivModel.text = html;
                    };

                    window.__kgPinnedRemoveById = function (targetId) {
                        const st = window.__kgPinnedState || { items: [] };
                        st.items = (st.items || []).filter(it => it && it.id !== targetId);
                        window.__kgPinnedState = st;
                        if (window.__kgPinnedRender) window.__kgPinnedRender();
                    };

                    return window.__kgPinnedState;
                }

                function ensurePinnedDelegation(divElement) {
                    // Attach a single document-level handler (capture) that is Shadow-DOM-safe.
                    // Bokeh widgets render inside shadow roots; regular document.getElementById won't find nodes there.
                    if (document.__kgPinnedHandlerAttached) return;
                    document.__kgPinnedHandlerAttached = true;
                    document.addEventListener('click', function (e) {
                        const path = (e && e.composedPath) ? e.composedPath() : [];

                        // Remove pinned items
                        let btn = null;
                        for (const n of path) {
                            if (n && n.getAttribute && n.getAttribute('data-action') === 'remove') {
                                btn = n;
                                break;
                            }
                        }
                        if (btn) {
                            const targetId = btn.getAttribute('data-target');
                            if (targetId) {
                                e.preventDefault();
                                e.stopPropagation();
                                try {
                                    if (window.__kgPinnedRemoveById && typeof window.__kgPinnedRemoveById === 'function') {
                                        window.__kgPinnedRemoveById(targetId);
                                    } else {
                                        const root = (btn.getRootNode && btn.getRootNode()) ? btn.getRootNode() : document;
                                        let el = null;
                                        if (root && root.querySelector) {
                                            el = root.querySelector('#' + targetId);
                                        }
                                        if (!el) {
                                            el = document.getElementById(targetId);
                                        }
                                        if (el) el.remove();
                                    }
                                } catch (err) {
                                    // no-op
                                }
                            }
                            return;
                        }

                        // Open external/file links reliably from within shadow DOM
                        let anchor = null;
                        for (const n of path) {
                            if (n && n.tagName && String(n.tagName).toLowerCase() === 'a' && n.href) {
                                anchor = n;
                                break;
                            }
                        }
                        if (!anchor) return;
                        const href = String(anchor.href || '');
                        if (!(href.startsWith('http://') || href.startsWith('https://') || href.startsWith('file://'))) return;
                        try {
                            const w = window.open(href, anchor.target || '_blank');
                            if (w) {
                                e.preventDefault();
                                e.stopPropagation();
                            }
                        } catch (err) {
                            // allow default
                        }
                    }, true);
                }

                // Short-term memory: keep pinned items in a JS state list and always render via div.text.
                // This prevents previously removed items from reappearing when pinning new connections.
                const st = ensurePinnedState();
                const divElement = document.getElementById(div.id);
                if (divElement) ensurePinnedDelegation(divElement);

                const item = { id: connectionId, startDisplay: startDisplay, endDisplay: endDisplay, reasons: reasons };
                st.items = (st.items || []).filter(it => it && it.id !== connectionId);
                st.items.unshift(item);
                window.__kgPinnedState = st;
                if (window.__kgPinnedRender) window.__kgPinnedRender();
        """)
        
        tap_tool = TapTool(renderers=[edge_circles])
        plot.add_tools(tap_tool)

        if Tap is not None:
            plot.js_on_event(Tap, tap_callback)
        
        logger.info(f"Added interactive edges with hover tooltips and click-to-pin functionality")
        
        return pinned_div, documents_index
    
    def _add_edge_hover(self, plot, graph, graph_renderer):
        """
        Add hover tooltips for edges with full sentences and source URLs.
        
        Args:
            plot: Bokeh plot object
            graph: NetworkX graph
            graph_renderer: Bokeh graph renderer
        """
        edge_start = []
        edge_end = []
        edge_reasons = []
        
        # Collect edge data
        for u, v, data in graph.edges(data=True):
            edge_start.append(u)
            edge_end.append(v)
            
            # Format reasons as HTML with proper text wrapping and source links
            reasons = data.get('reasons', [])
            if reasons:
                reasons_html = '<div style="max-width: 400px;">'
                
                # Show up to 3 most detailed reasons
                for idx, r in enumerate(reasons[:3], 1):
                    # Split reason and URL if present (format: "sentence|||url")
                    if '|||' in r:
                        sentence, url = r.split('|||', 1)
                    else:
                        sentence = r
                        url = None
                    
                    # Wrap text properly
                    reasons_html += f'<div style="margin-bottom: 10px; padding: 5px; border-left: 2px solid #0066cc;">'
                    reasons_html += f'<p style="margin: 0; white-space: normal; word-wrap: break-word;"><strong>{idx}.</strong> {sentence}</p>'
                    
                    if url:
                        # Truncate URL for display but keep full URL in href
                        display_url = url if len(url) <= 60 else url[:57] + '...'
                        reasons_html += f'<p style="margin: 5px 0 0 0; font-size: 0.85em;"><a href="{url}" target="_blank" style="color: #0066cc;">📎 {display_url}</a></p>'
                    
                    reasons_html += '</div>'
                
                if len(reasons) > 3:
                    reasons_html += f'<p style="margin-top: 5px; font-style: italic; color: #666;">... and {len(reasons) - 3} more connection(s)</p>'
                
                reasons_html += '</div>'
            else:
                reasons_html = '<p style="color: #999;">No specific reason found</p>'
            
            edge_reasons.append(reasons_html)
        
        # Calculate edge midpoints for hover targets
        edge_pos = graph_renderer.layout_provider.graph_layout
        edge_x = [
            (edge_pos[start][0] + edge_pos[end][0]) / 2 
            for start, end in zip(edge_start, edge_end)
        ]
        edge_y = [
            (edge_pos[start][1] + edge_pos[end][1]) / 2 
            for start, end in zip(edge_start, edge_end)
        ]
        
        # Create data source for edge hover
        # Create data source for edge hover (include human-readable display names)
        edge_source = ColumnDataSource(data=dict(
            x=edge_x,
            y=edge_y,
            start=edge_start,
            end=edge_end,
            start_display=[graph.nodes[n].get('display_name', n) for n in edge_start],
            end_display=[graph.nodes[n].get('display_name', n) for n in edge_end],
            reasons=edge_reasons
        ))
        
        # Create invisible circles at edge midpoints
        edge_hover_glyph = plot.circle(
            'x', 'y',
            source=edge_source,
            size=10,
            alpha=0,
            hover_alpha=0.3,
            hover_color="orange"
        )
        
        # Add edge hover tool with persistent tooltip
        edge_hover = HoverTool(
            renderers=[edge_hover_glyph],
            tooltips="""
            <div style="width: 450px; padding: 10px; background-color: white; border: 2px solid #0066cc; border-radius: 5px;">
                <div style="font-size: 14px; font-weight: bold; margin-bottom: 10px; color: #0066cc;">
                    @start_display → @end_display
                </div>
                @{relation_type}{
                    <div style="font-size: 12px; margin-bottom: 8px; padding: 4px 8px; background-color: #e8f4f8; border-radius: 3px; display: inline-block;">
                        <span style="font-weight: bold; color: #0066cc;">Relation:</span> @relation_type
                    </div>
                }
                <div style="font-size: 12px;">
                    @reasons{safe}
                </div>
            </div>
            """,
            mode='mouse',  # Show on mouse over
            point_policy='follow_mouse'  # Follow mouse movement
        )
        plot.add_tools(edge_hover)
