"""
Interactive Graph Visualization using Plotly

Creates interactive, zoomable, pannable graph visualizations with:
- Node hover information
- Clickable edges with source details
- Color coding by entity type
- Size coding by node importance
- Legend and filtering capabilities
"""

import logging
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple, Mapping, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class InteractiveGraphVisualizer:
    """
    Creates interactive graph visualizations using Plotly.
    """
    
    # Entity label colors
    LABEL_COLORS = {
        'PERSON': '#FF6B6B',  # Red
        'ORG': '#4ECDC4',  # Teal
        'GPE': '#45B7D1',  # Blue
        'LOC': '#96CEB4',  # Green
        'DATE': '#FFEAA7',  # Yellow
        'MONEY': '#DFE6E9',  # Gray
        'EVENT': '#A29BFE',  # Purple
        'WORK_OF_ART': '#FD79A8',  # Pink
        'UNKNOWN': '#B2BEC3',  # Light gray
        'DEFAULT': '#74B9FF'  # Sky blue
    }
    
    def __init__(
        self,
        graph: nx.Graph,
        title: str = "Knowledge Graph",
        node_size_by: str = "degree",
        color_by: str = "label",
        show_edge_labels: bool = False
    ):
        """
        Initialize the interactive visualizer.
        
        Args:
            graph: NetworkX graph to visualize
            title: Title for the visualization
            node_size_by: How to size nodes: "degree", "uniform", "confidence"
            color_by: How to color nodes: "label", "cluster", "query_distance"
            show_edge_labels: Whether to show relationship text on edges
        """
        self.graph = graph
        self.title = title
        self.node_size_by = node_size_by
        self.color_by = color_by
        self.show_edge_labels = show_edge_labels
        
        # Calculate layout
        logger.info(f"Calculating graph layout for {graph.number_of_nodes()} nodes...")
        self.pos = self._calculate_layout()
        logger.info("Layout calculation complete")
        
    def _calculate_layout(self) -> Mapping[Any, Tuple[float, float]]:
        """
        Calculate node positions using spring layout.
        
        Returns:
            Mapping from node keys to (x, y) coordinate tuples
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Use spring layout with improved parameters
        try:
            pos_raw = nx.spring_layout(
                self.graph,
                k=1/np.sqrt(self.graph.number_of_nodes()),  # Optimal distance
                iterations=50,
                seed=42  # Reproducible layout
            )
        except Exception as e:
            logger.warning(f"Spring layout failed: {e}. Using random layout.")
            pos_raw = nx.random_layout(self.graph, seed=42)
        
        # Convert numpy array-like coordinates to plain Python tuples of floats
        pos: Dict[Any, Tuple[float, float]] = {
            node: (float(coords[0]), float(coords[1]))
            for node, coords in pos_raw.items()
        }
        
        return pos

    def _display_name(self, node: Any) -> str:
        """Return a human-friendly name for a node.

        Graph nodes may use canonical IDs (e.g., Wikidata QIDs) as their keys.
        Prefer rich metadata (e.g., display_name) for rendering labels.
        """
        try:
            data = self.graph.nodes.get(node, {})  # type: ignore[attr-defined]
            if isinstance(data, dict):
                for key in ("display_name", "canonical_label", "label_name", "name", "title"):
                    value = data.get(key)
                    if value:
                        return str(value)
        except Exception:
            pass
        return str(node)
    
    def _get_node_sizes(self) -> Dict[str, float]:
        """
        Calculate node sizes based on configuration.
        
        Returns:
            Dictionary mapping node names to sizes
        """
        sizes = {}
        
        if self.node_size_by == "degree":
            # Size by degree (number of connections)
            # Convert degree view entries to ints so static type checkers treat them as numbers.
            degrees = {node: int(deg) for node, deg in self.graph.degree()} # type: ignore
            raw_max = max(degrees.values()) if degrees else 1
            max_degree = raw_max if raw_max > 0 else 1
            for node, degree in degrees.items():
                # Scale between 10 and 40
                sizes[node] = 10 + (degree / max_degree) * 30
                
        elif self.node_size_by == "confidence":
            # Size by average confidence of connected edges
            for node in self.graph.nodes():
                edges = self.graph.edges(node, data=True)
                confidences = [data.get('confidence', 0.5) for _, _, data in edges]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
                sizes[node] = 10 + avg_conf * 30
                
        else:  # uniform
            for node in self.graph.nodes():
                sizes[node] = 20
        
        return sizes
    
    def _get_node_colors(self) -> Dict[str, str]:
        """
        Calculate node colors based on configuration.
        
        Returns:
            Dictionary mapping node names to color strings
        """
        colors = {}
        
        if self.color_by == "label":
            # Color by entity type/label
            for node in self.graph.nodes():
                label = self.graph.nodes[node].get('label', 'UNKNOWN')
                colors[node] = self.LABEL_COLORS.get(label, self.LABEL_COLORS['DEFAULT'])
                
        elif self.color_by == "cluster":
            # Color by connected component
            components = list(nx.connected_components(self.graph))
            component_colors = {}
            
            # Generate distinct colors for each component
            for i, component in enumerate(components):
                hue = i / len(components)
                color = f'hsl({int(hue * 360)}, 70%, 60%)'
                for node in component:
                    component_colors[node] = color
            
            colors = component_colors
            
        else:  # default
            for node in self.graph.nodes():
                colors[node] = self.LABEL_COLORS['DEFAULT']
        
        return colors
    
    def _create_edge_trace(self) -> go.Scatter:
        """
        Create Plotly trace for edges.
        
        Returns:
            Plotly Scatter trace for edges
        """
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            
            if source not in self.pos or target not in self.pos:
                continue
            
            x0, y0 = self.pos[source]
            x1, y1 = self.pos[target]
            
            # Add edge line
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge hover text
            reasons = data.get('reasons', [])
            dates = data.get('dates', [])
            confidence = data.get('confidence', None)
            
            source_name = self._display_name(source)
            target_name = self._display_name(target)
            hover_text = f"<b>{source_name} ↔ {target_name}</b><br>"
            
            # Add temporal information
            if dates:
                dates_str = ', '.join(dates[:3])
                if len(dates) > 3:
                    dates_str += f' (+{len(dates)-3} more)'
                hover_text += f"📅 Timeline: {dates_str}<br>"
            
            # Add confidence
            if confidence:
                hover_text += f"Confidence: {confidence:.2%}<br>"
            
            hover_text += f"<br>Sources ({len(reasons)}):<br>"
            
            # Show first 3 reasons
            for i, reason in enumerate(reasons[:3]):
                if '|||' in reason:
                    text, url = reason.rsplit('|||', 1)
                    hover_text += f"• {text[:100]}...<br>"
                else:
                    hover_text += f"• {reason[:100]}...<br>"
            
            if len(reasons) > 3:
                hover_text += f"• ... and {len(reasons) - 3} more"
            
            edge_text.append(hover_text)
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=0.5,
                color='#888'
            ),
            hoverinfo='skip',
            showlegend=False
        )
        
        return edge_trace
    
    def _create_node_trace(self) -> go.Scatter:
        """
        Create Plotly trace for nodes.
        
        Returns:
            Plotly Scatter trace for nodes
        """
        node_x = []
        node_y = []
        node_text = []
        node_labels = []
        node_sizes = self._get_node_sizes()
        node_colors = self._get_node_colors()
        node_color_values = []
        
        for node in self.graph.nodes():
            if node not in self.pos:
                continue
            
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(self._display_name(node))
            
            # Node hover text
            label = self.graph.nodes[node].get('label', 'UNKNOWN')
            degree = self.graph.degree(node) # type: ignore
            
            hover_text = f"<b>{self._display_name(node)}</b><br>"
            hover_text += f"Type: {label}<br>"
            hover_text += f"Connections: {degree}<br>"
            
            # Show connected entities
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                hover_text += f"<br>Connected to:<br>"
                for i, neighbor in enumerate(neighbors[:5]):
                    hover_text += f"• {self._display_name(neighbor)}<br>"
                if len(neighbors) > 5:
                    hover_text += f"• ... and {len(neighbors) - 5} more"
            
            node_text.append(hover_text)
            node_color_values.append(node_colors.get(node, self.LABEL_COLORS['DEFAULT']))
        
        # Determine node sizes
        size_values = [node_sizes.get(node, 20) for node in self.graph.nodes() if node in self.pos]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=size_values,
                color=node_color_values,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            showlegend=False
        )
        
        return node_trace
    
    def _create_legend_traces(self) -> List[go.Scatter]:
        """
        Create legend entries for entity types.
        
        Returns:
            List of dummy traces for legend
        """
        # Count entities by label
        label_counts = defaultdict(int)
        for node in self.graph.nodes():
            label = self.graph.nodes[node].get('label', 'UNKNOWN')
            label_counts[label] += 1
        
        # Create legend traces
        legend_traces = []
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            color = self.LABEL_COLORS.get(label, self.LABEL_COLORS['DEFAULT'])
            
            legend_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                name=f"{label} ({count})",
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                showlegend=True
            )
            legend_traces.append(legend_trace)
        
        return legend_traces
    
    def create_figure(self) -> go.Figure:
        """
        Create the complete Plotly figure.
        
        Returns:
            Plotly Figure object
        """
        # Create traces
        edge_trace = self._create_edge_trace()
        node_trace = self._create_node_trace()
        legend_traces = self._create_legend_traces()
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace] + legend_traces,
            layout=go.Layout(
                title=dict(
                    text=self.title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
        )
        
        return fig
    
    def save(self, output_file: str):
        """
        Save the interactive visualization to HTML file.
        
        Args:
            output_file: Path to output HTML file
        """
        logger.info(f"Creating interactive visualization...")
        fig = self.create_figure()
        
        logger.info(f"Saving interactive visualization to {output_file}...")
        fig.write_html(
            output_file,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'knowledge_graph',
                    'height': 1200,
                    'width': 1600,
                    'scale': 2
                }
            }
        )
        
        logger.info(f"✓ Interactive visualization saved to {output_file}")
        
        # Print statistics
        logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  Edges: {self.graph.number_of_edges()}")
        logger.info(f"  Layout: {self.node_size_by} sizing, {self.color_by} coloring")


def create_interactive_visualization(
    graph: nx.Graph,
    output_file: str,
    title: str = "Knowledge Graph",
    node_size_by: str = "degree",
    color_by: str = "label",
    show_edge_labels: bool = False
):
    """
    Create and save an interactive graph visualization.
    
    Args:
        graph: NetworkX graph to visualize
        output_file: Path to output HTML file
        title: Title for the visualization
        node_size_by: How to size nodes: "degree", "uniform", "confidence"
        color_by: How to color nodes: "label", "cluster", "query_distance"
        show_edge_labels: Whether to show relationship text on edges
    """
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty, skipping interactive visualization")
        return
    
    visualizer = InteractiveGraphVisualizer(
        graph=graph,
        title=title,
        node_size_by=node_size_by,
        color_by=color_by,
        show_edge_labels=show_edge_labels
    )
    
    visualizer.save(output_file)
