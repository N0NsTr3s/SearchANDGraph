"""
Duplicate Node Detector - Find potentially duplicate entities with similar names.

This script analyzes a knowledge graph to detect nodes that might represent
the same entity but have slightly different names.
"""

import pickle
import sys
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import argparse


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names.
    
    Args:
        name1: First name
        name2: Second name
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize both names
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    # Use SequenceMatcher for similarity
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Boost similarity if one is substring of other
    if norm1 in norm2 or norm2 in norm1:
        similarity = max(similarity, 0.9)
    
    # Boost similarity if all words in shorter name appear in longer name
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 and words2:
        if words1.issubset(words2) or words2.issubset(words1):
            similarity = max(similarity, 0.88)
    
    return similarity


def normalize_name(name: str) -> str:
    """
    Normalize a name for comparison.
    
    Args:
        name: Name to normalize
        
    Returns:
        Normalized name
    """
    import unicodedata
    
    # Convert to lowercase
    text = name.lower()
    
    # Remove diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Handle special characters
    replacements = {
        'ø': 'o', 'œ': 'oe', 'æ': 'ae',
        'ß': 'ss', 'ð': 'd', 'þ': 'th',
        'ł': 'l', 'đ': 'd', 'ı': 'i'
    }
    for foreign, ascii_equiv in replacements.items():
        text = text.replace(foreign, ascii_equiv)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def find_duplicate_groups(nodes: list, similarity_threshold: float = 0.85) -> list:
    """
    Find groups of nodes that are likely duplicates.
    
    Args:
        nodes: List of node names
        similarity_threshold: Minimum similarity to consider duplicates
        
    Returns:
        List of duplicate groups
    """
    print(f"\nAnalyzing {len(nodes)} nodes for duplicates (threshold: {similarity_threshold})...")
    
    # Store duplicate groups
    duplicate_groups = []
    processed = set()
    
    # Compare each node with every other node
    for i, node1 in enumerate(nodes):
        if node1 in processed:
            continue
        
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(nodes)} nodes checked...")
        
        # Find all similar nodes
        similar_nodes = [node1]
        
        for node2 in nodes[i+1:]:
            if node2 in processed:
                continue
            
            similarity = calculate_similarity(node1, node2)
            
            if similarity >= similarity_threshold:
                similar_nodes.append(node2)
                processed.add(node2)
        
        # If we found similar nodes, add to groups
        if len(similar_nodes) > 1:
            duplicate_groups.append({
                'nodes': similar_nodes,
                'canonical': max(similar_nodes, key=len),  # Longest name as canonical
                'similarity': min(calculate_similarity(node1, n) for n in similar_nodes[1:])
            })
            processed.add(node1)
    
    print(f"  ✓ Found {len(duplicate_groups)} potential duplicate groups")
    
    return duplicate_groups


def analyze_graph(graph_file: str, similarity_threshold: float = 0.85, show_all: bool = False):
    """
    Analyze a knowledge graph for duplicate nodes.
    
    Args:
        graph_file: Path to the graph pickle file
        similarity_threshold: Minimum similarity to flag as duplicate
        show_all: If True, show all nodes (not just duplicates)
    """
    # Load the graph
    try:
        with open(graph_file, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"✓ Loaded graph from: {graph_file}")
    except Exception as e:
        print(f"✗ Error loading graph: {e}")
        return
    
    # Convert to KnowledgeGraph instance
    from graph_builder import KnowledgeGraph
    
    if isinstance(loaded_data, KnowledgeGraph):
        # Already a KnowledgeGraph instance
        knowledge_graph = loaded_data
        graph = knowledge_graph.get_graph()
    elif isinstance(loaded_data, dict) and 'graph' in loaded_data:
        # Legacy dict format - convert to KnowledgeGraph instance
        knowledge_graph = KnowledgeGraph.from_serialized(loaded_data)
        graph = knowledge_graph.get_graph()
    else:
        print(f"✗ Unsupported graph format loaded from: {graph_file} (type={type(loaded_data)})")
        if isinstance(loaded_data, dict):
            print(f"  Available keys: {list(loaded_data.keys())}")
        return
    
    nodes = list(graph.nodes())
    
    print(f"\n{'='*70}")
    print(f"KNOWLEDGE GRAPH ANALYSIS")
    print(f"{'='*70}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {graph.number_of_edges()}")
    
    # Show sample nodes
    if show_all and len(nodes) <= 50:
        print(f"\n{'='*70}")
        print(f"ALL NODES:")
        print(f"{'='*70}")
        for i, node in enumerate(sorted(nodes), 1):
            print(f"{i:3d}. {node}")
    elif len(nodes) <= 20:
        print(f"\n{'='*70}")
        print(f"ALL NODES:")
        print(f"{'='*70}")
        
        for i, node in enumerate(sorted(nodes), 1):
            degree = degree[node]
            print(f"{i:3d}. {node} (degree: {degree})")
    else:
        print(f"\n{'='*70}")
        print(f"SAMPLE NODES (first 20):")
        print(f"{'='*70}")
        for i, node in enumerate(sorted(nodes)[:20], 1):
            degree = degree[node]
            print(f"{i:3d}. {node} (degree: {degree})")
        print(f"... and {len(nodes) - 20} more")
    
    # Find duplicate groups
    print(f"\n{'='*70}")
    print(f"DUPLICATE DETECTION")
    print(f"{'='*70}")
    
    duplicate_groups = find_duplicate_groups(nodes, similarity_threshold)
    
    if not duplicate_groups:
        print("\n✓ No duplicate nodes found!")
        print("  All node names are sufficiently distinct.")
        return
    
    # Display duplicate groups
    print(f"\n{'='*70}")
    print(f"POTENTIAL DUPLICATES FOUND: {len(duplicate_groups)} groups")
    print(f"{'='*70}")
    
    total_duplicates = sum(len(group['nodes']) for group in duplicate_groups)
    potential_savings = total_duplicates - len(duplicate_groups)
    
    print(f"\nSummary:")
    print(f"  • {len(duplicate_groups)} groups of similar nodes")
    print(f"  • {total_duplicates} total nodes involved")
    print(f"  • {potential_savings} nodes could be merged")
    print(f"  • {len(nodes) - potential_savings} nodes after merging")
    
    print(f"\n{'─'*70}")
    
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nGroup {i}: {len(group['nodes'])} similar nodes")
        print(f"  Canonical (longest): {group['canonical']}")
        print(f"  Minimum similarity: {group['similarity']:.3f}")
        print(f"  Variants:")
        
        for node in sorted(group['nodes'], key=len, reverse=True):
            if node != group['canonical']:
                degree = degree[node]
                sim = calculate_similarity(group['canonical'], node)
                print(f"    • {node} (degree: {degree}, similarity: {sim:.3f})")
                
                # Show normalized version
                norm_canonical = normalize_name(group['canonical'])
                norm_variant = normalize_name(node)
                if norm_canonical != norm_variant:
                    print(f"      Normalized: '{norm_variant}' vs '{norm_canonical}'")
        
        # Show connections for nodes in this group
        print(f"  Connections:")
        for node in group['nodes']:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                print(f"    {node} → {', '.join(neighbors[:3])}")
                if len(neighbors) > 3:
                    print(f"      ... and {len(neighbors) - 3} more")
    
    # Show recommendations
    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"\n1. Entity Disambiguation:")
    print(f"   • Entity disambiguation is {'ENABLED' if knowledge_graph.use_disambiguation else 'DISABLED'}")
    if knowledge_graph.use_disambiguation and knowledge_graph.disambiguator:
        print(f"   • Current threshold: {knowledge_graph.disambiguator.similarity_threshold}")
        print(f"   • Canonical entities: {knowledge_graph.disambiguator.stats.get('canonical_entities', 0)}")
        print(f"   • Aliases merged: {knowledge_graph.disambiguator.stats.get('aliases_merged', 0)}")
    
    print(f"\n2. To reduce duplicates:")
    print(f"   • Enable entity disambiguation in config:")
    print(f"     config.graph.enable_disambiguation = True")
    print(f"   • Lower similarity threshold (currently: {similarity_threshold}):")
    print(f"     config.graph.similarity_threshold = 0.80  # More aggressive merging")
    print(f"   • Enable auto-discovery:")
    print(f"     config.graph.auto_discover_aliases = True")
    
    print(f"\n3. Manual fixes:")
    print(f"   • Add predefined aliases to disambiguation system")
    print(f"   • Review and manually merge obvious duplicates")
    print(f"   • Use consistent entity extraction (enable translation)")


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate nodes in knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default graph
  python check_duplicates.py

  # Analyze specific graph file
  python check_duplicates.py --graph scans/nicusor_dan/knowledge_graph.pkl
  
  # Use different similarity threshold
  python check_duplicates.py --threshold 0.80
  
  # Show all nodes
  python check_duplicates.py --show-all
        """
    )
    
    parser.add_argument(
        '--graph', '-g',
        default='knowledge_graph.pkl',
        help='Path to graph pickle file (default: knowledge_graph.pkl)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.85,
        help='Similarity threshold (0.0-1.0, default: 0.85)'
    )
    
    parser.add_argument(
        '--show-all', '-a',
        action='store_true',
        help='Show all nodes (not just duplicates)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.graph).exists():
        print(f"✗ Graph file not found: {args.graph}")
        print(f"\nAvailable scan directories:")
        
        # Look for scan directories
        scans_dir = Path("scans")
        if scans_dir.exists():
            for scan_dir in scans_dir.iterdir():
                if scan_dir.is_dir():
                    graph_file = scan_dir / "knowledge_graph.pkl"
                    if graph_file.exists():
                        print(f"  • {graph_file}")
        
        # Look in current directory
        current_graph = Path("knowledge_graph.pkl")
        if current_graph.exists():
            print(f"  • {current_graph}")
        
        return 1
    
    analyze_graph(args.graph, args.threshold, args.show_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
