"""
Utilities for managing query-specific scan directories.
"""
import re
from pathlib import Path
from datetime import datetime


def sanitize_query_for_path(query: str) -> str:
    """
    Convert a query string to a safe directory name.
    
    Args:
        query: The search query
        
    Returns:
        Sanitized string safe for use as directory name
        
    Examples:
        "Nicușor Dan" → "nicusor_dan"
        "What is AI?" → "what_is_ai"
        "COVID-19 Info" → "covid_19_info"
    """
    # Normalize unicode characters (remove diacritics)
    import unicodedata
    text = unicodedata.normalize('NFD', query)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special characters with underscores
    text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except spaces and hyphens
    text = re.sub(r'[\s-]+', '_', text)  # Replace spaces/hyphens with underscores
    text = re.sub(r'_+', '_', text)  # Collapse multiple underscores
    text = text.strip('_')  # Remove leading/trailing underscores
    
    # Limit length to avoid filesystem issues
    max_length = 50
    if len(text) > max_length:
        text = text[:max_length].rstrip('_')
    
    # Fallback if string becomes empty
    if not text:
        text = "query"
    
    return text


def get_scan_directory(query: str, base_dir: str = "scans", add_timestamp: bool = False) -> Path:
    """
    Get the directory path for a specific query scan.
    Creates the directory if it doesn't exist.
    
    Args:
        query: The search query
        base_dir: Base directory for all scans (default: "scans")
        add_timestamp: If True, append timestamp to make each scan unique
        
    Returns:
        Path object for the scan directory
        
    Examples:
        get_scan_directory("Nicușor Dan")
        → Path("scans/nicusor_dan")
        
        get_scan_directory("AI Research", add_timestamp=True)
        → Path("scans/ai_research_20251023_143022")
    """
    sanitized_query = sanitize_query_for_path(query)
    
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{sanitized_query}_{timestamp}"
    else:
        dir_name = sanitized_query
    
    scan_dir = Path(base_dir) / dir_name
    scan_dir.mkdir(parents=True, exist_ok=True)
    
    return scan_dir


def get_scan_paths(query: str, base_dir: str = "scans", add_timestamp: bool = False) -> dict:
    """
    Get all file paths for a scan organized in a query-specific directory.
    
    Args:
        query: The search query
        base_dir: Base directory for all scans
        add_timestamp: If True, append timestamp to make each scan unique
        
    Returns:
        Dictionary with paths for cache, graph, checkpoints, and visualizations
        
    Example:
        paths = get_scan_paths("Nicușor Dan")
        {
            'scan_dir': Path('scans/nicusor_dan'),
            'cache_dir': Path('scans/nicusor_dan/cache'),
            'graph_file': Path('scans/nicusor_dan/knowledge_graph.pkl'),
            'checkpoint_file': Path('scans/nicusor_dan/checkpoint.pkl'),
            'viz_file': Path('scans/nicusor_dan/knowledge_graph.html'),
            'interactive_viz_file': Path('scans/nicusor_dan/knowledge_graph_interactive.html')
        }
    """
    scan_dir = get_scan_directory(query, base_dir, add_timestamp)
    
    # Create cache subdirectory
    cache_dir = scan_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    return {
        'scan_dir': scan_dir,
        'cache_dir': cache_dir,
        'graph_file': scan_dir / "knowledge_graph.pkl",
        'checkpoint_file': scan_dir / "checkpoint.pkl",
        'viz_file': scan_dir / "knowledge_graph.html",
        'interactive_viz_file': scan_dir / "knowledge_graph_interactive.html",
        'log_file': scan_dir / "scan.log"
    }


def list_all_scans(base_dir: str = "scans") -> list[dict]:
    """
    List all existing scan directories.
    
    Args:
        base_dir: Base directory for all scans
        
    Returns:
        List of dictionaries with scan information
        
    Example:
        [
            {
                'query': 'nicusor_dan',
                'path': Path('scans/nicusor_dan'),
                'modified': datetime(2025, 10, 23, 14, 30),
                'has_graph': True,
                'has_cache': True
            },
            ...
        ]
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    scans = []
    for scan_dir in base_path.iterdir():
        if scan_dir.is_dir():
            graph_file = scan_dir / "knowledge_graph.pkl"
            cache_dir = scan_dir / "cache"
            
            scans.append({
                'query': scan_dir.name,
                'path': scan_dir,
                'modified': datetime.fromtimestamp(scan_dir.stat().st_mtime),
                'has_graph': graph_file.exists(),
                'has_cache': cache_dir.exists() and any(cache_dir.iterdir())
            })
    
    # Sort by modification time (newest first)
    scans.sort(key=lambda x: x['modified'], reverse=True)
    
    return scans
