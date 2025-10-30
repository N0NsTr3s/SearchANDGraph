"""
Knowledge Graph Extractor - Extract entities and relationships from web pages.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import AppConfig, CrawlerConfig, NLPConfig, VisualizationConfig
from .crawler import WebCrawler
from .nlp_processor import NLPProcessor
from .graph_builder import KnowledgeGraph
from .visualizer import GraphVisualizer
from .translator import TextTranslator
from .logger import setup_logger

__all__ = [
    'AppConfig',
    'CrawlerConfig',
    'NLPConfig',
    'VisualizationConfig',
    'WebCrawler',
    'NLPProcessor',
    'KnowledgeGraph',
    'GraphVisualizer',
    'TextTranslator',
    'setup_logger',
]
