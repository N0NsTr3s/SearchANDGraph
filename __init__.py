"""
Knowledge Graph Extractor - Extract entities and relationships from web pages.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from utils.config import AppConfig, CrawlerConfig, NLPConfig, VisualizationConfig
from scraper.crawler import WebCrawler
from processor.nlp_processor import NLPProcessor
from processor.graph_builder import KnowledgeGraph
from vizualization import GraphVisualizer
from utils.translator import TextTranslator
from utils.logger import setup_logger

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
