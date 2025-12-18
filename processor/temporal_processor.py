"""
Temporal Analysis Module

Extracts and normalizes temporal information from text to enable time-aware
knowledge graph relationships.
"""

from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
import re
from datetime import datetime
import dateparser
from ..utils.logger import setup_logger

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span
else:
    Doc = Any
    Span = Any

logger = setup_logger(__name__)


class TemporalProcessor:
    """
    Extracts and normalizes dates from text to associate temporal information
    with relationships.
    """
    
    def __init__(self):
        """Initialize the temporal processor."""
        # Configure dateparser for better accuracy
        self.dateparser_settings = {
            'STRICT_PARSING': False,
            'PREFER_DAY_OF_MONTH': 'first',
            'PREFER_DATES_FROM': 'past',
            'RETURN_AS_TIMEZONE_AWARE': False
        }
        
        # Common temporal relation patterns
        self.temporal_patterns = [
            # "since 2020", "from 2020", "in 2020"
            (r'\b(?:since|from|in|during|on)\s+(\d{4})\b', 'year'),
            # "between 2020 and 2021", "from 2020 to 2021"
            (r'\b(?:between|from)\s+(\d{4})\s+(?:and|to)\s+(\d{4})\b', 'range'),
            # "January 2020", "Jan 2020"
            (r'\b([A-Z][a-z]+\.?\s+\d{4})\b', 'month_year'),
            # "2020-01-15", "2020/01/15"
            (r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b', 'full_date'),
        ]
        
        logger.info("Temporal processor initialized")
    
    def extract_and_normalize_dates(
        self, 
        doc: Doc
    ) -> List[Dict[str, Any]]:
        """
        Extract dates from a spaCy document and normalize them to YYYY-MM-DD format.
        
        Args:
            doc: spaCy processed document
            
        Returns:
            List of dictionaries containing:
                - 'text': Original date text
                - 'normalized': Normalized date (YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD for ranges)
                - 'start_char': Character position in document
                - 'end_char': End character position
                - 'type': Date type (single, range, year_only, etc.)
        """
        dates = []
        seen_positions = set()
        
        # First, extract dates from spaCy's DATE entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                # Skip if we've already processed this position
                if (ent.start_char, ent.end_char) in seen_positions:
                    continue
                
                normalized = self._normalize_date(ent.text)
                if normalized:
                    dates.append({
                        'text': ent.text,
                        'normalized': normalized,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'type': self._get_date_type(normalized)
                    })
                    seen_positions.add((ent.start_char, ent.end_char))
                    logger.debug(f"Extracted DATE entity: {ent.text} -> {normalized}")
        
        # Second, use regex patterns to catch dates spaCy might have missed
        text = doc.text
        for pattern, pattern_type in self.temporal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                # Skip if already found
                if (start, end) in seen_positions:
                    continue
                
                matched_text = match.group(0)
                normalized = self._normalize_date(matched_text)
                
                if normalized:
                    dates.append({
                        'text': matched_text,
                        'normalized': normalized,
                        'start_char': start,
                        'end_char': end,
                        'type': pattern_type
                    })
                    seen_positions.add((start, end))
                    logger.debug(f"Extracted pattern date: {matched_text} -> {normalized}")
        
        return sorted(dates, key=lambda x: x['start_char'])
    
    def _normalize_date(self, date_text: str) -> Optional[str]:
        """
        Normalize a date string to YYYY-MM-DD format.
        
        Args:
            date_text: Raw date text
            
        Returns:
            Normalized date string or None if parsing failed
        """
        # Handle date ranges (e.g., "2020 to 2021", "between 2020 and 2021")
        range_match = re.search(
            r'(?:between|from)?\s*(\d{4})\s+(?:to|and|-)\s+(\d{4})',
            date_text,
            re.IGNORECASE
        )
        if range_match:
            start_year, end_year = range_match.groups()
            return f"{start_year}-01-01/{end_year}-12-31"
        
        # Try parsing with dateparser
        try:
            parsed_date = dateparser.parse(
                date_text,
                settings=self.dateparser_settings  # type: ignore[arg-type]
            )
            
            if parsed_date:
                # Check if it's just a year
                if re.match(r'^\d{4}$', date_text.strip()):
                    return f"{date_text.strip()}-01-01"
                
                return parsed_date.strftime('%Y-%m-%d')
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_text}': {e}")
        
        return None
    
    def _get_date_type(self, normalized: str) -> str:
        """
        Determine the type of date based on its normalized format.
        
        Args:
            normalized: Normalized date string
            
        Returns:
            Date type: 'range', 'year_only', 'month_year', or 'full_date'
        """
        if '/' in normalized:
            return 'range'
        elif normalized.endswith('-01-01'):
            return 'year_only'
        elif normalized.endswith('-01'):
            return 'month_year'
        else:
            return 'full_date'
    
    def associate_dates_with_relation(
        self,
        relation_span: Tuple[int, int],
        dates: List[Dict[str, Any]],
        max_distance: int = 50
    ) -> List[str]:
        """
        Find dates that are temporally relevant to a relation based on proximity.
        
        Args:
            relation_span: (start_char, end_char) of the relation in the text
            dates: List of extracted date dictionaries
            max_distance: Maximum character distance to consider a date relevant
            
        Returns:
            List of normalized date strings associated with the relation
        """
        if not dates:
            return []
        
        relation_start, relation_end = relation_span
        relation_center = (relation_start + relation_end) // 2
        
        associated_dates = []
        
        for date_info in dates:
            date_center = (date_info['start_char'] + date_info['end_char']) // 2
            distance = abs(date_center - relation_center)
            
            # Only associate dates within max_distance characters
            if distance <= max_distance:
                associated_dates.append({
                    'normalized': date_info['normalized'],
                    'text': date_info['text'],
                    'distance': distance,
                    'type': date_info['type']
                })
        
        # Sort by distance (closest first)
        associated_dates.sort(key=lambda x: x['distance'])
        
        # Return only the normalized dates
        return [d['normalized'] for d in associated_dates]
    
    def extract_temporal_facts(
        self,
        doc: Any,  # spacy.tokens.Doc
        entities: List[Any]  # List[spacy.tokens.Span]
    ) -> List[Dict[str, Any]]:
        """
        Extract temporal facts about entities (birth dates, founding dates, etc.).
        
        Args:
            doc: spaCy document
            entities: List of entity spans
            
        Returns:
            List of temporal facts: [{'entity': 'John', 'fact_type': 'born', 'date': '1990-01-15'}, ...]
        """
        temporal_facts = []
        
        # Patterns for common temporal facts
        fact_patterns = [
            (r'born\s+(?:in|on)?\s*([^,\.]+)', 'born'),
            (r'founded\s+(?:in|on)?\s*([^,\.]+)', 'founded'),
            (r'established\s+(?:in|on)?\s*([^,\.]+)', 'established'),
            (r'died\s+(?:in|on)?\s*([^,\.]+)', 'died'),
            (r'created\s+(?:in|on)?\s*([^,\.]+)', 'created'),
        ]
        
        text = doc.text.lower()
        
        for ent in entities:
            # Look for temporal facts near this entity (within 100 characters)
            ent_start = ent.start_char
            ent_end = ent.end_char
            
            # Search window around entity
            search_start = max(0, ent_start - 100)
            search_end = min(len(text), ent_end + 100)
            search_text = text[search_start:search_end]
            
            for pattern, fact_type in fact_patterns:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    date_text = match.group(1).strip()
                    normalized = self._normalize_date(date_text)
                    
                    if normalized:
                        temporal_facts.append({
                            'entity': ent.text,
                            'fact_type': fact_type,
                            'date': normalized,
                            'source_text': match.group(0)
                        })
                        logger.debug(f"Extracted temporal fact: {ent.text} {fact_type} {normalized}")
        
        return temporal_facts
