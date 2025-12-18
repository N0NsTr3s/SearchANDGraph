"""
Structured provenance tracking for knowledge graph relations.

Provenance records track the origin and confidence of extracted relations,
enabling filtering, auditing, and source attribution.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime
import re


@dataclass
class Provenance:
    """
    Structured provenance record for a relation extraction.
    
    Attributes:
        text: The sentence or text snippet supporting this relation
        source_url: URL of the source document
        confidence: Confidence score (0.0-1.0), None if not available
        timestamp: When this relation was extracted (ISO format)
        dates: List of temporal references found in the text
        sentence_offset: Character offset of the sentence in source document
        relation_type: Classified relation type (e.g., is_CEO_of, born_in), None if not classified
    """
    text: str
    source_url: str
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    dates: Optional[List[str]] = field(default_factory=list)
    sentence_offset: Optional[int] = None
    relation_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure confidence is in valid range
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        
        # Ensure dates is always a list
        if self.dates is None:
            self.dates = []
        elif isinstance(self.dates, str):
            self.dates = [self.dates]
        
        # Generate timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Provenance':
        """Create Provenance from dictionary."""
        return cls(**data)
    
    def to_legacy_string(self) -> str:
        """
        Convert to legacy 'text|||url|||confidence:X[dates:...]' format.
        Used for backwards compatibility.
        """
        parts = [self.text, self.source_url]
        
        if self.confidence is not None:
            parts.append(f"confidence:{self.confidence}")
        
        if self.dates:
            dates_str = ','.join(self.dates)
            if self.confidence is not None:
                parts[-1] += f"[dates:{dates_str}]"
            else:
                parts.append(f"[dates:{dates_str}]")
        
        return '|||'.join(parts)
    
    @classmethod
    def from_legacy_string(cls, legacy: str) -> 'Provenance':
        """
        Parse legacy 'text|||url|||confidence:X[dates:...]' format.
        
        Args:
            legacy: Legacy reason string
            
        Returns:
            Provenance object
            
        Examples:
            >>> Provenance.from_legacy_string("He was elected|||http://example.com")
            >>> Provenance.from_legacy_string("He served|||http://a.com|||confidence:0.85")
            >>> Provenance.from_legacy_string("Born in|||http://b.com|||confidence:0.9[dates:1972,1980]")
        """
        # Split by ||| delimiter
        parts = legacy.split('|||')
        
        text = parts[0].strip() if len(parts) > 0 else ""
        source_url = parts[1].strip() if len(parts) > 1 else ""
        confidence = None
        dates = []
        
        # Parse confidence and dates from remaining parts
        if len(parts) > 2:
            remainder = parts[2].strip()
            
            # Extract confidence
            conf_match = re.search(r'confidence:\s*([\d.]+)', remainder)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except ValueError:
                    pass
            
            # Extract dates
            dates_match = re.search(r'\[dates:(.*?)\]', remainder)
            if dates_match:
                dates_str = dates_match.group(1)
                dates = [d.strip() for d in dates_str.split(',') if d.strip()]
        
        return cls(
            text=text,
            source_url=source_url,
            confidence=confidence,
            dates=dates if dates else None
        )


def migrate_legacy_reasons(reasons: List[str]) -> List[Provenance]:
    """
    Convert list of legacy reason strings to Provenance objects.
    
    Args:
        reasons: List of legacy reason strings
        
    Returns:
        List of Provenance objects
    """
    provenances = []
    for reason in reasons:
        try:
            prov = Provenance.from_legacy_string(reason)
            provenances.append(prov)
        except Exception as e:
            # If parsing fails, create minimal provenance
            print(f"Warning: Failed to parse legacy reason: {reason[:50]}... Error: {e}")
            provenances.append(Provenance(
                text=reason,
                source_url="unknown",
                confidence=None
            ))
    return provenances


def provenances_to_dicts(provenances: List[Provenance]) -> List[dict]:
    """Convert list of Provenance objects to list of dicts for serialization."""
    return [p.to_dict() for p in provenances]


def dicts_to_provenances(dicts: List[dict]) -> List[Provenance]:
    """Convert list of dicts to list of Provenance objects."""
    return [Provenance.from_dict(d) for d in dicts]


def deduplicate_provenances(provenances: List[Provenance]) -> List[Provenance]:
    """
    Deduplicate provenance records by text content and source URL.
    
    Args:
        provenances: List of Provenance objects
        
    Returns:
        Deduplicated list
    """
    seen = {}
    result = []
    
    for prov in provenances:
        # Use normalized text + url as key
        key = (prov.text.lower().strip(), prov.source_url.lower().strip())
        
        if key not in seen:
            seen[key] = prov
            result.append(prov)
        else:
            # Merge confidence (take max)
            existing = seen[key]
            if prov.confidence is not None:
                if existing.confidence is None:
                    existing.confidence = prov.confidence
                else:
                    existing.confidence = max(existing.confidence, prov.confidence)
            
            # Merge dates
            if prov.dates:
                for date in prov.dates:
                    if date not in existing.dates:
                        existing.dates.append(date)
    
    return result
