"""
Entity Disambiguation Module

Handles entity name normalization, alias tracking, and canonicalization.
Works with translated (English) entity names from multi-language sources.
"""

import logging
from typing import Dict, Set, List, Optional
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)


class EntityDisambiguator:
    """
    Disambiguates entity names to handle variations and aliases.
    
    Features:
    - Wikidata QID-based primary disambiguation (e.g., Q312 for Apple Inc.)
    - Language-agnostic name normalization (works with translated names)
    - Grammatical case handling (genitive, dative)
    - Alias tracking and merging
    - Fuzzy matching for similar names (fallback when QID unavailable)
    - Automatic alias discovery from similar names
    
    NOTE: Expects entities to already be translated to English if translation is enabled.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, enable_auto_discovery: bool = True):
        """
        Initialize the entity disambiguator.
        
        Args:
            similarity_threshold: Minimum similarity (0-1) to consider names as aliases
            enable_auto_discovery: Automatically discover aliases from similar names
        """
        self.similarity_threshold = similarity_threshold
        self.enable_auto_discovery = enable_auto_discovery
        
        # QID-based mappings (primary disambiguation method)
        # QID -> canonical display name
        self.qid_to_canonical: Dict[str, str] = {}
        
        # QID -> set of aliases
        self.qid_aliases: Dict[str, Set[str]] = {}
        
        # Entity text -> QID (for quick lookup)
        self.text_to_qid: Dict[str, str] = {}
        
        # Fallback text-based mappings (when QID unavailable)
        # Canonical name -> set of aliases
        self.entity_aliases: Dict[str, Set[str]] = {}
        
        # Alias -> canonical name (reverse mapping for fast lookup)
        self.alias_to_canonical: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'total_entities': 0,
            'canonical_entities': 0,
            'aliases_merged': 0,
            'auto_discovered': 0,
            'qid_based_matches': 0,
            'text_based_matches': 0
        }
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching (language-agnostic).
        
        Features:
        - Converts non-English characters to ASCII equivalents (ă→a, ș→s, etc.)
        - Converts to lowercase for case-insensitive comparison
        - Removes extra whitespace
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text with ASCII characters
        """
        if not text:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Convert non-English characters to ASCII equivalents
        # This handles Romanian, German, French, Spanish, and many other languages
        import unicodedata
        
        # First normalize to NFD (decomposed form)
        # This separates base characters from diacritics
        # Example: "ă" becomes "a" + combining breve
        text = unicodedata.normalize('NFD', text)
        
        # Remove all combining marks (diacritics)
        # This leaves only the base ASCII characters
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Handle special cases that NFD doesn't decompose
        replacements = {
            'ø': 'o',   # Danish/Norwegian
            'œ': 'oe',  # French
            'æ': 'ae',  # Danish/Norwegian
            'ß': 'ss',  # German
            'ð': 'd',   # Icelandic
            'þ': 'th',  # Icelandic
            'ł': 'l',   # Polish
            'đ': 'd',   # Croatian/Vietnamese
            'ı': 'i',   # Turkish dotless i
        }
        
        for foreign, ascii_equiv in replacements.items():
            text = text.replace(foreign, ascii_equiv)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def register_entity_with_qid(self, entity_text: str, qid: str, canonical_label: str, aliases: Optional[List[str]] = None):
        """
        Register an entity with its Wikidata QID for QID-based disambiguation.
        
        Args:
            entity_text: Original entity text as found in document
            qid: Wikidata QID (e.g., "Q312" for Apple Inc.)
            canonical_label: Official label from Wikidata
            aliases: Optional list of known aliases
        """
        # Store QID mappings
        self.qid_to_canonical[qid] = canonical_label
        self.text_to_qid[entity_text] = qid
        
        # Initialize aliases set for this QID
        if qid not in self.qid_aliases:
            self.qid_aliases[qid] = {canonical_label, entity_text}
        else:
            self.qid_aliases[qid].add(entity_text)
        
        # Add provided aliases
        if aliases:
            for alias in aliases:
                self.qid_aliases[qid].add(alias)
                self.text_to_qid[alias] = qid
        
        # NEW: Create permanent links from normalized text to QID in alias_to_canonical
        # This ensures future lookups of this entity text always resolve to the same QID
        self.alias_to_canonical[self.normalize_text(entity_text)] = qid
        self.alias_to_canonical[self.normalize_text(canonical_label)] = qid
        for alias in (aliases or []):
            self.alias_to_canonical[self.normalize_text(alias)] = qid
        
        logger.debug(f"Registered '{entity_text}' with QID {qid} (canonical: '{canonical_label}')")
    
    def get_entity_identifier(self, entity_text: str, qid: Optional[str] = None) -> str:
        """
        Get the unique identifier for an entity (QID if available, otherwise cleaned text).
        
        Args:
            entity_text: Entity text
            qid: Optional Wikidata QID
            
        Returns:
            QID if available, otherwise cleaned entity text
        """
        # If QID provided, use it
        if qid:
            return qid
        
        # Check if we have a QID for this text
        if entity_text in self.text_to_qid:
            return self.text_to_qid[entity_text]
        
        # Fallback: return cleaned text
        return self.clean_entity_name(entity_text)
    
    def get_display_name(self, identifier: str) -> str:
        """
        Get the display name for an entity identifier.
        
        Args:
            identifier: QID or entity text
            
        Returns:
            Human-readable display name
        """
        # If it's a QID, return canonical label
        if identifier.startswith('Q') and identifier[1:].isdigit():
            return self.qid_to_canonical.get(identifier, identifier)
        
        # Otherwise, return the identifier itself
        return identifier
    
    def clean_entity_name(self, name: str) -> str:
        """
        Clean entity name by removing suffixes, prefixes, and noise.
        Handles various formats including titles, affiliations, and grammatical variations.
        
        Args:
            name: Entity name to clean
            
        Returns:
            Cleaned entity name
        """
        if not name:
            return name
        
        original = name
        
        # Remove common suffixes (e.g., "Name - Title", "Name | Organization")
        name = re.sub(r'\s*[-–—|]\s*.*$', '', name)
        
        # Remove parenthetical info (e.g., "John Smith (CEO)")
        name = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Remove bracketed info (e.g., "John Smith [expert]")
        name = re.sub(r'\s*\[[^\]]*\]', '', name)
        
        # Remove titles and honorifics (case-insensitive)
        # Common English titles
        titles = r'\b(mr|mrs|ms|miss|dr|prof|professor|sir|lord|lady|rev|father|sister|brother)\b\.?\s*'
        name = re.sub(titles, '', name, flags=re.IGNORECASE)
        
        # Remove academic/professional suffixes
        suffixes = r',?\s*\b(phd|md|esq|jr|sr|ii|iii|iv|ceo|cto|cfo|president|director)\b\.?'
        name = re.sub(suffixes, '', name, flags=re.IGNORECASE)
        
        # Remove possessive forms (e.g., "Smith's" -> "Smith")
        name = re.sub(r"'s\b", '', name)
        
        # Remove quotes around names
        name = re.sub(r'^["\']|["\']$', '', name)
        
        # Remove leading/trailing punctuation
        name = name.strip('.,;:!?')
        
        # Normalize multiple spaces to single space
        name = ' '.join(name.split())
        
        # If cleaning resulted in empty string, return original
        name = name.strip()
        if not name:
            logger.debug(f"Cleaning resulted in empty string for '{original}', returning original")
            return original.strip()
        
        return name
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize both names
        norm1 = self.normalize_text(name1)
        norm2 = self.normalize_text(name2)
        
        # Clean both names
        clean1 = self.clean_entity_name(norm1)
        clean2 = self.clean_entity_name(norm2)
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        
        # Boost similarity if one is substring of other (handles variations like "John Smith" vs "Dr. John Smith")
        if clean1 in clean2 or clean2 in clean1:
            similarity = max(similarity, 0.9)
        
        # Boost similarity if all words in shorter name appear in longer name
        words1 = set(clean1.split())
        words2 = set(clean2.split())
        if words1 and words2:
            if words1.issubset(words2) or words2.issubset(words1):
                similarity = max(similarity, 0.88)
        
        return similarity
    
    def add_alias(self, canonical: str, alias: str):
        """
        Manually add an alias for a canonical entity.
        
        Args:
            canonical: Canonical entity name
            alias: Alias to add
        """
        # Initialize if needed
        if canonical not in self.entity_aliases:
            self.entity_aliases[canonical] = {canonical}
            self.stats['canonical_entities'] += 1
        
        # Add alias
        if alias not in self.entity_aliases[canonical]:
            self.entity_aliases[canonical].add(alias)
            self.alias_to_canonical[alias] = canonical
            self.alias_to_canonical[canonical] = canonical  # Map canonical to itself
            self.stats['aliases_merged'] += 1
            logger.debug(f"Added alias: '{alias}' -> '{canonical}'")
    
    def canonicalize(self, entity: str, qid: Optional[str] = None) -> str:
        """
        Get the canonical form of an entity name (QID if available, otherwise text-based).
        
        Args:
            entity: Entity name (possibly an alias)
            qid: Optional Wikidata QID for QID-based disambiguation
            
        Returns:
            Canonical identifier (QID or entity name)
        """
        self.stats['total_entities'] += 1
        
        # PRIORITY 1: Use QID if provided
        if qid:
            self.stats['qid_based_matches'] += 1
            # Register this text as an alias for the QID
            if entity not in self.text_to_qid:
                self.text_to_qid[entity] = qid
                if qid in self.qid_aliases:
                    self.qid_aliases[qid].add(entity)
            return qid
        
        # PRIORITY 2: Check if we have a QID for this entity text
        if entity in self.text_to_qid:
            self.stats['qid_based_matches'] += 1
            return self.text_to_qid[entity]
        
        # PRIORITY 3: Fallback to text-based disambiguation
        self.stats['text_based_matches'] += 1
        
        # Fast lookup if we've seen this exact name before
        if entity in self.alias_to_canonical:
            return self.alias_to_canonical[entity]
        
        # Clean the entity name
        cleaned = self.clean_entity_name(entity)
        
        # Check cleaned version
        if cleaned in self.alias_to_canonical:
            # Remember this mapping for future
            self.alias_to_canonical[entity] = self.alias_to_canonical[cleaned]
            return self.alias_to_canonical[cleaned]
        
        # Auto-discover aliases if enabled
        if self.enable_auto_discovery:
            # Check if similar to any known canonical entity
            best_match = None
            best_similarity = 0.0
            
            for canonical in self.entity_aliases.keys():
                similarity = self.calculate_similarity(entity, canonical)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = canonical
            
            if best_match:
                # Found a match! Add as alias
                self.add_alias(best_match, entity)
                self.stats['auto_discovered'] += 1
                logger.info(f"Auto-discovered alias (similarity={best_similarity:.2f}): '{entity}' -> '{best_match}'")
                return best_match
        
        # No match found - this entity becomes its own canonical form
        if entity not in self.entity_aliases:
            self.entity_aliases[entity] = {entity}
            self.alias_to_canonical[entity] = entity
            self.stats['canonical_entities'] += 1
        
        return entity
    
    def merge_entities(self, entity1: str, entity2: str) -> str:
        """
        Merge two entities, treating them as aliases of each other.
        Chooses the longer/more complete name as canonical.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            Canonical entity name (the one chosen)
        """
        # Choose the longer name as canonical (usually more complete)
        if len(entity1) >= len(entity2):
            canonical = entity1
            alias = entity2
        else:
            canonical = entity2
            alias = entity1
        
        # If either is already canonical, use that
        if entity1 in self.entity_aliases and entity2 not in self.entity_aliases:
            canonical = entity1
            alias = entity2
        elif entity2 in self.entity_aliases and entity1 not in self.entity_aliases:
            canonical = entity2
            alias = entity1
        
        self.add_alias(canonical, alias)
        return canonical
    
    def get_all_aliases(self, entity: str, qid: Optional[str] = None) -> Set[str]:
        """
        Get all aliases for an entity (including canonical form).
        
        Args:
            entity: Entity name (canonical or alias)
            qid: Optional Wikidata QID
            
        Returns:
            Set of all aliases including canonical form
        """
        # If QID provided or known, return QID-based aliases
        if qid:
            return self.qid_aliases.get(qid, {entity})
        
        if entity in self.text_to_qid:
            qid = self.text_to_qid[entity]
            return self.qid_aliases.get(qid, {entity})
        
        # Fallback to text-based aliases
        canonical = self.canonicalize(entity)
        return self.entity_aliases.get(canonical, {entity})
    
    def load_predefined_aliases(self, aliases: Dict[str, List[str]]):
        """
        Load predefined aliases from a dictionary.
        
        Args:
            aliases: Dictionary mapping canonical names to lists of aliases
                     Example: {"Lia Olguța Vasilescu": ["Olguța Vasilescu", "Olguta Vasilescu"]}
        """
        for canonical, alias_list in aliases.items():
            for alias in alias_list:
                self.add_alias(canonical, alias)
        
        logger.info(f"Loaded {len(aliases)} predefined entities with {sum(len(v) for v in aliases.values())} total aliases")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get disambiguation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            'total_aliases': sum(len(aliases) for aliases in self.entity_aliases.values()),
            'unique_entities': len(self.entity_aliases),
            'qid_entities': len(self.qid_to_canonical),
            'qid_aliases': sum(len(aliases) for aliases in self.qid_aliases.values())
        }
    
    def export_aliases(self) -> Dict[str, List[str]]:
        """
        Export all discovered aliases for saving/inspection.
        
        Returns:
            Dictionary mapping canonical names to alias lists
        """
        return {
            canonical: sorted(list(aliases))
            for canonical, aliases in self.entity_aliases.items()
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"EntityDisambiguator(entities={stats['unique_entities']}, aliases={stats['total_aliases']}, auto_discovered={stats['auto_discovered']})"


# Singleton instance for convenience
_default_disambiguator: Optional[EntityDisambiguator] = None


def get_disambiguator(reset: bool = False, **kwargs) -> EntityDisambiguator:
    """
    Get or create the default entity disambiguator instance.
    
    Args:
        reset: If True, creates a new instance
        **kwargs: Arguments to pass to EntityDisambiguator constructor
        
    Returns:
        EntityDisambiguator instance
    """
    global _default_disambiguator
    
    if reset or _default_disambiguator is None:
        _default_disambiguator = EntityDisambiguator(**kwargs)
    
    return _default_disambiguator
