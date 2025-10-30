"""
Node name cleaning utilities for the knowledge graph.
Standardizes entity names before adding them to the graph.
"""
import re
import unicodedata
from unidecode import unidecode
from logger import setup_logger

logger = setup_logger(__name__)


def normalize_to_ascii(text: str) -> str:
    """
    Convert all non-English characters to ASCII equivalents using unidecode.
    This provides better transliteration than manual character replacement.
    
    Args:
        text: Text with potentially non-ASCII characters
        
    Returns:
        ASCII-normalized text
        
    Examples:
        "Nicușor Dan" → "Nicusor Dan"
        "François Müller" → "Francois Muller"
        "José López" → "Jose Lopez"
        "Москва" → "Moskva" (transliteration)
    """
    if not text:
        return text
    
    # Use unidecode for comprehensive transliteration
    return unidecode(text)


def remove_leading_the(text: str) -> str:
    """
    Remove "The" from the beginning of entity names.
    
    Args:
        text: Entity name
        
    Returns:
        Name without leading "The"
        
    Examples:
        "The United States" → "United States"
        "The Beatles" → "Beatles"
        "Theory" → "Theory" (not affected)
    """
    if not text:
        return text
    
    # Match "The " at the start (case-insensitive, followed by space)
    pattern = r'^The\s+'
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return cleaned.strip()


def remove_possessive_s(text: str) -> str:
    """
    Remove possessive 's from the end of entity names.
    Also handles unicode apostrophes and backticks.
    
    Args:
        text: Entity name
        
    Returns:
        Name without possessive 's
        
    Examples:
        "John's" → "John"
        "Microsoft's" → "Microsoft"
        "James`s" → "James"
        "James" → "James" (not affected, no apostrophe)
    """
    if not text:
        return text
    
    # Match 's, 's, `s at the end of the string (handles different apostrophe types)
    # Also handle backtick which might be rendered as `s
    pattern = r"['`']s$"
    cleaned = re.sub(pattern, '', text)
    
    return cleaned.strip()


def remove_parenthetical_info(text: str) -> str:
    """
    Remove parenthetical information like dates, disambiguation, or descriptions.
    This helps merge duplicate entities with different suffixes.
    
    Args:
        text: Entity name with potential parentheticals
        
    Returns:
        Name without parentheticals
        
    Examples:
        "John Doe (1950-2020)" → "John Doe"
        "Apple (company)" → "Apple"
        "Paris (France)" → "Paris"
        "Microsoft Corporation (founded 1975)" → "Microsoft Corporation"
        "Victor Ponta (born 1972)" → "Victor Ponta"
        "The Beatles (band)" → "The Beatles"
    """
    if not text:
        return text
    
    # Remove anything in parentheses, including:
    # - Birth/death dates: (1950-2020), (b. 1972), (born 1972), (d. 2020)
    # - Disambiguation: (company), (band), (politician)
    # - Locations: (France), (London)
    # - Other info: (founded 1975)
    pattern = r'\s*\([^)]*\)\s*'
    cleaned = re.sub(pattern, ' ', text)
    
    # Also remove square brackets (sometimes used for dates)
    pattern = r'\s*\[[^\]]*\]\s*'
    cleaned = re.sub(pattern, ' ', cleaned)
    
    # Clean up any double spaces created by removal
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()


INVALID_WORDS_EN = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    'about', 'as', 'into', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'who', 'what', 'where', 'when', 'why', 'how', 'which', 'that', 'this',
    'read', 'edit', 'view', 'history', 'see', 'also', 'external', 'links'
}


def is_valid_entity_text(text: str, nlp) -> bool:
    """Performs multi-stage checks on translated text to filter junk.

    This function expects English (or translated-to-English) text so POS tags
    from the English spaCy model are meaningful. If `nlp` is None the POS
    checks are skipped and heuristics are used instead.
    """
    if not text or not text.strip() or len(text.strip()) < 2:
        return False

    text_lower = text.lower().strip()
    words = text_lower.split()

    # Rule 1: Reject if composed entirely of stopwords / invalid single words
    if all(word in INVALID_WORDS_EN for word in words):
        logger.debug(f"Filtering entity composed of stopwords: '{text}'")
        return False

    # Rule 2: Reject if it looks like a sentence fragment
    if len(words) > 8 or any(p in text for p in ['?', '!', '...']):
        logger.debug(f"Filtering sentence-like entity: '{text}'")
        return False

    # Rule 3: POS-based checks (prefer robust English POS tagging on translated text)
    try:
        if nlp is not None:
            doc = nlp(text)
            # Reject if any verb appears in the span
            # === NEW: Stricter POS Checks ===
            # 1. If the syntactic root of the span is a verb, it's likely not a proper noun
            try:
                if hasattr(doc, 'root') and doc.root.pos_ == 'VERB':
                    logger.debug(f"Filtering entity whose root is a verb: '{text}'")
                    return False
            except Exception:
                # be defensive — if root access fails, continue with other checks
                pass

            # 2. Reject if any token is a VERB (strong indicator of a non-entity)
            if any(token.pos_ == 'VERB' for token in doc):
                logger.debug(f"Filtering entity containing a verb: '{text}'")
                return False

            # 3. Reject if single-token numeric/date-like labels
            if len(doc) == 1 and doc[0].pos_ in ['NUM', 'DATE']:
                logger.debug(f"Filtering mislabeled numeric/date entity: '{text}'")
                return False

            # 4. Reject if last token is an adposition/conjunction/pronoun/aux
            if doc and doc[-1].pos_ in ['ADP', 'CCONJ', 'SCONJ', 'PRON', 'AUX']:
                logger.debug(f"Filtering entity ending with invalid POS '{doc[-1].pos_}': '{text}'")
                return False
    except Exception as e:
        logger.debug(f"POS-check failed for '{text}', allowing it. Error: {e}")

    # Fallback heuristics if no nlp or POS checks passed
    # Reject strings that are obviously too short or purely punctuation
    if len(text_lower) <= 1:
        return False
    # If the string is purely punctuation/artifacts, reject it
    if all(ch in ".,;:-_\\/|()[]{}\"'" for ch in text_lower):
        return False

    return True


def is_valid_entity_name(name: str) -> bool:
    """Backward-compatible wrapper for older callers that don't pass a spaCy nlp object."""
    return is_valid_entity_text(name, None)


def clean_node_name(name: str) -> str | None:
    """
    Apply all cleaning operations to a node name.
    Returns None if the name is invalid (preposition, article, etc.).
    
    Operations performed (in order):
    1. Convert non-English characters to ASCII
    2. Remove leading "The"
    3. Remove possessive 's
    4. Remove parenthetical info (dates, disambiguation)
    5. Normalize whitespace
    6. Trim
    7. Validate (check if it's a valid entity name)
    
    Args:
        name: Original entity name
        
    Returns:
        Cleaned entity name, or None if invalid
        
    Examples:
        "The Nicușor's" → "Nicusor"
        "François's Car" → "Francois Car"
        "The   United   States" → "United States"
        "John Doe (1950-2020)" → "John Doe"
        "Apple (company)" → "Apple"
        "of" → None (invalid)
        "the" → None (invalid)
    """
    if not name:
        return None
    
    original_name = name
    
    # Step 1: Convert to ASCII
    name = normalize_to_ascii(name)
    
    # Step 2: Remove leading "The"
    name = remove_leading_the(name)
    
    # Step 3: Remove possessive 's
    name = remove_possessive_s(name)
    
    # Step 4: Remove parenthetical info (dates, disambiguation, etc.)
    name = remove_parenthetical_info(name)

    # Step 4.5: Add more aggressive artifact removal
    # Removes patterns like "' Read", "[edit", "wikitext]", etc.
    try:
        # Removes trailing patterns like "' Read" or "’ Read" or "` Read"
        name = re.sub(r"\s*['`´’]\s*\w+$", '', name)
        # Removes trailing bracket artifacts like "[edit" or "[citation"
        name = re.sub(r'\[\w+$', '', name)
        # Removes leading bracket artifacts like "wikitext]" or "cite]"
        name = re.sub(r'^\w+\]', '', name)
    except Exception:
        # Be defensive: if regex fails for any reason, continue with existing name
        pass

    # Step 5: Normalize whitespace (collapse multiple spaces)
    name = ' '.join(name.split())
    
    # Step 6: Trim
    name = name.strip()
    
    # Step 7: Validate
    if not is_valid_entity_name(name):
        logger.debug(f"Filtered out invalid entity: '{original_name}'")
        return None
    
    # Log if name was changed
    if name != original_name:
        logger.debug(f"Cleaned node name: '{original_name}' → '{name}'")
    
    return name


def get_canonical_name_mapping(names: list[str]) -> dict[str, str | None]:
    """
    Create a mapping from original names to cleaned canonical names.
    Also identifies duplicate names that should be merged.
    Filters out invalid names (returns None for those).
    
    Args:
        names: List of entity names
        
    Returns:
        Dictionary mapping original name → canonical cleaned name (or None if invalid)
        
    Example:
        Input: ["The Microsoft", "Microsoft's", "Nicușor Dan", "Nicusor Dan", "of"]
        Output: {
            "The Microsoft": "Microsoft",
            "Microsoft's": "Microsoft",
            "Nicușor Dan": "Nicusor Dan",
            "Nicusor Dan": "Nicusor Dan",
            "of": None
        }
    """
    mapping = {}
    canonical_to_originals = {}  # Track which originals map to same canonical
    
    for name in names:
        cleaned = clean_node_name(name)
        mapping[name] = cleaned
        
        # Track duplicates (only for valid names)
        if cleaned is not None:
            if cleaned not in canonical_to_originals:
                canonical_to_originals[cleaned] = []
            canonical_to_originals[cleaned].append(name)
    
    # Log duplicate groups
    duplicates_found = 0
    for canonical, originals in canonical_to_originals.items():
        if len(originals) > 1:
            duplicates_found += len(originals) - 1
            logger.info(f"Merging {len(originals)} variants into '{canonical}':")
            for orig in originals:
                if orig != canonical:
                    logger.info(f"  '{orig}' → '{canonical}'")
    
    if duplicates_found > 0:
        logger.info(f"Total: {duplicates_found} duplicate nodes will be merged")
    
    return mapping


def clean_entity_dict(entities: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Clean all entity names in a dictionary and create a mapping.
    Filters out invalid entities (prepositions, articles, etc.).
    
    Args:
        entities: Dictionary mapping entity name → label
        
    Returns:
        Tuple of:
        - Cleaned entities dictionary (canonical name → label)
        - Name mapping (original name → canonical name)
        
    Example:
        Input: {"The Microsoft": "ORG", "Microsoft's": "ORG", "of": "O"}
        Output: (
            {"Microsoft": "ORG"},
            {"The Microsoft": "Microsoft", "Microsoft's": "Microsoft"}
        )
        Note: "of" is filtered out
    """
    cleaned_entities = {}
    name_mapping = {}
    filtered_count = 0
    
    for original_name, label in entities.items():
        canonical_name = clean_node_name(original_name)
        
        # Skip invalid entities (None return means invalid)
        if canonical_name is None:
            filtered_count += 1
            logger.debug(f"Filtered out invalid entity: '{original_name}'")
            continue
        
        name_mapping[original_name] = canonical_name
        
        # If multiple entities map to same canonical name, keep the label
        # (they should have the same label anyway)
        cleaned_entities[canonical_name] = label
    
    # Log summary
    original_count = len(entities)
    cleaned_count = len(cleaned_entities)
    merged_count = original_count - cleaned_count - filtered_count
    
    if filtered_count > 0:
        logger.info(f"Entity cleaning: filtered out {filtered_count} invalid entities")
    if merged_count > 0:
        logger.info(f"Entity cleaning: {original_count} entities → {cleaned_count} unique ({merged_count} merged)")
    
    return cleaned_entities, name_mapping
