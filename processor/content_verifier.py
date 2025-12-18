"""
Content verification module for validating sentence-to-source URL mappings.

Ensures extracted relations actually come from their attributed sources.
"""
import re
import hashlib
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import diskcache
try:
    from ..utils.logger import setup_logger
except:
    from utils.logger import setup_logger
logger = setup_logger(__name__)


class ContentVerifier:
    """
    Verifies that extracted sentences actually exist in their source pages.
    Uses cached page content for validation.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize content verifier.
        
        Args:
            cache_dir: Directory containing cached page content
        """
        self.cache_dir = Path(cache_dir)
        self.cache = None
        
        if self.cache_dir.exists():
            try:
                self.cache = diskcache.Cache(str(self.cache_dir))
                logger.info(f"Content verifier initialized with cache at {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize content verifier cache: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL (same as crawler)."""
        return f"page:{hashlib.sha256(url.encode()).hexdigest()}"
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (remove extra whitespace, lowercase).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Lowercase for case-insensitive comparison
        text = text.lower()
        # Remove common punctuation differences
        text = text.replace('"', '').replace('"', '').replace('"', '')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()
    
    def _get_cached_content(self, url: str) -> Optional[str]:
        """
        Retrieve cached page content for a URL.
        
        Args:
            url: Source URL
            
        Returns:
            Cached content or None
        """
        if not self.cache:
            return None
        
        try:
            key = self._get_cache_key(url)
            content = self.cache.get(key)
            if content:
                return str(content)
        except Exception as e:
            logger.debug(f"Failed to retrieve cached content for {url}: {e}")
        
        return None
    
    def verify_sentence_in_source(
        self, 
        sentence: str, 
        source_url: str,
        min_match_length: int = 40
    ) -> Tuple[bool, float]:
        """
        Verify that a sentence exists in the cached source page.
        
        Args:
            sentence: Sentence text to verify
            source_url: URL of the source page
            min_match_length: Minimum character length for matching substring
            
        Returns:
            Tuple of (is_verified: bool, confidence: float [0-1])
        """
        # Skip verification for invalid URLs
        if not source_url or source_url.lower() == 'unknown':
            return False, 0.0
        
        if not source_url.startswith('http://') and not source_url.startswith('https://'):
            return False, 0.0
        
        # Get cached content
        cached_content = self._get_cached_content(source_url)
        if not cached_content:
            logger.debug(f"No cached content for verification: {source_url}")
            return False, 0.0  # Cannot verify - no cache
        
        # Normalize both texts
        normalized_sentence = self._normalize_text(sentence)
        normalized_content = self._normalize_text(cached_content)
        
        # Check for various match lengths
        sentence_len = len(normalized_sentence)
        
        # Try full sentence match first
        if normalized_sentence in normalized_content:
            logger.debug(f"✓ Full sentence verified in source: {source_url}")
            return True, 1.0
        
        # Try matching increasingly shorter substrings (handles slight edits)
        for match_len in [min(sentence_len, 100), min(sentence_len, 80), min_match_length]:
            if match_len > sentence_len:
                continue
            
            # Try beginning, middle, and end of sentence
            substrings = [
                normalized_sentence[:match_len],  # Beginning
                normalized_sentence[-match_len:],  # End
            ]
            
            if sentence_len > match_len * 2:
                # Add middle chunk for longer sentences
                mid_start = (sentence_len - match_len) // 2
                substrings.append(normalized_sentence[mid_start:mid_start + match_len])
            
            for substring in substrings:
                if len(substring) >= min_match_length and substring in normalized_content:
                    confidence = min(1.0, match_len / sentence_len)
                    logger.debug(f"✓ Partial sentence verified ({confidence:.1%} match) in: {source_url}")
                    return True, confidence
        
        logger.debug(f"✗ Sentence NOT found in source: {source_url}")
        return False, 0.0
    
    def detect_wikipedia_content(self, sentence: str) -> bool:
        """
        Detect if a sentence looks like Wikipedia content.
        
        Args:
            sentence: Sentence to check
            
        Returns:
            True if sentence appears to be from Wikipedia
        """
        # Common Wikipedia patterns
        wikipedia_patterns = [
            r'Reference This article used a translation',
            r'↑\s+[A-Z]+,\s+[A-Z]',  # Citation format: ↑ NAME, Initial
            r'\(\s*Romanian\s*\)\s*-\s*↑',  # (Romanian) - ↑
            r'Jump to:\s*a\s+b\s+c',  # Jump to: a b c
            r'^\s*\(\s*[A-Z]{2}\s*\)',  # Language codes like (RO), (EN)
        ]
        
        for pattern in wikipedia_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                logger.debug(f"Wikipedia content detected: {pattern}")
                return True
        
        return False
    
    def find_original_wikipedia_url(self, sentence: str, entity_name: str) -> Optional[str]:
        """
        Attempt to find the original Wikipedia URL for content.
        
        Args:
            sentence: Sentence that may be from Wikipedia
            entity_name: Name of the entity (for constructing Wikipedia URL)
            
        Returns:
            Wikipedia URL or None
        """
        if not self.detect_wikipedia_content(sentence):
            return None
        
        # Extract language code if present
        lang_match = re.search(r'\(\s*([A-Z]{2})\s*\)', sentence)
        lang_code = lang_match.group(1).lower() if lang_match else 'en'
        
        # Construct Wikipedia URL
        # Replace spaces with underscores for Wikipedia URL format
        wiki_title = entity_name.replace(' ', '_')
        wikipedia_url = f"https://{lang_code}.wikipedia.org/wiki/{wiki_title}"
        
        logger.debug(f"Suggested Wikipedia URL: {wikipedia_url}")
        return wikipedia_url
    
    def split_multi_source_content(
        self, 
        sentences: List[str], 
        source_url: str
    ) -> Dict[str, List[str]]:
        """
        Split sentences into original vs copied (Wikipedia) content.
        
        Args:
            sentences: List of extracted sentences from a page
            source_url: URL of the source page
            
        Returns:
            Dictionary with 'original' and 'wikipedia' lists
        """
        result = {
            'original': [],
            'wikipedia': [],
            'unverified': []
        }
        
        for sentence in sentences:
            # Check if sentence is Wikipedia content
            if self.detect_wikipedia_content(sentence):
                result['wikipedia'].append(sentence)
                continue
            
            # Verify if sentence exists in source
            is_verified, confidence = self.verify_sentence_in_source(sentence, source_url)
            
            if is_verified and confidence > 0.6:
                result['original'].append(sentence)
            else:
                result['unverified'].append(sentence)
        
        logger.debug(
            f"Content split: {len(result['original'])} original, "
            f"{len(result['wikipedia'])} Wikipedia, "
            f"{len(result['unverified'])} unverified"
        )
        
        return result
