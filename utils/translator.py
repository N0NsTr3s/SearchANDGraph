"""
Translation module for converting text to English for better entity detection.
"""
import translators as ts
from typing import Optional, List
from pathlib import Path
import diskcache
from config import CacheConfig
from logger import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from langdetect import detect, LangDetectException
from unidecode import unidecode
from cache_manager import CacheManager

logger = setup_logger(__name__)


class TextTranslator:
    """Handles text translation using the translators library."""
    
    # Maximum characters per translation request to avoid API limits
    MAX_CHUNK_SIZE = 2500  # Reduced for better reliability (was 4000)
    
    def __init__(self, target_language: str = "en", provider: str = "Deepl", cache_config: Optional[CacheConfig] = None):
        """
        Initialize the translator.
        
        Args:
            target_language: Target language code (default: "en" for English)
            provider: Translation provider (google, bing, baidu, etc.)
            cache_config: Optional cache configuration for persistent caching
        """
        self.target_language = target_language
        self.provider = provider
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize SQLite cache manager
        if self.cache_config.enabled:
            cache_dir = Path(self.cache_config.cache_dir)
            self._cache_manager = CacheManager(cache_dir=str(cache_dir), db_name="translation_cache.db")
            logger.info(f"Translation cache enabled (SQLite) at {cache_dir / 'translation_cache.db'}")
        else:
            self._cache_manager = None
            logger.info("Translation cache disabled")
        
        # Keep legacy diskcache for backwards compatibility (can remove later)
        self._cache = {}  # Fallback in-memory cache
        
        # Statistics
        self.total_translations = 0
        self.failed_translations = 0
        
        logger.info(f"Initialized translator: provider={provider}, target={target_language}")
    
    def _split_text_into_chunks(self, text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """
        Split text into smaller chunks for translation.
        Intelligently splits at paragraph boundaries to preserve context and meaning.
        Falls back to sentence boundaries if paragraphs are too large.
        
        Args:
            text: Text to split
            max_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        
        # First try to split by paragraphs (double newline or single newline with substantial break)
        paragraphs = []
        current_paragraph = ""
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Empty line indicates paragraph break
            if not line_stripped:
                if current_paragraph.strip():
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
            else:
                # Add line to current paragraph
                if current_paragraph:
                    current_paragraph += ' ' + line_stripped
                else:
                    current_paragraph = line_stripped
        
        # Add last paragraph
        if current_paragraph.strip():
            paragraphs.append(current_paragraph.strip())
        
        # If no paragraphs found (text has no newlines), fall back to sentence splitting
        if len(paragraphs) <= 1:
            logger.debug("No paragraph breaks found, splitting by sentences")
            return self._split_by_sentences(text, max_size)
        
        logger.debug(f"Found {len(paragraphs)} paragraphs in text")
        
        # Group paragraphs into chunks
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If a single paragraph is too large, split it by sentences
            if len(paragraph) > max_size:
                logger.debug(f"Paragraph too large ({len(paragraph)} chars), splitting by sentences")
                
                # Save current chunk if exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split the large paragraph and add chunks
                para_chunks = self._split_by_sentences(paragraph, max_size)
                chunks.extend(para_chunks)
                
            # Check if adding this paragraph would exceed max_size
            elif len(current_chunk) + len(paragraph) + 2 <= max_size:  # +2 for paragraph separator
                if current_chunk:
                    current_chunk += '\n\n' + paragraph  # Preserve paragraph breaks
                else:
                    current_chunk = paragraph
            else:
                # Current chunk is full, save it and start new chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text of {len(text)} chars into {len(chunks)} chunks at paragraph boundaries")
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int) -> List[str]:
        """
        Split text by sentences when paragraph splitting isn't suitable.
        
        Args:
            text: Text to split
            max_size: Maximum chunk size
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        # Split by sentences (look for sentence endings followed by space and capital letter, or newline)
        import re
        
        # More sophisticated sentence splitting
        sentence_endings = re.compile(r'([.!?]+[\s\n]+(?=[A-Z])|[.!?]+$)')
        sentences = sentence_endings.split(text)
        
        # Reconstruct sentences (split() separates the delimiters)
        reconstructed_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed_sentences.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed_sentences.append(sentences[i])
        
        # If last element wasn't paired, add it
        if len(sentences) % 2 == 1:
            reconstructed_sentences.append(sentences[-1])
        
        # Filter out empty sentences
        sentences = [s.strip() for s in reconstructed_sentences if s.strip()]
        
        # Group sentences into chunks
        for sentence in sentences:
            # If single sentence is too long, split it at word boundaries
            if len(sentence) > max_size:
                logger.debug(f"Sentence too large ({len(sentence)} chars), splitting at word boundaries")
                words = sentence.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 <= max_size:
                        if current_word_chunk:
                            current_word_chunk += ' ' + word
                        else:
                            current_word_chunk = word
                    else:
                        if current_word_chunk:
                            chunks.append(current_word_chunk)
                        current_word_chunk = word
                
                if current_word_chunk:
                    chunks.append(current_word_chunk)
                    
            elif len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"Split into {len(chunks)} chunks by sentences")
        return chunks
    
    def translate(self, text: str, source_language: str = "auto", site: str | None = None) -> Optional[str]:
        """
        Translate text to the target language.
        Automatically splits large texts into chunks.
        
        Args:
            text: Text to translate
            source_language: Source language code (default: "auto" for auto-detection)
            
        Returns:
            Translated text or None if translation fails
        """
        if not text or not text.strip():
            return text
        
        # Check if text is too large and needs chunking
        if len(text) > self.MAX_CHUNK_SIZE:
            logger.info(f"Text is {len(text)} chars, splitting into chunks for translation")
            return self._translate_large_text(text, source_language)
        
        # Check SQLite cache first (include site to keep same ID per site)
        if self._cache_manager:
            cached = self._cache_manager.get_translation(
                text=text,
                source_lang=source_language if source_language != "auto" else None,
                target_lang=self.target_language,
                provider=self.provider,
                site=site
            )
            if cached:
                logger.debug(f"Using cached translation for: {text[:50]}...")
                return cached
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Attempt translation
                translated = ts.translate_text(
                    query_text=text,
                    translator=self.provider,
                    from_language=source_language,
                    to_language=self.target_language
                )
                
                # Ensure we return a string
                if isinstance(translated, str):
                    # Cache the result in SQLite
                        if self._cache_manager:
                            try:
                                    self._cache_manager.set_translation(
                                        text=text,
                                        translated_text=translated,
                                        source_lang=source_language if source_language != "auto" else None,
                                        target_lang=self.target_language,
                                        provider=self.provider,
                                        ttl_days=90,
                                        site=site
                                    )
                            except Exception as e:
                                logger.debug(f"Cache write error: {e}")
                    
                        logger.debug(f"Translated: '{text[:50]}...' -> '{translated[:50]}...'")
                        return translated
                else:
                    logger.warning(f"Translation returned non-string type: {type(translated)}")
                    return text
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                if attempt < max_retries - 1:
                    logger.warning(f"Translation attempt {attempt + 1} failed: {error_msg}. Retrying in {retry_delay}s...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.warning(f"Translation failed after {max_retries} attempts: {error_msg}")
                    logger.debug(f"Failed text preview: {text[:100]}...")
                    # Return original text if all retries fail
                    return text
    
    def _translate_chunk_with_delay(self, chunk: str, source_language: str, chunk_index: int) -> tuple[str, int, bool]:
        """
        Translate a single chunk with staggered delay for rate limiting.
        
        Args:
            chunk: Text chunk to translate
            source_language: Source language
            chunk_index: Index of the chunk (for staggering)
            
        Returns:
            Tuple of (translated_text, chunk_index, success)
        """
        # Stagger requests to avoid rate limiting
        time.sleep(chunk_index * 0.3)  # 300ms delay per chunk
        
        try:
            self.total_translations += 1
            translated = ts.translate_text(
                query_text=chunk,
                translator=self.provider,
                from_language=source_language,
                to_language=self.target_language
            )
            
            if isinstance(translated, str):
                return translated, chunk_index, True
            else:
                logger.warning(f"Chunk {chunk_index} translation returned non-string: {type(translated)}")
                self.failed_translations += 1
                return chunk, chunk_index, False
                
        except Exception as e:
            logger.warning(f"Chunk {chunk_index} translation failed: {e}")
            self.failed_translations += 1
            return chunk, chunk_index, False
    
    def _translate_chunks_parallel(self, chunks: List[str], source_language: str = "auto", max_concurrent: int = 3, site: str | None = None) -> List[str]:
        """
        Translate multiple chunks in parallel with rate limiting.

        Args:
            chunks: List of text chunks to translate
            source_language: Source language code
            max_concurrent: Maximum concurrent translation requests

        Returns:
            List of translated chunks in original order (falls back to original text on failure)
        """

        def _normalize_cached_value(value: object) -> Optional[str]:
            """Normalize cached values that may come back as bytes or tuples."""
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8")
                except Exception:
                    return None
            if isinstance(value, tuple):
                for item in value:
                    normalized = _normalize_cached_value(item)
                    if normalized:
                        return normalized
                return None
            # Last resort: string conversion
            try:
                return str(value)
            except Exception:
                return None

        # Default to original chunks so we always return strings
        results: List[str] = list(chunks)
        cache_hits = 0
        to_translate: List[tuple[int, str]] = []

        # Check cache first for all chunks (prefer SQLite CacheManager when available)
        for i, chunk in enumerate(chunks):
            try:
                if getattr(self, '_cache_manager', None):
                    # Use persistent SQLite cache
                    src_lang = source_language if source_language != "auto" else None
                    cached_chunk = self._cache_manager.get_translation( # type: ignore
                        text=chunk,
                        source_lang=src_lang,
                        target_lang=self.target_language,
                        provider=self.provider,
                        site=site
                    )
                    if cached_chunk:
                        results[i] = cached_chunk
                        cache_hits += 1
                        continue
                    else:
                        to_translate.append((i, chunk))
                else:
                    # Fallback to legacy in-memory/diskcache
                    cache_key = f"{source_language}:{chunk[:100]}"
                    raw_cached = self._cache.get(cache_key) if isinstance(self._cache, diskcache.Cache) else self._cache.get(cache_key)
                    cached_chunk = _normalize_cached_value(raw_cached)
                    if cached_chunk:
                        results[i] = cached_chunk
                        cache_hits += 1
                    else:
                        to_translate.append((i, chunk))
            except Exception as e:
                logger.debug(f"Cache read error for chunk {i}: {e}")
                to_translate.append((i, chunk))

        logger.info(f"Cache hits: {cache_hits}/{len(chunks)}, translating {len(to_translate)} chunks in parallel")

        if not to_translate:
            return results

        # Translate remaining chunks in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(self._translate_chunk_with_delay, chunk, source_language, idx): (idx, chunk)
                for idx, chunk in to_translate
            }

            # Use a timeout on each future's result to prevent a single hang from stopping everything.
            for future in as_completed(futures):
                original_idx, original_chunk = futures[future]
                cache_idx = original_idx
                translated_to_cache = None

                try:
                    # Wait for a maximum of 20 seconds for a single chunk to translate.
                    res = future.result(timeout=20.0)

                    # Defensive handling: ensure res is a tuple of expected shape
                    if isinstance(res, tuple) and len(res) >= 3:
                        translated, idx, success = res[0], res[1], res[2]
                    else:
                        # Unexpected result from worker; treat as failure and fall back
                        logger.warning(f"Unexpected translation worker result for chunk {original_idx}: {res}")
                        translated, idx, success = None, original_idx, False

                    if success and isinstance(translated, str):
                        results[idx] = translated
                        translated_to_cache = translated
                    else:
                        # If the task reported failure or returned non-string, try a synchronous robust translation
                        logger.debug(f"Chunk {original_idx} worker reported failure or non-string. Falling back to sync translate()")
                        try:
                            fallback = self.translate(original_chunk, source_language, site=site)
                            if fallback and isinstance(fallback, str):
                                results[original_idx] = fallback
                                translated_to_cache = fallback
                            else:
                                results[original_idx] = original_chunk
                        except Exception as e:
                            logger.warning(f"Fallback sync translation failed for chunk {original_idx}: {e}")
                            results[original_idx] = original_chunk

                except Exception as e:
                    logger.warning(f"A translation chunk for index {original_idx} timed out or failed: {e}")
                    # On timeout/error, attempt synchronous fallback translation
                    try:
                        fallback = self.translate(original_chunk, source_language, site=site)
                        if fallback and isinstance(fallback, str):
                            results[original_idx] = fallback
                            translated_to_cache = fallback
                        else:
                            results[original_idx] = original_chunk
                    except Exception as e2:
                        logger.warning(f"Fallback sync translation also failed for chunk {original_idx}: {e2}")
                        results[original_idx] = original_chunk

                # Cache successful translations (prefer SQLite CacheManager when available)
                try:
                    if isinstance(translated_to_cache, str) and translated_to_cache != chunks[cache_idx]:
                        if getattr(self, '_cache_manager', None):
                            # Convert nlp_cache_ttl seconds -> days for CacheManager (fallback to 90 days)
                            ttl_sec = getattr(self.cache_config, 'nlp_cache_ttl', None)
                            if ttl_sec:
                                ttl_days = max(1, int(ttl_sec / 86400))
                            else:
                                ttl_days = 90

                            src_lang = source_language if source_language != "auto" else None
                            try:
                                self._cache_manager.set_translation( # type: ignore
                                    text=chunks[cache_idx],
                                    translated_text=translated_to_cache,
                                    source_lang=src_lang,
                                    target_lang=self.target_language,
                                    provider=self.provider,
                                    ttl_days=ttl_days,
                                    site=site
                                )
                            except Exception as e:
                                logger.debug(f"CacheManager write error for chunk {cache_idx}: {e}")
                        else:
                            # Legacy cache write
                            cache_key = f"{source_language}:{chunks[cache_idx][:100]}"
                            if isinstance(self._cache, diskcache.Cache):
                                expire = getattr(self.cache_config, "nlp_cache_ttl", None)
                                if expire is not None:
                                    self._cache.set(cache_key, translated_to_cache, expire=expire)
                                else:
                                    self._cache.set(cache_key, translated_to_cache)
                            else:
                                self._cache[cache_key] = translated_to_cache
                except Exception as e:
                    logger.debug(f"Cache write error for chunk {original_idx}: {e}")

        # Ensure the list contains plain strings
        return [chunk if isinstance(chunk, str) else str(chunk) for chunk in results]
    
    def _translate_large_text(self, text: str, source_language: str = "auto") -> str:
        """
        Translate large text by breaking it into chunks using parallel translation.
        
        Args:
            text: Large text to translate
            source_language: Source language code
            
        Returns:
            Translated text (all chunks combined)
        """
        chunks = self._split_text_into_chunks(text, self.MAX_CHUNK_SIZE)
        
        logger.info(f"Translating {len(chunks)} chunks in parallel...")
        
        # Use parallel translation for better performance
        translated_chunks = self._translate_chunks_parallel(
            chunks, 
            source_language, 
            max_concurrent=3,  # Limit concurrent requests
            site=None
        )
        
        # Combine translated chunks
        combined_translation = "\n\n".join(translated_chunks)
        
        success_count = sum(1 for chunk in translated_chunks if chunk)
        logger.info(f"Translation complete: {success_count}/{len(chunks)} chunks successful")
        
        return combined_translation
    
    def translate_if_needed(self, text: str, site: str | None = None) -> str:
        """
        Translate text only if it's not already in the target language.
        Uses langdetect to avoid unnecessary translations.
        
        Args:
            text: Text to potentially translate
            
        Returns:
            Translated or original text
        """
        if not text or not text.strip():
            return text
        
        # Use language detection to check if translation is needed
        try:
            # Detect language (requires at least ~20 chars for reliability)
            if len(text) < 20:
                # Too short for reliable detection, try translation (pass site if available)
                return self.translate(text, site=site) or text
            
            detected_lang = detect(text)
            
            # If already in target language, return as-is
            if detected_lang == self.target_language:
                logger.debug(f"Text already in {self.target_language}, skipping translation")
                return text
            
            # Need translation
            logger.debug(f"Detected {detected_lang}, translating to {self.target_language}")
            translated = self.translate(text, source_language=detected_lang, site=site)
            
            if translated and translated != text:
                return translated
            return text
            
        except LangDetectException:
            # Language detection failed (text too short/ambiguous)
            logger.debug("Language detection inconclusive, attempting translation")
            try:
                translated = self.translate(text, site=site)
                if translated and translated != text:
                    return translated
                return text
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                return text
                
        except Exception as e:
            logger.warning(f"Translation check failed: {e}")
            return text
    
    @staticmethod
    def normalize_for_matching(text: str) -> str:
        """
        Normalize text with transliteration for better cross-language entity matching.
        This helps match entities across different scripts and character sets.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized ASCII text
            
        Examples:
            - "București" → "bucuresti"
            - "Ștefan cel Mare" → "stefan cel mare"
            - "Москва" → "moskva"
        """
        if not text:
            return text
        
        # Transliterate to ASCII (Ș → S, ă → a, etc.)
        ascii_text = unidecode(text)
        
        # Lowercase and clean
        normalized = ascii_text.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def clear_cache(self):
        """Clear the translation cache."""
        try:
            if getattr(self, '_cache_manager', None):
                # Clear entire SQLite cache (translation + others)
                self._cache_manager.clear_all() # type: ignore
                logger.info("SQLite translation cache cleared")
            else:
                if isinstance(self._cache, diskcache.Cache):
                    self._cache.clear()
                else:
                    self._cache.clear()
                logger.info("Legacy in-memory/disk translation cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_cache_size(self) -> int:
        """Get the number of cached translations."""
        try:
            if getattr(self, '_cache_manager', None):
                stats = self._cache_manager.get_stats() # type: ignore
                return int(stats.get('translation', {}).get('entries', 0))
            else:
                if isinstance(self._cache, diskcache.Cache):
                    return len(list(self._cache.iterkeys()))  # For disk cache
                else:
                    return len(self._cache)  # For dict
        except Exception as e:
            logger.warning(f"Failed to get cache size: {e}")
            return 0
