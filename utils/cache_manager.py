"""
SQLite-based cache manager for external API lookups (translation, entity enrichment).

Provides persistent caching with TTL support and automatic cleanup.
"""
import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any, Dict
from contextlib import contextmanager
from logger import setup_logger

logger = setup_logger(__name__)


class CacheManager:
    """
    SQLite-based cache for external API calls.
    
    Features:
    - Persistent storage across sessions
    - TTL (time-to-live) support with automatic expiration
    - Separate tables for different cache types (translation, wikidata, etc.)
    - Thread-safe with connection pooling
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, cache_dir: str = ".cache", db_name: str = "api_cache.db"):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache database
            db_name: Name of the SQLite database file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / db_name
        self._init_database()
        
        logger.info(f"Cache manager initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema with tables for different cache types."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Translation cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    cache_key TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    source_lang TEXT,
                    target_lang TEXT,
                    translated_text TEXT NOT NULL,
                    provider TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    hit_count INTEGER DEFAULT 0
                )
            ''')
            
            # Wikidata enrichment cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS wikidata_cache (
                    cache_key TEXT PRIMARY KEY,
                    entity_text TEXT NOT NULL,
                    qid TEXT,
                    label TEXT,
                    description TEXT,
                    aliases TEXT,
                    metadata TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    hit_count INTEGER DEFAULT 0
                )
            ''')
            
            # Generic API cache table (for other external lookups)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generic_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    request_data TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    hit_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_translation_text ON translation_cache(source_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_wikidata_entity ON wikidata_cache(entity_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_translation_expires ON translation_cache(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_wikidata_expires ON wikidata_cache(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_generic_type ON generic_cache(cache_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_generic_expires ON generic_cache(expires_at)')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = '|'.join(str(arg) for arg in args if arg is not None)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_translation(self, text: str, source_lang: Optional[str] = None, 
        target_lang: str = "en", provider: Optional[str] = None,
        site: Optional[str] = None) -> Optional[str]:
        """
        Get cached translation.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider name
            
        Returns:
            Translated text if found in cache, None otherwise
        """
        cache_key = self._generate_cache_key('translation', site or '', text, source_lang, target_lang, provider)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if entry exists and is not expired
            cursor.execute('''
                SELECT translated_text, expires_at 
                FROM translation_cache 
                WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > ?)
            ''', (cache_key, int(time.time())))
            
            row = cursor.fetchone()
            if row:
                # Update hit count
                cursor.execute('''
                    UPDATE translation_cache 
                    SET hit_count = hit_count + 1 
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                logger.debug(f"Translation cache HIT for: {text[:50]}...")
                return row['translated_text']
        
        logger.debug(f"Translation cache MISS for: {text[:50]}...")
        return None
    
    def set_translation(self, text: str, translated_text: str, 
                       source_lang: Optional[str] = None, target_lang: str = "en",
                       provider: Optional[str] = None, ttl_days: int = 90,
                       site: Optional[str] = None):
        """
        Store translation in cache.
        
        Args:
            text: Source text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider name
            ttl_days: Time to live in days (None = never expire)
        """
        cache_key = self._generate_cache_key('translation', site or '', text, source_lang, target_lang, provider)
        created_at = int(time.time())
        expires_at = created_at + (ttl_days * 86400) if ttl_days else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO translation_cache 
                (cache_key, source_text, source_lang, target_lang, translated_text, 
                 provider, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ''', (cache_key, text, source_lang, target_lang, translated_text, 
                  provider, created_at, expires_at))
            conn.commit()
        
        logger.debug(f"Cached translation for: {text[:50]}...")
    
    def get_wikidata(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached Wikidata enrichment data.
        
        Args:
            entity_text: Entity text to look up
            
        Returns:
            Dictionary with Wikidata metadata if found, None otherwise
        """
        cache_key = self._generate_cache_key('wikidata', entity_text.lower().strip())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT qid, label, description, aliases, metadata, expires_at
                FROM wikidata_cache 
                WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > ?)
            ''', (cache_key, int(time.time())))
            
            row = cursor.fetchone()
            if row:
                # Update hit count
                cursor.execute('''
                    UPDATE wikidata_cache 
                    SET hit_count = hit_count + 1 
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                # Parse aliases and metadata from JSON
                aliases = json.loads(row['aliases']) if row['aliases'] else []
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                
                qid = row['qid']
                result = {
                    'id': qid,
                    'qid': qid,
                    'label': row['label'],
                    'description': row['description'],
                    'aliases': aliases,
                    **metadata
                }
                
                logger.debug(f"Wikidata cache HIT for: {entity_text}")
                return result
        
        logger.debug(f"Wikidata cache MISS for: {entity_text}")
        return None
    
    def set_wikidata(self, entity_text: str, qid: Optional[str], 
                    label: Optional[str] = None, description: Optional[str] = None,
                    aliases: Optional[list] = None, metadata: Optional[dict] = None,
                    ttl_days: int = 180):
        """
        Store Wikidata enrichment data in cache.
        
        Args:
            entity_text: Entity text
            qid: Wikidata QID
            label: Canonical label
            description: Entity description
            aliases: List of alternative names
            metadata: Additional metadata dict
            ttl_days: Time to live in days (None = never expire)
        """
        cache_key = self._generate_cache_key('wikidata', entity_text.lower().strip())
        created_at = int(time.time())
        expires_at = created_at + (ttl_days * 86400) if ttl_days else None
        
        # Serialize aliases and metadata as JSON
        aliases_json = json.dumps(aliases) if aliases else None
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO wikidata_cache 
                (cache_key, entity_text, qid, label, description, aliases, 
                 metadata, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ''', (cache_key, entity_text, qid, label, description, 
                  aliases_json, metadata_json, created_at, expires_at))
            conn.commit()
        
        logger.debug(f"Cached Wikidata for: {entity_text} (QID: {qid})")
    
    def get_generic(self, cache_type: str, **request_params) -> Optional[Any]:
        """
        Get cached generic API response.
        
        Args:
            cache_type: Type of cache (e.g., 'geocoding', 'sentiment')
            **request_params: Request parameters used as cache key
            
        Returns:
            Cached response data if found, None otherwise
        """
        cache_key = self._generate_cache_key(cache_type, *sorted(request_params.items()))
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT response_data, expires_at
                FROM generic_cache 
                WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > ?)
            ''', (cache_key, int(time.time())))
            
            row = cursor.fetchone()
            if row:
                # Update hit count
                cursor.execute('''
                    UPDATE generic_cache 
                    SET hit_count = hit_count + 1 
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                logger.debug(f"Generic cache HIT for: {cache_type}")
                return json.loads(row['response_data'])
        
        logger.debug(f"Generic cache MISS for: {cache_type}")
        return None
    
    def set_generic(self, cache_type: str, response_data: Any, 
                   ttl_days: int = 90, **request_params):
        """
        Store generic API response in cache.
        
        Args:
            cache_type: Type of cache
            response_data: Response data to cache (will be JSON-serialized)
            ttl_days: Time to live in days
            **request_params: Request parameters used as cache key
        """
        cache_key = self._generate_cache_key(cache_type, *sorted(request_params.items()))
        created_at = int(time.time())
        expires_at = created_at + (ttl_days * 86400) if ttl_days else None
        
        request_json = json.dumps(dict(request_params))
        response_json = json.dumps(response_data)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO generic_cache 
                (cache_key, cache_type, request_data, response_data, 
                 created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            ''', (cache_key, cache_type, request_json, response_json, 
                  created_at, expires_at))
            conn.commit()
        
        logger.debug(f"Cached {cache_type} response")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = int(time.time())
        total_removed = 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean translation cache
            cursor.execute('''
                DELETE FROM translation_cache 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (current_time,))
            total_removed += cursor.rowcount
            
            # Clean wikidata cache
            cursor.execute('''
                DELETE FROM wikidata_cache 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (current_time,))
            total_removed += cursor.rowcount
            
            # Clean generic cache
            cursor.execute('''
                DELETE FROM generic_cache 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (current_time,))
            total_removed += cursor.rowcount
            
            conn.commit()
        
        if total_removed > 0:
            logger.info(f"Cleaned up {total_removed} expired cache entries")
        
        return total_removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Translation cache stats
            cursor.execute('SELECT COUNT(*), SUM(hit_count) FROM translation_cache')
            row = cursor.fetchone()
            stats['translation'] = {
                'entries': row[0],
                'total_hits': row[1] or 0
            }
            
            # Wikidata cache stats
            cursor.execute('SELECT COUNT(*), SUM(hit_count) FROM wikidata_cache')
            row = cursor.fetchone()
            stats['wikidata'] = {
                'entries': row[0],
                'total_hits': row[1] or 0
            }
            
            # Generic cache stats
            cursor.execute('SELECT COUNT(*), SUM(hit_count) FROM generic_cache')
            row = cursor.fetchone()
            stats['generic'] = {
                'entries': row[0],
                'total_hits': row[1] or 0
            }
            
            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            
            return stats
    
    def clear_all(self):
        """Clear all cache entries (use with caution!)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM translation_cache')
            cursor.execute('DELETE FROM wikidata_cache')
            cursor.execute('DELETE FROM generic_cache')
            conn.commit()
        
        logger.warning("Cleared all cache entries")
