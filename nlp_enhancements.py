"""
Advanced NLP enhancements for relation extraction.
Adds: Coreference Resolution, Enhanced Relations, Entity Linking, and Relation Confidence.
"""
import re
import time
import requests
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from requests.exceptions import RequestException, Timeout, ConnectionError
from logger import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from spacy.tokens import Span, Token, Doc
from cache_manager import CacheManager

logger = setup_logger(__name__)

try:
    from temporal_processor import TemporalProcessor
    TEMPORAL_AVAILABLE = True
except ImportError:
    logger.warning("Temporal processor not available")
    TEMPORAL_AVAILABLE = False
    TemporalProcessor = None  # type: ignore


class CoreferenceResolver:
    """
    Resolve pronouns (he/she/it/they) to actual entity mentions.
    Lightweight implementation using spaCy's built-in features.
    """
    
    def __init__(self):
        self.pronouns = {
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves'
        }
    
    def resolve_coreferences(self, doc) -> Dict[str, str]:
        """
        Map pronouns to their likely antecedent entities.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary mapping pronoun text (with position) to entity text
        """
        coref_map = {}
        entities_by_sent = defaultdict(list)
        
        # Group entities by sentence
        for ent in doc.ents:
            for sent_idx, sent in enumerate(doc.sents):
                if sent.start <= ent.start < sent.end:
                    entities_by_sent[sent_idx].append(ent)
                    break
        
        # Process each sentence
        for sent_idx, sent in enumerate(doc.sents):
            for token in sent:
                if token.text.lower() in self.pronouns and token.pos_ == 'PRON':
                    # Look for antecedent in previous sentences (recency bias)
                    antecedent = self._find_antecedent(
                        token, sent_idx, entities_by_sent, doc
                    )
                    if antecedent:
                        key = f"{token.text}_{token.i}"  # Include position to handle multiple pronouns
                        coref_map[key] = antecedent.text
                        logger.debug(f"Resolved '{token.text}' -> '{antecedent.text}'")
        
        return coref_map
    def _find_antecedent(self, pronoun_token: Token, sent_idx: int, entities_by_sent: Dict[int, List[Span]], doc: Doc) -> Optional[Span]:
        """
        Find the most likely entity that a pronoun refers to.
        
        Uses simple heuristics:
        1. Look in same sentence first
        2. Then previous sentence
        3. Then 2 sentences back
        4. Match gender/number when possible
        """

        # Gender/number hints
        masculine = {'he', 'him', 'his', 'himself'}
        feminine = {'she', 'her', 'hers', 'herself'}
        neuter = {'it', 'its', 'itself'}
        plural = {'they', 'them', 'their', 'theirs', 'themselves'}
        
        pronoun_lower = pronoun_token.text.lower()
        
        # Search in current and previous sentences (recency matters)
        for lookback in range(3):  # Current + 2 previous sentences
            search_sent_idx = sent_idx - lookback
            if search_sent_idx < 0:
                break
            
            candidates = entities_by_sent.get(search_sent_idx, [])
            
            # Reverse order to get most recent mention first
            for ent in reversed(candidates):
                # Skip if entity comes after pronoun
                if ent.start > pronoun_token.i:
                    continue
                
                # Match PERSON entities with gendered pronouns
                if ent.label_ == 'PERSON':
                    if pronoun_lower in masculine or pronoun_lower in feminine:
                        return ent  # Assume PERSON fits gendered pronouns
                    elif pronoun_lower in plural:
                        # Check if entity is plural (contains "and" or comma)
                        if ' and ' in ent.text or ',' in ent.text:
                            return ent
                
                # Match ORG/GPE with "it" or "they"
                elif ent.label_ in {'ORG', 'GPE'}:
                    if pronoun_lower in neuter or pronoun_lower in plural:
                        return ent
                
                # Match other entities with "it"
                elif pronoun_lower in neuter:
                    return ent
        
        return None
    
    def expand_relations_with_coreferences(
        self,
        relations: Dict[Tuple[str, str], List[str]],
        coref_map: Dict[str, str],
        doc
    ) -> Dict[Tuple[str, str], List[str]]:
        """
        
        Args:
            relations: Original relations dictionary
            coref_map: Pronoun -> entity mapping
            doc: spaCy document
            
        Returns:
            Expanded relations with pronouns resolved
        """
        expanded = defaultdict(list)
        
        # Copy original relations
        for key, reasons in relations.items():
            expanded[key].extend(reasons)
        
        # Add new relations by resolving pronouns in reasons
        for (ent1, ent2), reasons in relations.items():
            for reason in reasons:
                # Check if reason contains pronouns
                for pronoun_key, entity in coref_map.items():
                    pronoun_text = pronoun_key.split('_')[0]  # Remove position suffix
                    
                    # Replace pronoun in entities
                    new_ent1 = entity if ent1.lower() == pronoun_text.lower() else ent1
                    new_ent2 = entity if ent2.lower() == pronoun_text.lower() else ent2
                    
                    if new_ent1 != ent1 or new_ent2 != ent2:
                        # Found a coreference resolution
                        new_key = (new_ent1, new_ent2)
                        if new_key not in expanded or reason not in expanded[new_key]:
                            expanded[new_key].append(reason + " [coref]")
                            logger.debug(f"Added coref relation: {new_key}")
        
        return dict(expanded)


class EnhancedRelationExtractor:
    """
    Extract relationships using advanced dependency parsing patterns.
    Covers: employment, location, education, family, temporal, and more.
    """
    
    def __init__(self, enable_temporal: bool = True):
        # Define relationship patterns using dependency paths
        self.patterns = self._build_patterns()
        
        # Initialize temporal processor if available
        self.temporal_processor = None
        if enable_temporal and TEMPORAL_AVAILABLE and TemporalProcessor is not None:
            try:
                self.temporal_processor = TemporalProcessor()
                logger.info("Temporal processor enabled for enhanced relations")
            except Exception as e:
                logger.warning(f"Failed to initialize temporal processor: {e}")
        else:
            if enable_temporal and not TEMPORAL_AVAILABLE:
                logger.debug("Temporal processor not available, skipping initialization")
            elif enable_temporal and TemporalProcessor is None:
                logger.debug("TemporalProcessor is None, skipping initialization")
    
    def _build_patterns(self) -> List[Dict]:
        """
        Build comprehensive relationship patterns.
        Each pattern specifies: trigger words, dependency pattern, relation type.
        """
        return [
            # Employment relations
            {
                'triggers': ['ceo', 'cto', 'cfo', 'president', 'director', 'founder', 'employee', 'works', 'worked'],
                'entity_types': [('PERSON', 'ORG')],
                'relation': 'employment',
                'pattern': r'\b(ceo|cto|cfo|president|director|founder|chairman|executive|manager|employee)\s+(of|at|for)\b',
            },
            {
                'triggers': ['works at', 'employed by', 'hired by', 'joined', 'position at'],
                'entity_types': [('PERSON', 'ORG')],
                'relation': 'employment',
                'pattern': r'\b(works?|worked|employed|hired|joined|position)\s+(at|by|with)\b',
            },
            
            # Location relations
            {
                'triggers': ['based in', 'located in', 'headquarters', 'headquartered', 'office in'],
                'entity_types': [('ORG', 'GPE'), ('ORG', 'LOC')],
                'relation': 'location',
                'pattern': r'\b(based|located|headquartered|headquarters|office|offices)\s+(in|at)\b',
            },
            {
                'triggers': ['lives in', 'resides in', 'born in', 'from'],
                'entity_types': [('PERSON', 'GPE'), ('PERSON', 'LOC')],
                'relation': 'location',
                'pattern': r'\b(lives?|lived|resides?|resided|born|from)\s+(in|at)\b',
            },
            
            # Education relations
            {
                'triggers': ['graduated from', 'studied at', 'attended', 'alumnus', 'student at'],
                'entity_types': [('PERSON', 'ORG')],
                'relation': 'education',
                'pattern': r'\b(graduated?|studied|attended|alumnus|student|degree)\s+(from|at|in)\b',
            },
            
            # Family relations
            {
                'triggers': ['son of', 'daughter of', 'father', 'mother', 'parent', 'child', 'brother', 'sister'],
                'entity_types': [('PERSON', 'PERSON')],
                'relation': 'family',
                'pattern': r'\b(son|daughter|father|mother|parent|child|brother|sister|spouse|wife|husband)\s+(of)?\b',
            },
            
            # Ownership/founding
            {
                'triggers': ['founded', 'established', 'created', 'started', 'launched', 'owns', 'acquired'],
                'entity_types': [('PERSON', 'ORG'), ('ORG', 'ORG')],
                'relation': 'ownership',
                'pattern': r'\b(founded|established|created|started|launched|owns?|owned|acquired)\b',
            },
            
            # Temporal relations
            {
                'triggers': ['founded in', 'established in', 'created in', 'born on', 'died on'],
                'entity_types': [('PERSON', 'DATE'), ('ORG', 'DATE'), ('EVENT', 'DATE')],
                'relation': 'temporal',
                'pattern': r'\b(founded|established|created|born|died|occurred|happened)\s+(in|on|at)\b',
            },
            
            # Membership/affiliation
            {
                'triggers': ['member of', 'part of', 'affiliated with', 'associated with', 'belongs to'],
                'entity_types': [('PERSON', 'ORG'), ('PERSON', 'NORP'), ('ORG', 'ORG')],
                'relation': 'membership',
                'pattern': r'\b(member|part|affiliated|associated|belongs?)\s+(of|to|with)\b',
            },
        ]
    
    def extract_enhanced_relations(self, doc) -> Dict[Tuple[str, str], List[str]]:
        """
        Extract relations using pattern matching on dependency parses.
        Includes temporal information when available.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dictionary of relations with enhanced patterns (including temporal data)
        """
        relations = defaultdict(list)
        
        # Extract dates from document if temporal processor is available
        dates = []
        if self.temporal_processor:
            try:
                dates = self.temporal_processor.extract_and_normalize_dates(doc)
                if dates:
                    logger.debug(f"Extracted {len(dates)} dates from document")
            except Exception as e:
                logger.warning(f"Temporal extraction failed: {e}")
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            sent_ents = [ent for ent in sent.ents]
            
            if len(sent_ents) < 2:
                continue
            
            # Check each pattern
            for pattern_dict in self.patterns:
                # Check if any trigger word is in sentence
                if any(trigger in sent_text for trigger in pattern_dict['triggers']):
                    # Try to match the regex pattern
                    if re.search(pattern_dict['pattern'], sent_text, re.IGNORECASE):
                        # Find entity pairs matching the expected types
                        for i, ent1 in enumerate(sent_ents):
                            for ent2 in sent_ents[i+1:]:
                                if self._entities_match_pattern(ent1, ent2, pattern_dict):
                                    rel_type = pattern_dict['relation']
                                    reason = f"{sent.text} [enhanced:{rel_type}]"
                                    
                                    # Add temporal information if available
                                    if dates and self.temporal_processor:
                                        try:
                                            relation_span = (sent.start_char, sent.end_char)
                                            associated_dates = self.temporal_processor.associate_dates_with_relation(
                                                relation_span,
                                                dates,
                                                max_distance=50
                                            )
                                            if associated_dates:
                                                # Add dates to the reason string
                                                date_str = ", ".join(associated_dates)
                                                reason += f" [dates:{date_str}]"
                                                logger.debug(f"Associated dates {date_str} with {ent1.text} <-> {ent2.text}")
                                        except Exception as e:
                                            logger.debug(f"Failed to associate dates: {e}")
                                    
                                    relations[(ent1.text, ent2.text)].append(reason)
                                    logger.debug(f"Enhanced relation ({rel_type}): {ent1.text} <-> {ent2.text}")
        
        return dict(relations)
    
    def _entities_match_pattern(self, ent1, ent2, pattern_dict) -> bool:
        """Check if two entities match the expected types for a pattern."""
        for type1, type2 in pattern_dict['entity_types']:
            if (ent1.label_ == type1 and ent2.label_ == type2) or \
               (ent1.label_ == type2 and ent2.label_ == type1):
                return True
        return False


class EntityLinker:
    """
    Link entities to Wikidata/DBpedia for disambiguation and enrichment.
    """
    
    def __init__(
        self,
        threshold: float = 0.85,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.75,
        cache_dir: str = ".cache"
    ):
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.backoff_factor = max(0.0, backoff_factor)
        
        # Use persistent SQLite cache
        self._cache_manager = CacheManager(cache_dir=cache_dir, db_name="wikidata_cache.db")
        logger.info(f"EntityLinker initialized with SQLite cache at {cache_dir}/wikidata_cache.db")
    
    def link_entity(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """
        Link an entity to Wikidata and retrieve canonical information.
        
        Args:
            entity_text: Entity text to link
            entity_type: spaCy entity type (PERSON, ORG, etc.)
            
        Returns:
            Dictionary with: {
                'id': Wikidata ID,
                'label': Canonical label,
                'aliases': List of alternate names,
                'description': Short description,
                'url': Wikidata URL
            }
        """
        # Check SQLite cache
        cached = self._cache_manager.get_wikidata(entity_text)
        if cached:
            logger.debug(f"Cache HIT for entity: {entity_text}")
            return cached
        
        try:
            # Search Wikidata
            result = self._search_wikidata(entity_text, entity_type)
            
            if result:
                # Store in cache
                self._cache_manager.set_wikidata(
                    entity_text=entity_text,
                    qid=result.get('id'),
                    label=result.get('label'),
                    description=result.get('description'),
                    aliases=result.get('aliases', []),
                    metadata={'url': result.get('url'), 'confidence': result.get('confidence')},
                    ttl_days=180
                )
                logger.info(f"Linked '{entity_text}' -> {result['label']} ({result['id']})")
            else:
                # Cache negative result (no QID found) with shorter TTL
                self._cache_manager.set_wikidata(
                    entity_text=entity_text,
                    qid=None,
                    label=None,
                    description=None,
                    aliases=None,
                    metadata=None,
                    ttl_days=30
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"Entity linking failed for '{entity_text}': {e}")
            return None

    def link_entities_batch(self, entities: Dict[str, str]) -> Dict[str, Dict]:
        """
        Link a batch of entities to Wikidata in parallel.
        
        Args:
            entities: Dictionary of {entity_text: entity_label}
            
        Returns:
            Dictionary mapping entity_text to its linked metadata.
        """
        if not entities:
            return {}

        linked_entity_metadata: Dict[str, Dict] = {}

        # Use a reasonable number of workers; cap at 8 for API friendliness
        max_workers = min(8, max(1, len(entities)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entity = {
                executor.submit(self.link_entity, entity_text, entity_label): entity_text
                for entity_text, entity_label in entities.items()
            }

            logger.info(f"Linking {len(entities)} entities in parallel with {max_workers} workers...")

            for future in as_completed(future_to_entity):
                entity_text = future_to_entity[future]
                try:
                    result = future.result()
                    if result:
                        linked_entity_metadata[entity_text] = result
                except Exception as e:
                    logger.error(f"Entity linking failed for '{entity_text}' in batch: {e}")

        logger.info(f"Successfully linked {len(linked_entity_metadata)}/{len(entities)} entities in batch.")
        return linked_entity_metadata
    
    def _search_wikidata(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """
        Search Wikidata API for entity.
        """
        # Wikidata search API
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': 'en',
            'search': entity_text,
            'limit': 5
        }
        
        # Add user agent to avoid 403 errors
        headers = {
            'User-Agent': 'RelationsExtractor/1.0 (Educational project; contact@example.com)'
        }
        
        last_error: Optional[RequestException] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    search_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds
                )
                response.raise_for_status()
                data = response.json()
                
                if 'search' not in data or not data['search']:
                    return None
                
                # Get top result
                top_result = data['search'][0]
                
                # Calculate string similarity (simple metric)
                similarity = self._string_similarity(entity_text.lower(), top_result['label'].lower())
                
                if similarity < self.threshold:
                    logger.debug(
                        f"Low similarity ({similarity:.2f}) for '{entity_text}' -> '{top_result['label']}'"
                    )
                    return None
                
                # Get detailed entity info
                entity_id = top_result['id']
                entity_url = f"https://www.wikidata.org/wiki/{entity_id}"
                
                # Extract aliases
                aliases = []
                if 'aliases' in top_result:
                    aliases = [alias for alias in top_result['aliases'] if alias != top_result['label']]
                
                return {
                    'id': entity_id,
                    'label': top_result['label'],
                    'aliases': aliases,
                    'description': top_result.get('description', ''),
                    'url': entity_url,
                    'confidence': similarity
                }
            except (Timeout, ConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_seconds = self.backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "Wikidata request timed out for '%s' (attempt %d/%d). Retrying in %.2fs...",
                        entity_text,
                        attempt,
                        self.max_retries,
                        wait_seconds
                    )
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)
                    continue
                logger.warning(
                    "Wikidata request timed out for '%s' after %d attempts: %s",
                    entity_text,
                    self.max_retries,
                    e
                )
            except RequestException as e:
                logger.warning(
                    "Wikidata API request failed for '%s' on attempt %d/%d: %s",
                    entity_text,
                    attempt,
                    self.max_retries,
                    e
                )
                return None
        
        if last_error:
            logger.warning(
                "Wikidata API request aborted for '%s' after %d attempts: %s",
                entity_text,
                self.max_retries,
                last_error
            )
        
        return None
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using Levenshtein-like metric.
        Returns value between 0.0 and 1.0.
        """
        # Simple character overlap metric
        s1_set = set(s1.lower())
        s2_set = set(s2.lower())
        
        if not s1_set or not s2_set:
            return 0.0
        
        intersection = len(s1_set & s2_set)
        union = len(s1_set | s2_set)
        
        return intersection / union if union > 0 else 0.0


class RelationConfidenceScorer:
    """
    Enhanced confidence scoring for relations.
    """
    
    def calculate_confidence(
        self,
        ent1,
        ent2,
        sent,
        source_url: Optional[str] = None,
        relation_type: Optional[str] = None
    ) -> float:
        """
        Calculate comprehensive confidence score for a relation.
        
        Factors:
        1. Entity distance (closer = higher)
        2. Sentence structure (clear grammar = higher)
        3. Entity types compatibility
        4. Source credibility
        5. Relation type strength
        
        Returns:
            Confidence score 0.0 to 1.0
        """
        score = 0.0
        
        # 1. Distance factor (0-0.25)
        token_distance = abs(ent1.start - ent2.start)
        if token_distance <= 5:
            score += 0.25
        elif token_distance <= 10:
            score += 0.15
        elif token_distance <= 15:
            score += 0.05
        
        # 2. Sentence structure (0-0.25)
        # Check for verb between entities
        tokens_between = list(sent[min(ent1.end, ent2.end):max(ent1.start, ent2.start)])
        has_verb = any(t.pos_ == 'VERB' for t in tokens_between)
        has_prep = any(t.pos_ == 'ADP' for t in tokens_between)
        
        if has_verb:
            score += 0.15
        if has_prep:
            score += 0.10
        
        # 3. Entity types compatibility (0-0.25)
        compatible_pairs = {
            ('PERSON', 'ORG'), ('PERSON', 'GPE'), ('PERSON', 'EVENT'),
            ('ORG', 'GPE'), ('ORG', 'EVENT'), ('ORG', 'PRODUCT')
        }
        pair = (ent1.label_, ent2.label_)
        if pair in compatible_pairs or (pair[1], pair[0]) in compatible_pairs:
            score += 0.25
        else:
            score += 0.10  # Still give some points
        
        # 4. Source credibility (0-0.15)
        if source_url:
            credible_domains = [
                'wikipedia.org', 'bbc.com', 'nytimes.com', 'reuters.com',
                'apnews.com', 'theguardian.com', 'wsj.com', 'forbes.com',
                'bloomberg.com', 'nature.com', 'science.org'
            ]
            if any(domain in source_url.lower() for domain in credible_domains):
                score += 0.15
            else:
                score += 0.05
        
        # 5. Relation type strength (0-0.10)
        if relation_type:
            strong_types = {'employment', 'family', 'ownership', 'education'}
            if relation_type in strong_types:
                score += 0.10
            else:
                score += 0.05
        
        return min(max(score, 0.0), 1.0)
