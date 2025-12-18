"""
Knowledge graph construction and management.
"""
import networkx as nx
import pickle
from pathlib import Path
import json
import gzip
import hmac
import hashlib
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Sequence, Optional, Union, cast
try:
    from ..utils.logger import setup_logger
except:
    from utils.logger import setup_logger
from processor.entity_disambiguation import EntityDisambiguator
from processor.node_cleaner import clean_node_name, clean_entity_dict
from processor.provenance import Provenance, migrate_legacy_reasons, provenances_to_dicts, dicts_to_provenances, deduplicate_provenances
import re
logger = setup_logger(__name__)

# Label priority map used when merging nodes: higher number = higher priority
LABEL_PRIORITY = {
    'PERSON': 5,
    'ORG': 4,
    'GPE': 3,
    'LOC': 3,
    'EVENT': 2,
    'WORK_OF_ART': 2,
    'PRODUCT': 2,
    'DATE': 1,
    'UNKNOWN': 0
}


class KnowledgeGraph:
    """Manages the construction and manipulation of a knowledge graph."""
    
    def __init__(self, use_disambiguation: bool = True, similarity_threshold: float = 0.85):
        """
        Initialize an empty knowledge graph.
        
        Args:
            use_disambiguation: Enable entity disambiguation
            similarity_threshold: Minimum similarity for entity aliasing (0-1)
        """
        self.graph = nx.Graph()
        self.entities = {}
        self.relations = defaultdict(list)
        self.entity_mapping = {}  # Maps partial names to full names
        self.metadata = {
            'version': '1.0',
            'total_pages_processed': 0,
            'sources': set()
        }
        
        # Entity disambiguation
        self.use_disambiguation = use_disambiguation
        if use_disambiguation:
            self.disambiguator = EntityDisambiguator(
                similarity_threshold=similarity_threshold,
                enable_auto_discovery=True
            )
            logger.info(f"Entity disambiguation enabled (threshold={similarity_threshold})")
        else:
            self.disambiguator = None
    
    @classmethod
    def from_serialized(cls, data: dict, use_disambiguation: bool = True, similarity_threshold: float = 0.85) -> 'KnowledgeGraph':
        """
        Create a KnowledgeGraph instance from a serialized dict (legacy pickle format).
        
        Args:
            data: Dictionary containing 'graph', 'entities', 'relations', etc.
            use_disambiguation: Enable entity disambiguation
            similarity_threshold: Similarity threshold for disambiguation
            
        Returns:
            KnowledgeGraph instance with data loaded
        """
        # Create new instance
        kg = cls(use_disambiguation=use_disambiguation, similarity_threshold=similarity_threshold)
        
        # Load data from dict
        kg.graph = data.get('graph', nx.Graph())
        kg.entities = data.get('entities', {})
        kg.relations = defaultdict(list, data.get('relations', {}))
        kg.entity_mapping = data.get('entity_mapping', {})
        kg.metadata = data.get('metadata', {
            'version': '1.0',
            'total_pages_processed': 0,
            'sources': set()
        })
        
        logger.info(f"Loaded KnowledgeGraph from serialized data: {kg.get_statistics()['nodes']} nodes, {kg.get_statistics()['edges']} edges")
        
        return kg
    
    def _get_canonical_node_id(self, entity_text: str, entity_metadata: Optional[Dict[str, Dict]] = None) -> str:
        """
        Get canonical node ID: QID if available, otherwise cleaned text.
        
        Args:
            entity_text: Original entity text
            entity_metadata: Optional metadata dictionary with QIDs
            
        Returns:
            Canonical node ID (QID like 'Q218' or cleaned text)
        """
        # Try to get QID from metadata
        if entity_metadata and entity_text in entity_metadata:
            meta = entity_metadata[entity_text]
            qid = meta.get('id') or meta.get('qid')
            if qid:
                return qid
        
        # NEW: Prioritize disambiguator canonicalize over fallback
        # This ensures entities that were previously linked always resolve to their QID
        if self.use_disambiguation and self.disambiguator:
            canonical_id = self.disambiguator.canonicalize(entity_text)
            return canonical_id
        
        # Fallback: use cleaned and ASCII-normalized text for consistency
        from processor.node_cleaner import normalize_to_ascii
        cleaned = clean_node_name(entity_text)
        if cleaned:
            return normalize_to_ascii(cleaned)
        return entity_text
    
    def _get_display_name(self, node_id: str, original_text: str, entity_metadata: Optional[Dict[str, Dict]] = None) -> str:
        """
        Get human-readable display name for a node.
        
        Args:
            node_id: Canonical node ID
            original_text: Original entity text
            entity_metadata: Optional metadata with labels
            
        Returns:
            Display name (human-readable)
        """
        # Try metadata first
        if entity_metadata and original_text in entity_metadata:
            meta = entity_metadata[original_text]
            label = meta.get('label') or meta.get('canonical_label')
            if label:
                return label
        
        # Try disambiguator
        if self.use_disambiguation and self.disambiguator:
            display = self.disambiguator.get_display_name(node_id)
            if display and display != node_id:
                return display
        
        # If node_id is a QID but we have no metadata, use original text
        if node_id.startswith('Q') and node_id[1:].isdigit():
            return original_text
        
        # Fallback to node_id
        return node_id

    def _create_unification_key(self, name: str) -> str:
        """Creates a simple, ruthlessly normalized key for entity matching.

        Uses unidecode to ASCII-normalize, lowercases and removes non-alphanumerics.
        """
        try:
            from unidecode import unidecode
            import re
            key = unidecode(str(name)).lower()
            key = re.sub(r'[^a-z0-9]', '', key)
            return key
        except Exception:
            # Fallback: very conservative normalization
            return ''.join(ch for ch in str(name).lower() if ch.isalnum())

    def unify_duplicate_nodes(self) -> int:
        """
        Finds and merges duplicate nodes based on a normalized key.
        This is a final cleanup step to connect graph fragments.
        Returns the number of merged nodes (losers merged into winners).
        """
        logger.info("Starting final graph unification pass to merge duplicate nodes...")

        # Build groups by normalized key
        nodes_by_key: Dict[str, List[str]] = {}
        for node in list(self.graph.nodes()):
            key = self._create_unification_key(node)
            nodes_by_key.setdefault(key, []).append(node)

        merged_count = 0
        for key, nodes in nodes_by_key.items():
            if len(nodes) <= 1:
                continue

            # Pick winner as the node with highest degree
            nodes.sort(key=lambda n: nx.degree(self.graph, n), reverse=True)
            winner = nodes[0]
            losers = nodes[1:]

            logger.info(f"Unifying {len(losers)} duplicate node(s) into '{winner}': {losers}")

            for loser in losers:
                # Merge basic attributes (aliases, description, qid) into winner
                try:
                    winner_data = self.graph.nodes[winner]
                    loser_data = self.graph.nodes[loser]

                    # Merge aliases
                    if 'aliases' in loser_data:
                        winner_data.setdefault('aliases', set()).update(loser_data.get('aliases', set()))

                    # Prefer description if winner lacks one
                    if not winner_data.get('description') and loser_data.get('description'):
                        winner_data['description'] = loser_data.get('description')

                    # Prefer qid if winner lacks one
                    if not winner_data.get('qid') and loser_data.get('qid'):
                        winner_data['qid'] = loser_data.get('qid')
                        if loser_data.get('qid'):
                            winner_data['wikidata_url'] = f"https://www.wikidata.org/wiki/{loser_data.get('qid')}"
                except Exception:
                    pass

                # Contract loser into winner in-place
                try:
                    # networkx.contracted_nodes with copy=False modifies in place
                    self.graph = nx.contracted_nodes(self.graph, winner, loser, self_loops=False, copy=False)
                    merged_count += 1
                except Exception as e:
                    logger.warning(f"Failed to contract node '{loser}' into '{winner}': {e}")

        if merged_count > 0:
            logger.info(f"✓ Unification complete. Merged {merged_count} duplicate nodes.")
        else:
            logger.info("✓ Unification complete. No duplicate nodes found.")

        return merged_count
    
    def _ensure_node_exists(self, node_id: str, original_text: str, label: str, entity_metadata: Optional[Dict[str, Dict]] = None):
        """
        Ensure node exists in graph with full metadata.
        
        Args:
            node_id: Canonical node ID
            original_text: Original entity text
            label: Entity type label
            entity_metadata: Optional metadata dictionary
        """
        if not self.graph.has_node(node_id):
            display_name = self._get_display_name(node_id, original_text, entity_metadata)
            
            # Extract metadata
            qid = None
            wikidata_url = None
            aliases = set()
            description = None
            
            if entity_metadata and original_text in entity_metadata:
                meta = entity_metadata[original_text]
                qid = meta.get('id') or meta.get('qid')
                aliases = set(meta.get('aliases', []))
                description = meta.get('description')
                if qid:
                    wikidata_url = f"https://www.wikidata.org/wiki/{qid}"
            
            # Create node with rich metadata
            self.graph.add_node(
                node_id,
                label=label,
                display_name=display_name,
                qid=qid,
                wikidata_url=wikidata_url,
                aliases=aliases,
                description=description,
                provenance=set()  # URLs where this entity was mentioned
            )
        else:
            # Update existing node (merge metadata)
            node_data = self.graph.nodes[node_id]

            # Update label using priority map: only overwrite if incoming label has
            # equal or higher priority than the existing one.
            try:
                existing_label = node_data.get('label', 'UNKNOWN')
                incoming_label = label or 'UNKNOWN'
                existing_prio = LABEL_PRIORITY.get(existing_label, 0)
                incoming_prio = LABEL_PRIORITY.get(incoming_label, 0)

                if incoming_label != existing_label and incoming_prio >= existing_prio and incoming_label != 'UNKNOWN':
                    logger.debug(f"Updating label for node {node_id}: {existing_label} -> {incoming_label} (prio {existing_prio} -> {incoming_prio})")
                    node_data['label'] = incoming_label
            except Exception:
                # Fall back to permissive update if anything goes wrong
                if label and label != 'UNKNOWN':
                    node_data['label'] = label

            # Merge aliases
            if entity_metadata and original_text in entity_metadata:
                meta = entity_metadata[original_text]
                if 'aliases' in meta:
                    if 'aliases' not in node_data:
                        node_data['aliases'] = set()
                    node_data['aliases'].update(meta['aliases'])
    
    def add_entities(self, entities: Dict[str, str], entity_metadata: Optional[Dict[str, Dict]] = None):
        """
        Add entities to the graph with optional Wikidata metadata.
        
        Args:
            entities: Dictionary mapping entity text to entity label
            entity_metadata: Optional dictionary mapping entity text to metadata
                            (e.g., {'qid': 'Q312', 'canonical_label': 'Apple Inc.', 'aliases': [...]})
        """
        # STEP 1: Clean all entity names (remove diacritics, "The", "'s", etc.)
        cleaned_entities, name_mapping = clean_entity_dict(entities)
        
        # Store mapping from original to cleaned names
        for original, cleaned in name_mapping.items():
            if original != cleaned:
                self.entity_mapping[original] = cleaned
        
        # STEP 2: Register entities with disambiguator (including QID if available)
        if self.use_disambiguation and self.disambiguator and entity_metadata:
            for entity_text, metadata in entity_metadata.items():
                cleaned_text = name_mapping.get(entity_text, entity_text)
                qid = metadata.get('id') or metadata.get('qid')
                canonical_label = metadata.get('label') or metadata.get('canonical_label')
                aliases = metadata.get('aliases', [])
                
                if qid and canonical_label:
                    self.disambiguator.register_entity_with_qid(
                        entity_text=cleaned_text,
                        qid=qid,
                        canonical_label=canonical_label,
                        aliases=aliases
                    )
        
        # STEP 3: Update internal entities dict with cleaned names
        self.entities.update(cleaned_entities)
        
        # STEP 4: Build entity mapping for consolidation (partial names to full names)
        self._build_entity_mapping(cleaned_entities)
        
        # STEP 5: Add nodes to graph with canonical identifiers (QID or cleaned name)
        for entity, label in cleaned_entities.items():
            # Get canonical node ID (QID if available)
            node_id = self._get_canonical_node_id(entity, entity_metadata)
            
            # Ensure node exists with full metadata
            self._ensure_node_exists(node_id, entity, label, entity_metadata)
            
            logger.debug(f"Added entity: {entity} -> node_id={node_id}")
    
    def _build_entity_mapping(self, new_entities: Dict[str, str]):
        """
        Build mapping from partial entity names to full names.
        
        Args:
            new_entities: New entities being added
        """
        all_entities = list(self.entities.keys()) + list(new_entities.keys())
        
        for entity in all_entities:
            entity_lower = entity.lower()
            entity_words = entity_lower.split()
            
            # Find if this entity contains other entities as subparts
            for other_entity in all_entities:
                if entity == other_entity:
                    continue
                
                other_lower = other_entity.lower()
                
                # If other_entity is a word within entity, map it
                if other_lower in entity_words:
                    self.entity_mapping[other_entity] = entity
                    logger.debug(f"Mapping '{other_entity}' -> '{entity}'")
    
    def add_relations(self, relations: Dict[Tuple[str, str], Union[List[str], List[Provenance]]], temporal_info: Optional[List[Dict]] = None, entity_metadata: Optional[Dict[str, Dict]] = None):
        """
        Add relations to the graph with optional temporal information and entity metadata.
        Uses entity disambiguation if enabled.
        
        Args:
            relations: Dictionary mapping entity pairs to relationship reasons (legacy strings) or Provenance objects
            temporal_info: Optional list of temporal information dictionaries (deprecated, use dates in reasons)
            entity_metadata: Optional dictionary mapping entity text to Wikidata metadata
        """
        # Register any new entities with QIDs
        if self.use_disambiguation and self.disambiguator and entity_metadata:
            for entity_text, metadata in entity_metadata.items():
                qid = metadata.get('id') or metadata.get('qid')
                canonical_label = metadata.get('label') or metadata.get('canonical_label')
                aliases = metadata.get('aliases', [])
                
                if qid and canonical_label:
                    self.disambiguator.register_entity_with_qid(
                        entity_text=entity_text,
                        qid=qid,
                        canonical_label=canonical_label,
                        aliases=aliases
                    )
        
        for (source, target), reasons in relations.items():
            # STEP 1: Clean source and target entity names first
            cleaned_source = clean_node_name(source)
            cleaned_target = clean_node_name(target)
            
            # Skip if either entity is invalid (preposition, article, etc.)
            if not cleaned_source or not cleaned_target:
                logger.debug(f"Skipping relation with invalid entity: '{source}' -> '{target}'")
                continue
            
            # STEP 2: Get canonical node IDs (QID if available, otherwise cleaned text)
            source_id = self._get_canonical_node_id(cleaned_source, entity_metadata)
            target_id = self._get_canonical_node_id(cleaned_target, entity_metadata)
            
            # Get display names for logging
            source_display = self._get_display_name(source_id, cleaned_source, entity_metadata)
            target_display = self._get_display_name(target_id, cleaned_target, entity_metadata)
            
            # Log if entities were canonicalized to QIDs
            if source_id != cleaned_source:
                logger.debug(f"Canonicalized: '{cleaned_source}' -> '{source_display}' ({source_id})")
            if target_id != cleaned_target:
                logger.debug(f"Canonicalized: '{cleaned_target}' -> '{target_display}' ({target_id})")
            
            # Skip self-loops
            if source_id == target_id:
                continue
            
            # STEP 3: Convert reasons to Provenance objects if they're legacy strings
            provenances: List[Provenance] = []
            if reasons and isinstance(reasons[0], str):
                # Legacy string format - migrate to Provenance
                # Use typing.cast to satisfy static type checkers that 'reasons' is List[str] in this branch
                provenances = migrate_legacy_reasons(cast(List[str], reasons))
                logger.debug(f"Migrated {len(reasons)} legacy reasons to Provenance objects")
            elif reasons and isinstance(reasons[0], Provenance):
                # Already Provenance objects
                provenances = cast(List[Provenance], reasons)
            else:
                logger.warning(f"Unexpected reasons type for {source_id} -> {target_id}: {type(reasons[0]) if reasons else 'empty'}")
                continue
            
            # Add to internal tracking (keep as legacy reasons for now for backwards compat)
            existing_reasons = self.relations[(source_id, target_id)]
            for prov in provenances:
                legacy_str = prov.to_legacy_string()
                if legacy_str not in existing_reasons:
                    existing_reasons.append(legacy_str)
            
            # Ensure both entities exist in the graph with full metadata
            self._ensure_node_exists(source_id, cleaned_source, "UNKNOWN", entity_metadata)
            self._ensure_node_exists(target_id, cleaned_target, "UNKNOWN", entity_metadata)
            
            # STEP 4: Deduplicate provenances
            provenances = deduplicate_provenances(provenances)
            
            # Calculate average confidence from provenances
            confidences = [p.confidence for p in provenances if p.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            # Collect all dates from provenances
            all_dates = []
            for prov in provenances:
                if prov.dates:
                    all_dates.extend(prov.dates)
            unique_dates = sorted(list(set(all_dates))) if all_dates else None
            
            # Collect relation types from provenances (use most common non-None type)
            relation_types = [p.relation_type for p in provenances if p.relation_type is not None]
            primary_relation_type = None
            if relation_types:
                # Use the most common relation type
                from collections import Counter
                type_counts = Counter(relation_types)
                primary_relation_type = type_counts.most_common(1)[0][0]
            
            # STEP 5: Add or update edge with structured provenance
            from typing import Any
            edge_data: Dict[str, Any] = {
                'provenance': provenances_to_dicts(provenances),  # Store as dicts for serialization
                'reasons': [p.text for p in provenances]  # Keep text-only list for backwards compat
            }
            
            if avg_confidence is not None:
                edge_data['confidence'] = avg_confidence
            
            # Store relation type if found
            if primary_relation_type:
                edge_data['relation_type'] = primary_relation_type
            
            # Store dates if found
            if unique_dates:
                edge_data['dates'] = unique_dates
            
            # Legacy temporal_info support (prefer dates from provenances)
            if temporal_info:
                edge_data['temporal'] = temporal_info
            
            # Add or update edge
            if self.graph.has_edge(source_id, target_id):
                # Merge with existing edge
                existing = self.graph.edges[source_id, target_id]
                
                # Merge provenances
                existing_provs = dicts_to_provenances(existing.get('provenance', []))
                merged_provs = deduplicate_provenances(existing_provs + provenances)
                existing['provenance'] = provenances_to_dicts(merged_provs)
                existing['reasons'] = [p.text for p in merged_provs]
                
                # Update confidence (average)
                if avg_confidence is not None:
                    if 'confidence' in existing:
                        all_confs = [existing['confidence'], avg_confidence]
                        existing['confidence'] = sum(all_confs) / len(all_confs)
                    else:
                        existing['confidence'] = avg_confidence
                
                # Merge relation types (use most common across all provenances)
                if primary_relation_type:
                    merged_types = [p.relation_type for p in merged_provs if p.relation_type is not None]
                    if merged_types:
                        from collections import Counter
                        type_counts = Counter(merged_types)
                        existing['relation_type'] = type_counts.most_common(1)[0][0]
                
                # Merge dates
                if unique_dates:
                    existing_dates = existing.get('dates', [])
                    for date in unique_dates:
                        if date not in existing_dates:
                            existing_dates.append(date)
                    existing['dates'] = sorted(existing_dates)
                
                # Merge temporal info (legacy)
                if temporal_info:
                    existing.setdefault('temporal', []).extend(temporal_info)
            else:
                # Create new edge
                self.graph.add_edge(source_id, target_id, **edge_data)
    
    def merge_entities_and_relations(
        self, 
        entities: Dict[str, str], 
        relations: Dict[Tuple[str, str], Union[List[str], List[Provenance]]],
        entity_metadata: Optional[Dict[str, Dict]] = None
    ):
        """
        Add entities and relations to the graph with optional metadata.
        
        Args:
            entities: Dictionary of entities to add
            relations: Dictionary of relations to add (can be legacy strings or Provenance objects)
            entity_metadata: Optional dictionary with Wikidata metadata
        """
        self.add_entities(entities, entity_metadata)
        self.add_relations(relations, entity_metadata=entity_metadata)
        
        # Clean up partial entity nodes after merging
        self._consolidate_nodes()
    
    def _consolidate_nodes(self):
        """
        Remove partial entity nodes and redirect their edges to full names.
        """
        nodes_to_remove = set()
        edges_to_add = []
        
        for partial, full in self.entity_mapping.items():
            if self.graph.has_node(partial) and self.graph.has_node(full):
                # Transfer edges from partial to full name
                for neighbor in list(self.graph.neighbors(partial)):
                    if neighbor != full:  # Avoid self-loops
                        # Get edge data
                        edge_data = self.graph.edges[partial, neighbor]
                        
                        # Add edge to full name
                        if self.graph.has_edge(full, neighbor):
                            # Merge reasons
                            existing_reasons = self.graph.edges[full, neighbor]['reasons']
                            for reason in edge_data.get('reasons', []):
                                if reason not in existing_reasons:
                                    existing_reasons.append(reason)
                        else:
                            edges_to_add.append((full, neighbor, edge_data))
                
                nodes_to_remove.add(partial)
        
        # Add new edges
        for source, target, data in edges_to_add:
            self.graph.add_edge(source, target, **data)
        
        # Remove partial nodes
        self.graph.remove_nodes_from(nodes_to_remove)
        
        if nodes_to_remove:
            logger.info(f"Consolidated {len(nodes_to_remove)} partial entity names into full names")
    
    def get_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph instance.
        
        Returns:
            NetworkX Graph object
        """
        return self.graph
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'unique_entities': len(self.entities),
            'unique_relations': len(self.relations)
        }
    
    def get_all_entities(self) -> list[str]:
        """
        Get a list of all entity names in the graph.
        
        Returns:
            List of entity names
        """
        return list(self.graph.nodes())
    
    def get_all_relations(self) -> dict:
        """
        Get all relationships in the graph.
        
        Returns:
            Dictionary mapping (source, target) -> [reasons]
        """
        relations = {}
        for source, target, data in self.graph.edges(data=True):
            reasons = data.get('reasons', [])
            relations[(source, target)] = reasons
        return relations
    
    def save_to_file(self, filepath: str | Path):
        """
        Save the knowledge graph to a file for incremental building.
        
        Args:
            filepath: Path to save the graph
        """
        filepath = Path(filepath)
        
        try:
            # Serialize graph to node-link JSON (safe, portable)
            graph_data = nx.readwrite.json_graph.node_link_data(self.graph)
            # Serialize relations as a list of [source, target, reasons] so JSON keys
            # are always strings (JSON object keys must be strings).
            relations_serializable = []
            for (src, tgt), reasons in self.relations.items():
                relations_serializable.append([src, tgt, reasons])

            data = {
                'graph': graph_data,
                'entities': self.entities,
                'relations': relations_serializable,
                'entity_mapping': self.entity_mapping,
                'metadata': self.metadata
            }

            json_bytes = json.dumps(data, default=str).encode('utf-8')

            # Optional HMAC key from environment for signing persisted files
            key_hex = os.environ.get('SAG_PERSISTENCE_KEY')
            if key_hex:
                try:
                    key = bytes.fromhex(key_hex)
                except Exception:
                    key = None
            else:
                key = None

            with gzip.open(filepath, 'wb') as f:
                if key:
                    sig = hmac.new(key, json_bytes, hashlib.sha256).hexdigest().encode('ascii')
                    f.write(sig + b"\n" + json_bytes)
                else:
                    # No signing key configured; write unsigned payload with explicit marker
                    f.write(b"NOSIG\n" + json_bytes)

            logger.info(f"Saved knowledge graph to {filepath} ({self.get_statistics()['nodes']} nodes, {self.get_statistics()['edges']} edges)")
        except Exception as e:
            logger.error(f"Failed to save graph to {filepath}: {e}")
            raise
    
    def load_from_file(self, filepath: str | Path, merge: bool = True) -> bool:
        """
        Load a knowledge graph from file. Can merge with existing graph or replace.
        
        Args:
            filepath: Path to load the graph from
            merge: If True, merge with existing graph. If False, replace.
            
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return False
        
        try:
            # First attempt: our signed gzipped JSON format
            try:
                with gzip.open(filepath, 'rb') as f:
                    first_line = f.readline().strip()
                    payload = f.read()

                if first_line == b"NOSIG":
                    logger.warning("Loading unsigned graph file (NOSIG). This is less secure; consider configuring SAG_PERSISTENCE_KEY for signing.")
                    data = json.loads(payload.decode('utf-8'))
                else:
                    # Validate HMAC if key available
                    key_hex = os.environ.get('SAG_PERSISTENCE_KEY')
                    key = None
                    if key_hex:
                        try:
                            key = bytes.fromhex(key_hex)
                        except Exception:
                            key = None

                    if key is None:
                        logger.error("Graph file appears signed but no SAG_PERSISTENCE_KEY is configured; refusing to load to avoid unsafe deserialization")
                        return False

                    expected = first_line.decode('ascii')
                    actual = hmac.new(key, payload, hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(expected, actual):
                        logger.error("Graph file HMAC mismatch; refusing to load")
                        return False

                    data = json.loads(payload.decode('utf-8'))
            except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as e:
                # Fallback: allow legacy pickle only if explicitly enabled via env var
                allow_pickle = os.environ.get('SAG_ALLOW_LEGACY_PICKLE', '0') == '1'
                if not allow_pickle:
                    logger.error("Failed to parse graph file as signed JSON (and pickles are disabled). To allow legacy pickle loading set SAG_ALLOW_LEGACY_PICKLE=1")
                    logger.debug("Parse error: %s", e)
                    return False

                logger.warning("Attempting to load legacy pickle graph file because SAG_ALLOW_LEGACY_PICKLE=1 (unsafe)")
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

            # At this point `data` should be a dict with a node-link graph
            if 'graph' in data and isinstance(data['graph'], dict):
                loaded_graph = nx.readwrite.json_graph.node_link_graph(data['graph'])
            else:
                # Support older format where the graph may already be a NetworkX Graph
                loaded_graph = data.get('graph', nx.Graph())

            if merge:
                logger.info(f"Merging loaded graph with existing graph...")
                old_stats = self.get_statistics()
                self.graph = nx.compose(self.graph, loaded_graph)
                self.entities.update(data.get('entities', {}))
                # Support legacy dict mappings as well as the new list-of-triples
                relations_data = data.get('relations', {})
                if isinstance(relations_data, list):
                    for item in relations_data:
                        try:
                            src, tgt, reasons = item
                        except Exception:
                            continue
                        key = (src, tgt)
                        if key in self.relations:
                            for reason in reasons:
                                if reason not in self.relations[key]:
                                    self.relations[key].append(reason)
                        else:
                            self.relations[key] = list(reasons)
                elif isinstance(relations_data, dict):
                    for key, value in relations_data.items():
                        # Key may already be a tuple (legacy pickle) or something else.
                        if isinstance(key, (list, tuple)):
                            k = tuple(key)
                        else:
                            k = key

                        if k in self.relations:
                            for reason in value:
                                if reason not in self.relations[k]:
                                    self.relations[k].append(reason)
                        else:
                            self.relations[k] = list(value)

                self.entity_mapping.update(data.get('entity_mapping', {}))
                if 'metadata' in data:
                    self.metadata['total_pages_processed'] += data['metadata'].get('total_pages_processed', 0)
                    # metadata.sources may be list in JSON; convert to set
                    sources = data['metadata'].get('sources', [])
                    if isinstance(sources, (list, set)):
                        self.metadata['sources'].update(set(sources))

                new_stats = self.get_statistics()
                logger.info(f"Merged graph: {old_stats['nodes']} -> {new_stats['nodes']} nodes, {old_stats['edges']} -> {new_stats['edges']} edges")
            else:
                logger.info(f"Replacing existing graph with loaded graph...")
                self.graph = loaded_graph
                self.entities = data.get('entities', {})
                # Reconstruct relations into defaultdict with tuple keys.
                raw_rel = data.get('relations', {})
                new_rel = defaultdict(list)
                if isinstance(raw_rel, list):
                    for item in raw_rel:
                        try:
                            src, tgt, reasons = item
                        except Exception:
                            continue
                        new_rel[(src, tgt)] = list(reasons)
                elif isinstance(raw_rel, dict):
                    for key, value in raw_rel.items():
                        if isinstance(key, (list, tuple)):
                            k = tuple(key)
                        else:
                            k = key
                        new_rel[k] = list(value)
                self.relations = new_rel
                self.entity_mapping = data.get('entity_mapping', {})
                md = data.get('metadata', {})
                self.metadata = {
                    'version': md.get('version', '1.0'),
                    'total_pages_processed': md.get('total_pages_processed', 0),
                    'sources': set(md.get('sources', []))
                }
                stats = self.get_statistics()
                logger.info(f"Loaded graph: {stats['nodes']} nodes, {stats['edges']} edges")

            return True
        except Exception as e:
            logger.error(f"Failed to load graph from {filepath}: {e}")
            return False
    
    def deduplicate_and_merge_reasons(self) -> int:
        """
        Deduplicate reasons by text content and merge multiple sources.
        
        If the same reason text appears multiple times with different URLs,
        merge them into one reason with multiple source links.
        
        Format: "reason text|||url1|||url2|||url3"
        
        Returns:
            Number of duplicate reasons merged
        """
        total_merged = 0
        
        for source, target, data in self.graph.edges(data=True):
            reasons = data.get('reasons', [])
            
            if len(reasons) <= 1:
                continue
            
            # Group reasons by text content
            reason_groups = {}  # text -> [urls]
            
            for reason in reasons:
                if '|||' in reason:
                    # Extract text and URL
                    parts = reason.split('|||')
                    text = parts[0].strip()
                    url = parts[1].strip() if len(parts) > 1 else None
                    
                    # Normalize text for comparison (remove extra spaces, lowercase)
                    normalized_text = ' '.join(text.split()).lower()
                    
                    if normalized_text not in reason_groups:
                        reason_groups[normalized_text] = {
                            'original_text': text,  # Keep original casing
                            'urls': []
                        }
                    
                    if url and url not in reason_groups[normalized_text]['urls']:
                        reason_groups[normalized_text]['urls'].append(url)
                else:
                    # Reason without URL
                    text = reason.strip()
                    normalized_text = ' '.join(text.split()).lower()
                    
                    if normalized_text not in reason_groups:
                        reason_groups[normalized_text] = {
                            'original_text': text,
                            'urls': []
                        }
            
            # Rebuild reasons list with merged sources
            merged_reasons = []
            duplicates_found = 0
            
            for normalized_text, group_data in reason_groups.items():
                original_text = group_data['original_text']
                urls = group_data['urls']
                
                if urls:
                    # Create merged reason with all URLs
                    merged_reason = original_text + '|||' + '|||'.join(urls)
                    merged_reasons.append(merged_reason)
                    
                    if len(urls) > 1:
                        duplicates_found += len(urls) - 1
                        logger.debug(f"Merged {len(urls)} sources for reason: {original_text[:50]}...")
                else:
                    # Reason without URL
                    merged_reasons.append(original_text)
            
            # Update edge with deduplicated reasons
            if len(merged_reasons) < len(reasons):
                self.graph.edges[source, target]['reasons'] = merged_reasons
                total_merged += (len(reasons) - len(merged_reasons))
        
        if total_merged > 0:
            logger.info(f"Merged {total_merged} duplicate reasons across all edges")
        
        return total_merged
    
    def filter_by_degree(self, min_degree: int = 1) -> nx.Graph:
        """
        Create a filtered graph containing only nodes with minimum degree.
        Removes isolated nodes (degree 0) and weakly connected nodes.
        
        Args:
            min_degree: Minimum degree for nodes to include
            
        Returns:
            Filtered NetworkX Graph
        """
        filtered_graph = self.graph.copy()
        # Get degree view and filter nodes
        degree_dict = dict(filtered_graph.degree())  # type: ignore
        nodes_to_remove = [
            node for node, degree in degree_dict.items()
            if int(degree) < int(min_degree)
        ]
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        logger.info(f"Filtered graph: removed {len(nodes_to_remove)} nodes with degree < {min_degree}")
        return filtered_graph
    
    def remove_isolated_nodes(self) -> int:
        """
        Remove all isolated nodes (nodes with no connections) from the graph.
        
        Returns:
            Number of nodes removed
        """
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        
        logger.info(f"Removed {len(isolated_nodes)} isolated nodes")
        return len(isolated_nodes)
    
    def filter_by_relevance(self, query_terms: Sequence[str], min_relevance: float = 0.0) -> nx.Graph:
        """
        Filter graph to keep only entities relevant to query terms.
        
        Args:
            query_terms: List of query terms to check relevance
            min_relevance: Minimum relevance score (0.0 to 1.0)
            
        Returns:
            Filtered graph with relevant nodes only
        """
        filtered_graph = self.graph.copy()
        nodes_to_remove = []
        
        query_lower = [term.lower() for term in query_terms]
        
        for node in filtered_graph.nodes():
            node_lower = node.lower()
            
            # Check if any query term appears in node name
            is_relevant = any(term in node_lower or node_lower in term for term in query_lower)
            
            if not is_relevant:
                # Check if node is connected to relevant nodes
                neighbors = list(filtered_graph.neighbors(node))
                relevant_neighbors = sum(
                    1 for neighbor in neighbors
                    if any(term in neighbor.lower() or neighbor.lower() in term for term in query_lower)
                )
                
                relevance_score = relevant_neighbors / len(neighbors) if neighbors else 0.0
                
                if relevance_score < min_relevance:
                    nodes_to_remove.append(node)
        
        filtered_graph.remove_nodes_from(nodes_to_remove)
        logger.info(f"Filtered by relevance: removed {len(nodes_to_remove)} irrelevant nodes")
        
        return filtered_graph

    def remove_nodes_by_label(self, labels: Sequence[str]) -> int:
        """
        Remove nodes whose node attribute 'label' matches any of the provided labels.

        Args:
            labels: Sequence of labels to remove (case-insensitive)

        Returns:
            Number of nodes removed
        """
        to_remove = []
        labels_lower = {l.lower() for l in labels}

        for node in list(self.graph.nodes()):
            node_label = self.graph.nodes[node].get('label')
            if node_label and node_label.lower() in labels_lower:
                to_remove.append(node)

        if to_remove:
            self.graph.remove_nodes_from(to_remove)
            logger.info(f"Removed {len(to_remove)} nodes by label: {', '.join(labels)}")

        return len(to_remove)
    
    def filter_unconnected_nodes(self, query_terms: Sequence[str], max_distance: int | None = None) -> int:
        """
        Remove all nodes that are not connected to query-relevant entities.
        Uses NetworkX connected components to find the query's component.
        
        Args:
            query_terms: List of query terms to identify core entities
            max_distance: Maximum graph distance from query entities (None = unlimited, i.e., any connection)
            
        Returns:
            Number of nodes removed
        """
        from collections import deque
        import unicodedata
        
        def normalize_romanian(text):
            """Normalize Romanian diacritics for matching."""
            replacements = {
                'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
                'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            # Also remove any remaining diacritics
            text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                          if unicodedata.category(c) != 'Mn')
            return text
        
        query_lower = [term.lower() for term in query_terms]
        query_normalized = [normalize_romanian(term.lower()) for term in query_terms]
        full_query = " ".join(query_terms).lower()
        full_query_normalized = normalize_romanian(full_query)
        
        # Find all query-relevant entities (core nodes)
        # Match full query, individual terms, or nodes containing terms
        core_nodes = set()
        
        for node in self.graph.nodes():
            node_lower = node.lower()
            node_normalized = normalize_romanian(node_lower)
            
            # Exact match with full query (original or normalized)
            if node_lower == full_query or node_normalized == full_query_normalized:
                core_nodes.add(node)
                logger.debug(f"Core query node (exact match): {node}")
            # Any query term matches (original or normalized)
            elif any(term.lower() == node_lower or normalize_romanian(term.lower()) == node_normalized 
                    for term in query_terms):
                core_nodes.add(node)
                logger.debug(f"Core query node (term match): {node}")
            # Node contains whole-word query term or normalized term (use word boundaries)
            elif any(re.search(r'\b' + re.escape(term) + r'\b', node_lower) or
                     re.search(r'\b' + re.escape(norm_term) + r'\b', node_normalized)
                    for term, norm_term in zip(query_lower, query_normalized)):
                core_nodes.add(node)
                logger.debug(f"Core query node (partial word match): {node}")
        
        if not core_nodes:
            logger.warning("No core query nodes found! Keeping all nodes.")
            return 0
        
        logger.info(f"Found {len(core_nodes)} core query nodes: {list(core_nodes)[:5]}{'...' if len(core_nodes) > 5 else ''}")
        
        # Use NetworkX to find all connected components
        connected_components = list(nx.connected_components(self.graph))
        logger.info(f"Found {len(connected_components)} connected components in graph")
        
        # Find which component(s) contain the query nodes
        query_components = set()
        for component in connected_components:
            if any(core_node in component for core_node in core_nodes):
                query_components.update(component)
        
        if not query_components:
            logger.warning("Query nodes not found in any component! Keeping all nodes.")
            return 0
        
        # If max_distance is specified, filter by distance using BFS
        if max_distance is not None:
            nodes_to_keep = set(core_nodes)
            
            for core_node in core_nodes:
                # BFS from this core node
                queue = deque([(core_node, 0)])
                visited = {core_node}
                
                while queue:
                    current, distance = queue.popleft()
                    
                    if distance < max_distance:
                        for neighbor in self.graph.neighbors(current):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                nodes_to_keep.add(neighbor)
                                queue.append((neighbor, distance + 1))
                                logger.debug(f"Keeping node at distance {distance + 1}: {neighbor}")
        else:
            # Keep all nodes in the query's connected component
            nodes_to_keep = query_components
        
        # Remove all nodes not in the keep set
        all_nodes = set(self.graph.nodes())
        nodes_to_remove = all_nodes - nodes_to_keep
        
        self.graph.remove_nodes_from(nodes_to_remove)
        
        if nodes_to_remove:
            distance_msg = f"within {max_distance} hops" if max_distance is not None else "in connected component"
            logger.info(f"Removed {len(nodes_to_remove)} nodes not connected to query ({distance_msg})")
            logger.info(f"Kept {len(nodes_to_keep)} nodes connected to query")
            
            # Log some examples of removed nodes
            if nodes_to_remove:
                examples = list(nodes_to_remove)[:10]
                logger.info(f"Examples of removed nodes: {examples}")
        
        return len(nodes_to_remove)
    
    def remove_irrelevant_nodes(self, query_terms: Sequence[str], min_relevance: float = 0.0) -> int:
        """
        Remove nodes that have no relevance to the query terms.
        More aggressive than filter_unconnected_nodes - checks actual text similarity.
        
        Args:
            query_terms: List of query terms
            min_relevance: Minimum relevance score (0.0 = any match, 1.0 = exact match)
            
        Returns:
            Number of nodes removed
        """
        query_lower = [term.lower() for term in query_terms]
        query_set = set(query_lower)
        
        nodes_to_remove = []
        
        for node in list(self.graph.nodes()):
            node_lower = node.lower()
            node_words = set(node_lower.replace('_', ' ').split())
            
            # Calculate relevance score
            relevance = 0.0
            
            # Exact match
            if node_lower in query_lower or any(term in node_lower for term in query_lower):
                relevance = 1.0
            # Word overlap
            elif query_set & node_words:  # Set intersection
                relevance = len(query_set & node_words) / len(query_set)
            # Check if node is connected to relevant nodes (has edges)
            elif self.graph.has_node(node):
                degree = int(dict(self.graph.degree())[node])  # type: ignore
                if degree > 0:
                    # Check neighbors for relevance
                    neighbors = list(self.graph.neighbors(node))
                    relevant_neighbors = 0
                    for neighbor in neighbors:
                        neighbor_lower = neighbor.lower()
                        if any(term in neighbor_lower for term in query_lower):
                            relevant_neighbors += 1
                    
                    if len(neighbors) > 0:
                        relevance = relevant_neighbors / len(neighbors) * 0.5  # Indirect relevance
            
            # Remove if not relevant enough
            if relevance <= min_relevance:
                nodes_to_remove.append(node)
                logger.debug(f"Removing irrelevant node (score: {relevance:.2f}): {node}")
        
        if nodes_to_remove:
            self.graph.remove_nodes_from(nodes_to_remove)
            logger.info(f"Removed {len(nodes_to_remove)} irrelevant nodes")
        
        return len(nodes_to_remove)
    
    def filter_year_nodes(self, query_terms: Sequence[str]) -> int:
        """
        Remove nodes that represent standalone years and are not related to the query.
        Keeps year nodes that are directly connected to query-relevant entities.
        
        Args:
            query_terms: List of query terms to check relevance
            
        Returns:
            Number of year nodes removed
        """
        import re
        nodes_to_remove = []
        query_lower = [term.lower() for term in query_terms]
        
        for node in self.graph.nodes():
            # Check if node is a year (4 digits, possibly with BC/AD, or formats like "1990s")
            node_text = node.strip()
            is_year = bool(re.match(r'^(AD\s*|BC\s*)?\d{4}(s)?(\s*(AD|BC))?$', node_text, re.IGNORECASE))
            
            if is_year:
                # Check if this year node is connected to query-relevant entities
                neighbors = list(self.graph.neighbors(node))
                
                if not neighbors:
                    # Isolated year node - remove it
                    nodes_to_remove.append(node)
                    logger.debug(f"Removing isolated year node: {node}")
                else:
                    # Check if any neighbor is related to the query
                    has_relevant_connection = False
                    for neighbor in neighbors:
                        neighbor_lower = neighbor.lower()
                        if any(term in neighbor_lower or neighbor_lower in term for term in query_lower):
                            has_relevant_connection = True
                            break
                    
                    # If not connected to query-relevant entities, remove it
                    if not has_relevant_connection:
                        nodes_to_remove.append(node)
                        logger.debug(f"Removing unrelated year node: {node} (neighbors: {neighbors})")
        
        self.graph.remove_nodes_from(nodes_to_remove)
        
        if nodes_to_remove:
            logger.info(f"Removed {len(nodes_to_remove)} year nodes not related to query")
        
        return len(nodes_to_remove)
    
    def remove_image_and_file_nodes(self) -> int:
        """
        Remove nodes and edges that reference images or local files.
        Checks node names and edge reasons for image extensions and file paths.
        
        Returns:
            Number of nodes and edges removed
        """
        import re
        
        # Image extensions and patterns
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico', 
                          '.tiff', '.tif', '.raw', '.heic', '.heif', '.pdf']
        
        # File path patterns
        file_patterns = [
            r'[a-zA-Z]:\\',  # Windows paths: C:\, D:\
            r'^/[a-zA-Z]/',  # Unix absolute paths starting with /
            r'file://',      # File protocol
            r'\\\\',         # UNC paths: \\server\
        ]
        
        nodes_to_remove = []
        edges_to_remove = []
        
        # Check nodes
        for node in list(self.graph.nodes()):
            node_lower = node.lower()
            
            # Check for image extensions
            if any(ext in node_lower for ext in image_extensions):
                nodes_to_remove.append(node)
                logger.debug(f"Removing image node: {node}")
                continue
            
            # Check for file paths
            if any(re.search(pattern, node, re.IGNORECASE) for pattern in file_patterns):
                nodes_to_remove.append(node)
                logger.debug(f"Removing file path node: {node}")
                continue
        
        # Check edges (reasons)
        for source, target, data in list(self.graph.edges(data=True)):
            reasons = data.get('reasons', [])
            
            # Check each reason for image/file references
            filtered_reasons = []
            for reason in reasons:
                reason_lower = reason.lower()
                
                # Check for image extensions in reason
                has_image = any(ext in reason_lower for ext in image_extensions)
                
                # Check for file paths in reason
                has_file_path = any(re.search(pattern, reason, re.IGNORECASE) for pattern in file_patterns)
                
                if not has_image and not has_file_path:
                    filtered_reasons.append(reason)
                else:
                    logger.debug(f"Removing reason with image/file reference: {reason[:100]}...")
            
            # If no reasons left, mark edge for removal
            if len(filtered_reasons) == 0:
                edges_to_remove.append((source, target))
                logger.debug(f"Removing edge with no valid reasons: {source} -> {target}")
            elif len(filtered_reasons) < len(reasons):
                # Update edge with filtered reasons
                self.graph.edges[source, target]['reasons'] = filtered_reasons
        
        # Remove marked nodes
        self.graph.remove_nodes_from(nodes_to_remove)
        
        # Remove marked edges
        self.graph.remove_edges_from(edges_to_remove)
        
        total_removed = len(nodes_to_remove) + len(edges_to_remove)
        
        if nodes_to_remove or edges_to_remove:
            logger.info(f"Removed {len(nodes_to_remove)} image/file nodes and {len(edges_to_remove)} edges with image/file references")
        
        return total_removed
    
    def remove_file_protocol_relations(self) -> int:
        """
        Remove relations (edges) that have reasons pointing to files instead of websites.
        This includes:
        - file:// protocol URLs
        - Local file paths (C:\\, D:\\, /path/to/file, etc.)
        - Non-HTTP(S) URLs (ftp://, data:, blob:, etc.)
        
        Also attempts to fix malformed URLs (e.g., file:///D:/path/%7Chttp://real-url)
        by extracting the real URL from the malformed path.
        
        After removing invalid reasons, if an edge has no valid reasons left, remove the edge.
        Then remove any nodes that become isolated as a result.
        
        Returns:
            Total number of edges removed + nodes removed
        """
        import re
        from urllib.parse import unquote
        
        edges_to_remove = []
        
        # Patterns for local file paths and non-web URLs
        invalid_url_patterns = [
            r'^file://',           # file:// protocol
            r'^[a-zA-Z]:\\',      # Windows paths: C:\, D:\
            r'^/[a-zA-Z]/',       # Unix absolute paths
            r'^\\\\',             # UNC paths: \\server\
            r'^ftp://',           # FTP protocol
            r'^data:',            # Data URLs
            r'^blob:',            # Blob URLs
            r'^javascript:',      # JavaScript URLs
            r'^mailto:',          # Email links
        ]
        
        # Check all edges for invalid URLs in reasons
        for source, target, data in self.graph.edges(data=True):
            reasons = data.get('reasons', [])
            filtered_reasons = []
            
            for reason in reasons:
                is_valid_reason = True
                updated_reason = reason
                
                # Extract URL from reason (format: "text|||url")
                if '|||' in reason:
                    text_part, url = reason.rsplit('|||', 1)
                    url = url.strip()
                    
                    # Try to fix malformed URLs (e.g., file:///D:/path/%7Chttps://real-url)
                    fixed_url = self._extract_real_url(url)
                    
                    if fixed_url and fixed_url != url:
                        logger.debug(f"Fixed malformed URL: {url[:80]} -> {fixed_url[:80]}")
                        url = fixed_url
                        updated_reason = f"{text_part}|||{url}"
                    
                    # Check if URL matches any invalid pattern
                    for pattern in invalid_url_patterns:
                        if re.match(pattern, url, re.IGNORECASE):
                            logger.debug(f"Filtering non-web URL reason: {url[:100]}...")
                            is_valid_reason = False
                            break
                    
                    # Also check if URL starts with http:// or https://
                    if is_valid_reason and url:
                        if not url.lower().startswith('http://') and not url.lower().startswith('https://'):
                            logger.debug(f"Filtering non-HTTP(S) URL: {url[:100]}...")
                            is_valid_reason = False
                else:
                    # Reason without URL separator - check if it contains file path patterns
                    for pattern in invalid_url_patterns[:5]:  # Check file-related patterns only
                        if re.search(pattern, reason, re.IGNORECASE):
                            logger.debug(f"Filtering reason with file path: {reason[:100]}...")
                            is_valid_reason = False
                            break
                
                if is_valid_reason:
                    filtered_reasons.append(updated_reason)
            
            # If no valid reasons remain, mark edge for removal
            if len(filtered_reasons) == 0:
                edges_to_remove.append((source, target))
                logger.debug(f"Removing edge with no valid web reasons: {source} -> {target}")
            elif len(filtered_reasons) < len(reasons):
                # Update edge with filtered reasons
                self.graph.edges[source, target]['reasons'] = filtered_reasons
        
        # Remove edges with no valid web URLs
        edges_removed = len(edges_to_remove)
        self.graph.remove_edges_from(edges_to_remove)
        
        if edges_removed > 0:
            logger.info(f"Removed {edges_removed} edges with only file/non-web URL references")
        
        # Now remove any nodes that became isolated after removing edges
        isolated_nodes = list(nx.isolates(self.graph))
        nodes_removed = len(isolated_nodes)
        self.graph.remove_nodes_from(isolated_nodes)
        
        if nodes_removed > 0:
            logger.info(f"Removed {nodes_removed} nodes that became isolated after edge removal")
        
        return edges_removed + nodes_removed
    
    def _extract_real_url(self, url: str) -> str:
        """
        Extract real URL from malformed URLs with comprehensive pattern matching.
        
        Examples:
        - file:///D:/relations_extractor/%7Chttps://example.com -> https://example.com
        - file:///path/%7Chttp://example.com -> http://example.com
        - file:///C:/Users/cache/page.html|https://real-site.com -> https://real-site.com
        - C:\\path\\to\\file|https://example.com -> https://example.com
        
        Args:
            url: Potentially malformed URL
            
        Returns:
            Extracted real URL or original URL if no fix needed
        """
        import re
        from urllib.parse import unquote
        
        if not url:
            return url
        
        # Quick check: if it's already a clean HTTP(S) URL, return as-is
        if url.startswith('http://') or url.startswith('https://'):
            # Make sure it doesn't contain embedded malformed parts
            if not any(marker in url for marker in ['file:///', '%7C', '|', 'C:\\', 'D:\\']):
                return url
        
        # Decode URL-encoded characters (%7C = |, %20 = space, etc.)
        decoded_url = unquote(url)
        
        # Pattern 1: Extract URL after pipe character (| or %7C)
        # Covers: file:///path|https://..., C:\path|http://..., any_prefix|http://...
        pipe_patterns = [
            r'\|(https?://[^\s|]+)',          # After literal pipe
            r'%7[Cc](https?://[^\s|]+)',      # After encoded pipe %7C or %7c
        ]
        
        for pattern in pipe_patterns:
            match = re.search(pattern, url)
            if match:
                extracted = unquote(match.group(1))
                if self._is_valid_http_url(extracted):
                    return extracted
        
        # Pattern 2: Extract from file:/// paths containing embedded URLs
        if 'file:///' in decoded_url.lower():
            # Try to find any http(s) URL within the path
            match = re.search(r'(https?://[^\s\'"<>|]+)', decoded_url)
            if match:
                extracted = match.group(1)
                if self._is_valid_http_url(extracted):
                    return extracted
        
        # Pattern 3: Windows path with embedded URL (C:\, D:\, etc.)
        if re.match(r'^[A-Z]:\\', url, re.IGNORECASE):
            match = re.search(r'(https?://[^\s\'"<>|]+)', decoded_url)
            if match:
                extracted = match.group(1)
                if self._is_valid_http_url(extracted):
                    return extracted
        
        # Pattern 4: URL after any separator (space, %20, tab, newline)
        match = re.search(r'[\s%20\t\n](https?://[^\s|]+)', decoded_url)
        if match:
            extracted = match.group(1)
            if self._is_valid_http_url(extracted):
                return extracted
        
        # Pattern 5: Try to extract first valid-looking HTTP URL from anywhere in string
        match = re.search(r'(https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+)', decoded_url)
        if match:
            extracted = match.group(1)
            if self._is_valid_http_url(extracted):
                return extracted
        
        # No valid URL found, return original (will be filtered out later)
        return url
    
    def _is_valid_http_url(self, url: str) -> bool:
        """
        Validate that a string is a proper HTTP/HTTPS URL.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid HTTP(S) URL, False otherwise
        """
        if not url:
            return False
        
        # Must start with http:// or https://
        if not (url.startswith('http://') or url.startswith('https://')):
            return False
        
        # Must have a domain after protocol
        import re
        # Basic check: protocol + domain + optional path
        pattern = r'^https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+$'
        if not re.match(pattern, url):
            return False
        
        # Must not contain file path markers
        invalid_markers = ['file:///', 'C:\\', 'D:\\', 'E:\\', '\\\\', '%7C']
        if any(marker in url for marker in invalid_markers):
            return False
        
        # Check for minimum domain structure (at least one dot and valid TLD)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc or '.' not in parsed.netloc:
                return False
            # Domain should not be absurdly long
            if len(parsed.netloc) > 253:  # DNS limit
                return False
        except Exception:
            return False
        
        return True
    
    def get_entity_types(self) -> Dict[str, int]:
        """
        Get count of entities by type.
        
        Returns:
            Dictionary mapping entity types to counts
        """
        type_counts = defaultdict(int)
        for entity, label in self.entities.items():
            type_counts[label] += 1
        return dict(type_counts)
