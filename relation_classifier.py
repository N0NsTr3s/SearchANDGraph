"""
Relation Classification Module

Analyzes dependency paths between entities to classify relationship types.
Maps natural language patterns to structured relation types like is_CEO_of, was_born_in, etc.
"""
import spacy
from typing import Optional, List, Tuple, Set
from logger import setup_logger
from spacy import tokens
logger = setup_logger(__name__)


class RelationClassifier:
    """Classifies relationships between entities based on dependency parsing."""
    
    # Relation type patterns based on dependency paths
    RELATION_PATTERNS = {
        # Employment/Leadership relations
        'is_CEO_of': {
            'verbs': {'is', 'serves', 'works'},
            'nouns': {'ceo', 'chief executive', 'chief executive officer'},
            'preps': {'of', 'at', 'for'}
        },
        'is_founder_of': {
            'verbs': {'founded', 'established', 'created', 'started', 'co-founded'},
            'nouns': {'founder', 'co-founder'},
            'preps': {'of'}
        },
        'is_president_of': {
            'verbs': {'is', 'serves', 'elected'},
            'nouns': {'president', 'chairman', 'chair'},
            'preps': {'of', 'at'}
        },
        'is_director_of': {
            'verbs': {'directs', 'heads', 'leads', 'manages'},
            'nouns': {'director', 'head', 'leader', 'manager'},
            'preps': {'of', 'at'}
        },
        'works_for': {
            'verbs': {'works', 'employed', 'serves', 'hired'},
            'nouns': {'employee', 'staff', 'worker'},
            'preps': {'for', 'at', 'in'}
        },
        
        # Ownership/Control relations
        'owns': {
            'verbs': {'owns', 'possesses', 'holds'},
            'nouns': {'owner', 'ownership'},
            'preps': {'of'}
        },
        'acquired': {
            'verbs': {'acquired', 'bought', 'purchased'},
            'nouns': {'acquisition', 'purchase'},
            'preps': {'by', 'from'}
        },
        'sold_to': {
            'verbs': {'sold', 'divested'},
            'nouns': {'sale', 'divestiture'},
            'preps': {'to', 'by'}
        },
        
        # Location relations
        'located_in': {
            'verbs': {'located', 'based', 'situated', 'headquartered'},
            'nouns': {'location', 'headquarters', 'office', 'base'},
            'preps': {'in', 'at'}
        },
        'born_in': {
            'verbs': {'born'},
            'nouns': {'birth', 'birthplace'},
            'preps': {'in', 'at'}
        },
        'lives_in': {
            'verbs': {'lives', 'resides', 'stays'},
            'nouns': {'residence', 'home'},
            'preps': {'in', 'at'}
        },
        
        # Family relations
        'married_to': {
            'verbs': {'married', 'wed'},
            'nouns': {'spouse', 'wife', 'husband', 'partner'},
            'preps': {'to', 'with'}
        },
        'parent_of': {
            'verbs': {'fathered', 'mothered'},
            'nouns': {'father', 'mother', 'parent', 'dad', 'mom'},
            'preps': {'of'}
        },
        'child_of': {
            'verbs': {'born'},
            'nouns': {'son', 'daughter', 'child'},
            'preps': {'of', 'to'}
        },
        
        # Educational relations
        'studied_at': {
            'verbs': {'studied', 'attended', 'graduated'},
            'nouns': {'student', 'graduate', 'alumnus', 'alumni'},
            'preps': {'at', 'from'}
        },
        'graduated_from': {
            'verbs': {'graduated'},
            'nouns': {'graduate', 'degree', 'diploma'},
            'preps': {'from', 'at'}
        },
        
        # Political relations
        'elected_to': {
            'verbs': {'elected', 'appointed', 'nominated'},
            'nouns': {'election', 'appointment'},
            'preps': {'to', 'as'}
        },
        'member_of': {
            'verbs': {'joined', 'member'},
            'nouns': {'member', 'membership'},
            'preps': {'of', 'in'}
        },
        
        # Business relations
        'partnered_with': {
            'verbs': {'partnered', 'collaborated', 'cooperated'},
            'nouns': {'partner', 'partnership', 'collaboration'},
            'preps': {'with', 'between'}
        },
        'competes_with': {
            'verbs': {'competes', 'rivals'},
            'nouns': {'competitor', 'rival', 'competition'},
            'preps': {'with', 'against'}
        },
    }
    
    def __init__(self):
        """Initialize the relation classifier."""
        self.relation_counts = {}  # Track usage for debugging
        
    def classify_relation(
        self,
        entity1: tokens.Span,
        entity2: tokens.Span,
        sentence: tokens.Span
    ) -> Optional[str]:
        """
        Classify the relationship between two entities based on dependency path.
        
        Args:
            entity1: First entity span
            entity2: Second entity span
            sentence: Sentence span containing both entities
            
        Returns:
            Relation type string or None if no clear relation found
        """
        # Extract dependency path
        path_tokens = self._get_dependency_path(entity1, entity2, sentence)
        
        if not path_tokens:
            return None
        
        # Extract verbs, nouns, and prepositions from path
        path_verbs = {token.lemma_.lower() for token in path_tokens if token.pos_ == 'VERB'}
        path_nouns = {token.lemma_.lower() for token in path_tokens if token.pos_ == 'NOUN'}
        path_preps = {token.lemma_.lower() for token in path_tokens if token.pos_ == 'ADP'}
        
        # Also check for compound nouns (e.g., "chief executive officer")
        compound_nouns = self._extract_compound_nouns(path_tokens)
        path_nouns.update(compound_nouns)
        
        # Match against known patterns
        best_match = None
        best_score = 0
        
        for relation_type, pattern in self.RELATION_PATTERNS.items():
            score = 0
            
            # Check verb matches
            verb_matches = path_verbs & pattern['verbs']
            if verb_matches:
                score += 3  # Verbs are strong indicators
            
            # Check noun matches
            noun_matches = path_nouns & pattern['nouns']
            if noun_matches:
                score += 2  # Nouns are important
            
            # Check preposition matches
            prep_matches = path_preps & pattern['preps']
            if prep_matches:
                score += 1  # Prepositions help disambiguate
            
            # Need at least verb or noun match
            if (verb_matches or noun_matches) and score > best_score:
                best_score = score
                best_match = relation_type
        
        if best_match:
            self.relation_counts[best_match] = self.relation_counts.get(best_match, 0) + 1
            logger.debug(f"Classified relation: {entity1.text} --[{best_match}]--> {entity2.text} (score: {best_score})")
        
        return best_match
    
    def _get_dependency_path(
        self,
        entity1: tokens.Span,
        entity2: tokens.Span,
        sentence: tokens.Span
    ) -> List[tokens.Token]:
        """
        Find the shortest dependency path between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            sentence: Sentence containing both
            
        Returns:
            List of tokens on the path
        """
        # Get root tokens of each entity
        ent1_tokens = [token for token in sentence if entity1.start <= token.i < entity1.end]
        ent2_tokens = [token for token in sentence if entity2.start <= token.i < entity2.end]
        
        if not ent1_tokens or not ent2_tokens:
            return []
        
        # Use head token of each entity
        ent1_head = ent1_tokens[0].head
        ent2_head = ent2_tokens[0].head
        
        # Find path from entity1 to root
        path1 = []
        current = ent1_head
        visited1 = set()
        while current and current.i not in visited1:
            path1.append(current)
            visited1.add(current.i)
            if current == current.head:  # Reached root
                break
            current = current.head
        
        # Find path from entity2 to root
        path2 = []
        current = ent2_head
        visited2 = set()
        while current and current.i not in visited2:
            path2.append(current)
            visited2.add(current.i)
            if current == current.head:  # Reached root
                break
            current = current.head
        
        # Find common ancestor
        common_ancestor = None
        for token in path1:
            if token in path2:
                common_ancestor = token
                break
        
        if not common_ancestor:
            # No common ancestor, return tokens between entities
            start_idx = min(ent1_head.i, ent2_head.i)
            end_idx = max(ent1_head.i, ent2_head.i)
            return [token for token in sentence if start_idx <= token.i <= end_idx]
        
        # Build path: entity1 -> common ancestor -> entity2
        path_to_ancestor = []
        for token in path1:
            path_to_ancestor.append(token)
            if token == common_ancestor:
                break
        
        path_from_ancestor = []
        for token in path2:
            if token == common_ancestor:
                break
            path_from_ancestor.append(token)
        
        # Combine paths
        full_path = path_to_ancestor + list(reversed(path_from_ancestor))
        
        return full_path

    def _extract_compound_nouns(self, tokens: List[tokens.Token]) -> Set[str]:
        """
        Extract compound nouns like 'chief executive officer' from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Set of compound noun phrases
        """
        compounds = set()
        
        for i, token in enumerate(tokens):
            if token.pos_ == 'NOUN':
                # Look for compound structure
                phrase_tokens = [token]
                
                # Check children for compounds
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ in {'NOUN', 'ADJ'}:
                        phrase_tokens.insert(0, child)
                
                # Build phrase
                if len(phrase_tokens) > 1:
                    phrase = ' '.join(t.lemma_.lower() for t in sorted(phrase_tokens, key=lambda x: x.i))
                    compounds.add(phrase)
        
        return compounds
    
    def get_statistics(self) -> dict:
        """
        Get statistics about classified relations.
        
        Returns:
            Dictionary with relation type counts
        """
        return {
            'total_classified': sum(self.relation_counts.values()),
            'relation_types': len(self.relation_counts),
            'counts_by_type': dict(sorted(self.relation_counts.items(), key=lambda x: x[1], reverse=True))
        }
