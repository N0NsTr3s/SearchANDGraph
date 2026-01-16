"""Contact/identifier extraction for OSINT-style scans.

This module extracts contact-like identifiers (emails, phones, social profile URLs)
from raw text and turns them into graph-ready entity labels + evidence.

Design goals:
- Multi-country friendly: phone extraction is permissive by default and can
  optionally use `phonenumbers` when installed.
- Evidence-first: each hit retains the supporting sentence snippet.
- Conservative linking: contacts are linked to PERSONs in the same sentence if
  present, otherwise to ORGs in the same sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import re

from processor.provenance import Provenance


_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])([a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,})(?![\w.+-])")

# A permissive international phone matcher.
# We keep it simple and normalize afterwards.
_PHONE_RE = re.compile(
    r"(?x)\b(\+?\d[\d\s().-]{6,}\d)\b"
)

# Common numeric date patterns that can be mistaken for phone numbers.
_DATE_LIKE_RE = re.compile(
    r"(?x)\b("
    r"(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])"
    r"|(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}"
    r"|(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])"
    r")\b"
)

# Year ranges like 2012-2016, 2012–2016, 2012—2016
_YEAR_RANGE_RE = re.compile(
    r"(?x)\b(?:19|20)\d{2}\s*[-/\u2013\u2014]\s*(?:19|20)\d{2}\b"
)

# Compact year spans like 20172018
_YEAR_SPAN_RE = re.compile(r"\b(?:19|20)\d{2}(?:19|20)\d{2}\b")

# Social/profile URLs we want to treat as OSINT identifiers.
# Keep this list small and high-signal.
_SOCIAL_URL_RE = re.compile(
    r"(?i)\b(https?://(?:www\.)?(?:"
    r"linkedin\.com/(?:in|company)/"
    r"|x\.com/|twitter\.com/|facebook\.com/|instagram\.com/|github\.com/"
    r")[^\s\]\[\)\(<>\"']{2,})"
)


def looks_like_email(value: str) -> bool:
    return bool(_EMAIL_RE.fullmatch(value.strip()))


def looks_like_social_url(value: str) -> bool:
    return bool(_SOCIAL_URL_RE.fullmatch(value.strip()))


def looks_like_phone(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    if _looks_like_date(v):
        return False
    # Must contain at least 7 digits.
    digits = sum(ch.isdigit() for ch in v)
    return digits >= 7 and bool(_PHONE_RE.fullmatch(v))


def _looks_like_date(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    if _DATE_LIKE_RE.fullmatch(v):
        return True
    if _YEAR_RANGE_RE.fullmatch(v):
        return True
    if _YEAR_SPAN_RE.fullmatch(v):
        return True
    compact = re.sub(r"\s+", "", v)
    return bool(_DATE_LIKE_RE.fullmatch(compact))


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_social_url(value: str) -> str:
    v = value.strip()
    # Strip trailing punctuation common in prose.
    v = v.rstrip(".,;:)]}'\"")
    # Remove common tracking query params (keep it simple; don't fully parse).
    if "?" in v:
        base, _q = v.split("?", 1)
        v = base
    # Strip trailing slash (except scheme root)
    if v.endswith("/") and len(v) > len("https://") + 2:
        v = v.rstrip("/")
    return v


def normalize_phone(value: str, default_region: Optional[str] = None) -> str:
    """Normalize a phone-like string.

    If `phonenumbers` is installed we attempt E.164. Otherwise we fall back to
    a simple canonical form that preserves leading '+'.
    """
    raw = value.strip()
    # Keep + and digits only.
    plus = raw.strip().startswith("+")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw

    # Optional: phonenumbers for better multi-country normalization.
    try:
        import phonenumbers  # type: ignore

        region = (default_region or "").upper() or None
        parsed = phonenumbers.parse(raw, region)
        if phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass

    return ("+" if plus else "") + digits


@dataclass(frozen=True)
class ContactHit:
    kind: str  # EMAIL | PHONE | SOCIAL
    value: str
    sentence: str


def iter_contact_hits(text: str) -> Iterable[ContactHit]:
    """Yield contact-like hits from text using regexes."""
    if not text:
        return

    # Work on sentence-ish chunks to keep evidence clean.
    # We don't depend on spaCy here to keep this usable for PDF text too.
    chunks = re.split(r"(?<=[.!?\n])\s+", text)

    for chunk in chunks:
        sentence = chunk.strip()
        if not sentence:
            continue

        for m in _EMAIL_RE.finditer(sentence):
            yield ContactHit("EMAIL", normalize_email(m.group(1)), sentence)

        for m in _SOCIAL_URL_RE.finditer(sentence):
            yield ContactHit("SOCIAL", normalize_social_url(m.group(1)), sentence)

        for m in _PHONE_RE.finditer(sentence):
            candidate = m.group(1)
            # Avoid swallowing years/IDs; require enough digits.
            if sum(ch.isdigit() for ch in candidate) < 7:
                continue
            # Skip common date-like patterns (e.g., 2024-01-15, 15/01/2024).
            if _looks_like_date(candidate):
                continue
            yield ContactHit("PHONE", normalize_phone(candidate), sentence)


def build_contact_entities_and_relations(
    text: str,
    source_url: str,
    sentence_entities: List[Tuple[str, List[Tuple[str, str]]]] | None = None,
    default_region: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], List[Provenance]]]:
    """Build entity labels + relations for contacts.

    Args:
        text: Input text (translated/canonical preferred)
        source_url: Provenance URL
        sentence_entities: Optional list of (sentence_text, [(ent_text, ent_label), ...])
            Typically derived from spaCy so we can link contacts to nearby PERSON/ORG.
        default_region: Optional region hint for phone normalization (e.g., 'US', 'RO').

    Returns:
        (entities, relations) where:
        - entities maps contact identifier -> label (EMAIL/PHONE/SOCIAL)
        - relations maps (anchor_entity, contact_identifier) -> [Provenance]
    """
    entities: Dict[str, str] = {}
    relations: Dict[Tuple[str, str], List[Provenance]] = {}

    # Fallback: if no sentence entities were provided, treat all contacts as unanchored.
    sent_to_ents = {s: ents for s, ents in (sentence_entities or [])}

    for hit in iter_contact_hits(text):
        value = hit.value
        label = hit.kind
        entities[value] = label

        # Determine anchors for linking.
        anchors: List[Tuple[str, str]] = []
        if sentence_entities is not None:
            ents = sent_to_ents.get(hit.sentence)
            if ents:
                # Prefer PERSON anchors, otherwise ORG.
                persons = [e for e in ents if e[1] == "PERSON"]
                orgs = [e for e in ents if e[1] == "ORG"]
                anchors = persons or orgs

        if not anchors:
            # No local anchor found; we keep the contact as a node only.
            continue

        for ent_text, ent_label in anchors:
            prov = Provenance(
                text=hit.sentence,
                source_url=source_url,
                confidence=1.0,
                relation_type=(
                    "has_email" if label == "EMAIL" else
                    "has_phone" if label == "PHONE" else
                    "has_social_profile"
                ),
            )
            relations.setdefault((ent_text, value), []).append(prov)

    return entities, relations
