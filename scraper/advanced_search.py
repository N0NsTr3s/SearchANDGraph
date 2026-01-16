"""
Advanced search query builder with Google search operators.
"""
from typing import List, Optional, Dict, Any, Iterable
from dataclasses import dataclass
from urllib.parse import urlparse
try:
    from ..utils.logger import setup_logger
except:
    from utils.logger import setup_logger
logger = setup_logger(__name__)


@dataclass
class SearchQuery:
    """Represents an advanced search query with operators."""
    query: str
    exact_phrases: Optional[List[str]] = None
    excluded_words: Optional[List[str]] = None
    site: Optional[str] = None
    site_optional: bool = False
    filetype: Optional[str] = None
    intitle: Optional[str] = None
    inurl: Optional[str] = None
    after_date: Optional[str] = None  # Format: YYYY-MM-DD
    before_date: Optional[str] = None
    
    def __post_init__(self):
        if self.exact_phrases is None:
            self.exact_phrases = []
        if self.excluded_words is None:
            self.excluded_words = []
    
    def build(self) -> str:
        """
        Build the complete search query with all operators.
        
        Returns:
            Formatted search query string
        """
        base_parts = []
        
        # Base query
        if self.query:
            base_parts.append(self.query)
        
        # Exact phrase matches (quoted)
        for phrase in self.exact_phrases: # type: ignore
            base_parts.append(f'"{phrase}"')
        
        # Excluded words (with minus sign)
        for word in self.excluded_words: # type: ignore
            base_parts.append(f"-{word}")
        
        # File type filter
        if self.filetype:
            base_parts.append(f"filetype:{self.filetype}")
        
        # Title keyword
        if self.intitle:
            base_parts.append(f'intitle:"{self.intitle}"')
        
        # URL keyword
        if self.inurl:
            base_parts.append(f"inurl:{self.inurl}")
        
        # Date range filters
        if self.after_date:
            base_parts.append(f"after:{self.after_date}")
        if self.before_date:
            base_parts.append(f"before:{self.before_date}")

        # Site restriction
        site_part = ""
        if self.site:
            site_part = f"site:{self.site}"

        if site_part and self.site_optional:
            with_site = " ".join([p for p in base_parts + [site_part] if p])
            without_site = " ".join([p for p in base_parts if p])
            if with_site and without_site:
                query_string = f"({with_site}) OR ({without_site})"
            else:
                query_string = with_site or without_site
        else:
            parts = base_parts + ([site_part] if site_part else [])
            query_string = " ".join(parts)
        logger.debug(f"Built search query: {query_string}")
        return query_string


class AdvancedSearchBuilder:
    """Build sophisticated search queries for targeted information retrieval."""

    @staticmethod
    def _normalize_site(site: str) -> Optional[str]:
        if not site:
            return None
        raw = str(site).strip()
        if not raw:
            return None
        if "//" not in raw:
            raw = f"https://{raw}"
        try:
            parsed = urlparse(raw)
            host = (parsed.netloc or "").lower().strip()
            if host.startswith("www."):
                host = host[4:]
            return host or None
        except Exception:
            return None

    @staticmethod
    def _normalize_sites(sites: Optional[Iterable[str]]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for site in sites or []:
            cleaned = AdvancedSearchBuilder._normalize_site(str(site))
            if not cleaned:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @staticmethod
    def create_contextual_dorks(
        name: str,
        related_entities: Optional[Iterable[str]] = None,
        related_sites: Optional[Iterable[str]] = None
    ) -> List[SearchQuery]:
        """Create custom dorks using related entities and related sites."""
        queries: List[SearchQuery] = []

        base = (name or "").strip()
        if not base:
            return queries

        rel_entities: List[str] = []
        seen_entities = set()
        for ent in related_entities or []:
            ent_clean = str(ent).strip()
            if not ent_clean:
                continue
            if ent_clean.lower() == base.lower():
                continue
            if ent_clean.lower() in seen_entities:
                continue
            seen_entities.add(ent_clean.lower())
            rel_entities.append(ent_clean)

        rel_sites = AdvancedSearchBuilder._normalize_sites(related_sites)

        # Cross-entity dorks (example: "{query}" + "{Other big node}")
        for ent in rel_entities:
            queries.append(SearchQuery(query=f'"{base}" + "{ent}"'))
            queries.append(SearchQuery(query=f'"{base}" "{ent}"'))
            queries.append(SearchQuery(query=f'"{base}" "{ent}" relationship OR connection'))

        # Site-related dorks: bias toward related sites without hard restriction
        for site in rel_sites:
            queries.append(SearchQuery(query=f'"{base}"', site=site, site_optional=True))
            for ent in rel_entities:
                queries.append(SearchQuery(query=f'"{base}" "{ent}"', site=site, site_optional=True))

        return queries
    
    @staticmethod
    def create_person_query(name: str, context: Optional[str] = None) -> List[SearchQuery]:
        """
        Create search queries optimized for finding information about a person.
        
        Args:
            name: Person's name
            context: Optional context (e.g., company, role, location)
            
        Returns:
            List of SearchQuery objects for comprehensive person search
        """
        queries = []
        
        # General biography query
        queries.append(SearchQuery(
            query=f"{name} biography",
            exact_phrases=[name]
        ))
        
        # News articles
        queries.append(SearchQuery(
            query=f"{name}",
            exact_phrases=[name],
            site="",  # Will be filled with news domains
            after_date="2020-01-01"  # Recent news
        ))
        
        # LinkedIn profile
        queries.append(SearchQuery(
            query=f"{name}",
            exact_phrases=[name],
            site="linkedin.com",
            inurl="in"
        ))
        
        # Context-specific search
        if context:
            queries.append(SearchQuery(
                query=f"{name} {context}",
                exact_phrases=[name]
            ))
        
        # PDF documents (reports, papers)
        queries.append(SearchQuery(
            query=f"{name}",
            exact_phrases=[name],
            filetype="pdf"
        ))
        
        return queries
    
    @staticmethod
    def create_organization_query(org_name: str, context: Optional[str] = None) -> List[SearchQuery]:
        """
        Create search queries optimized for finding organization information.
        
        Args:
            org_name: Organization name
            context: Optional context (e.g., industry, event)
            
        Returns:
            List of SearchQuery objects for comprehensive organization search
        """
        queries = []
        
        # Official website
        queries.append(SearchQuery(
            query=f"{org_name} official",
            exact_phrases=[org_name]
        ))
        
        # About page
        queries.append(SearchQuery(
            query=f"{org_name}",
            exact_phrases=[org_name],
            inurl="about"
        ))
        
        # Annual reports and presentations (PDF)
        queries.append(SearchQuery(
            query=f"{org_name} annual report",
            exact_phrases=[org_name],
            filetype="pdf"
        ))
        
        queries.append(SearchQuery(
            query=f"{org_name} investor presentation",
            exact_phrases=[org_name],
            filetype="pdf"
        ))
        
        # Press releases
        queries.append(SearchQuery(
            query=f"{org_name} press release",
            exact_phrases=[org_name],
            after_date="2023-01-01"
        ))
        
        # Crunchbase
        queries.append(SearchQuery(
            query=f"{org_name}",
            exact_phrases=[org_name],
            site="crunchbase.com"
        ))
        
        # Context-specific
        if context:
            queries.append(SearchQuery(
                query=f"{org_name} {context}",
                exact_phrases=[org_name]
            ))
        
        return queries
    
    @staticmethod
    def create_event_query(event_name: str, date_hint: Optional[str] = None) -> List[SearchQuery]:
        """
        Create search queries optimized for finding event information.
        
        Args:
            event_name: Event name
            date_hint: Optional date or year hint
            
        Returns:
            List of SearchQuery objects for comprehensive event search
        """
        queries = []
        
        # General event query
        queries.append(SearchQuery(
            query=f"{event_name}",
            exact_phrases=[event_name]
        ))
        
        # News coverage
        queries.append(SearchQuery(
            query=f"{event_name} news",
            exact_phrases=[event_name]
        ))
        
        # Official site
        queries.append(SearchQuery(
            query=f"{event_name} official",
            exact_phrases=[event_name]
        ))
        
        # Reports and summaries (PDF)
        queries.append(SearchQuery(
            query=f"{event_name} report",
            exact_phrases=[event_name],
            filetype="pdf"
        ))
        
        # Presentations (PDF/PPT)
        queries.append(SearchQuery(
            query=f"{event_name} presentation",
            exact_phrases=[event_name],
            filetype="pdf"
        ))
        
        queries.append(SearchQuery(
            query=f"{event_name} presentation",
            exact_phrases=[event_name],
            filetype="ppt"
        ))
        
        # Date-specific search
        if date_hint:
            queries.append(SearchQuery(
                query=f"{event_name} {date_hint}",
                exact_phrases=[event_name]
            ))
        
        return queries
    
    @staticmethod
    def create_document_query(topic: str, doc_types: Optional[List[str]] = None) -> List[SearchQuery]:
        """
        Create queries specifically for finding documents (PDFs, presentations, reports).
        
        Args:
            topic: Search topic
            doc_types: List of document types to search for (pdf, doc, ppt, xls)
            
        Returns:
            List of SearchQuery objects for document search
        """
        if doc_types is None:
            doc_types = ["pdf", "ppt", "doc"]
        
        queries = []
        
        for doc_type in doc_types:
            # General document search
            queries.append(SearchQuery(
                query=topic,
                filetype=doc_type
            ))
            
            # Reports
            queries.append(SearchQuery(
                query=f"{topic} report",
                filetype=doc_type
            ))
            
            # Presentations
            if doc_type in ["pdf", "ppt", "pptx"]:
                queries.append(SearchQuery(
                    query=f"{topic} presentation",
                    filetype=doc_type
                ))
            
            # Research papers
            if doc_type == "pdf":
                queries.append(SearchQuery(
                    query=f"{topic} research",
                    filetype=doc_type,
                    site="edu"  # Academic sites
                ))
        
        return queries
    
    
    @staticmethod
    def create_wildcard_query(partial_name: str, category: Optional[str] = None) -> SearchQuery:
        """
        Create a wildcard query when you don't recall complete information.
        
        Args:
            partial_name: Partial name or keyword
            category: Optional category (person, company, event, etc.)
            
        Returns:
            SearchQuery with wildcard pattern
        """
        query_str = f"{partial_name}"
        if category:
            query_str = f"{category} {partial_name}"
        
        return SearchQuery(query=query_str)

    # ========== OSINT DORK BUILDERS ==========

    @staticmethod
    def create_osint_person_dorks(name: str, domain: Optional[str] = None, context: Optional[str] = None) -> List[SearchQuery]:
        """
        Create comprehensive OSINT-style Google dorks for a PERSON.
        Uses full operator set: site:, filetype:, intext:, intitle:, inurl:

        Args:
            name: Person's name
            domain: Optional company/organization domain (e.g., example.com)
            context: Optional context (company name, role, location)

        Returns:
            List of SearchQuery objects for OSINT person investigation
        """
        queries: List[SearchQuery] = []

        # === EMAIL DISCOVERY ===
        if domain:
            # Domain-wide email search
            queries.append(SearchQuery(query=f'"{name}" "email" OR "contact"', site=domain, site_optional=True))
            # Specific email pattern hunting
            queries.append(SearchQuery(query=f'"*@{domain}"', site=domain, site_optional=True))
            # Document-based emails
            queries.append(SearchQuery(query=f'"{name}" "email" OR "contact"', site=domain, site_optional=True, filetype="pdf"))
            # Outside mentions (emails on other sites)
            queries.append(SearchQuery(query=f'intext:"@{domain}" "{name}"', excluded_words=[f"site:{domain}"]))

        # === PHONE & ADDRESS DISCOVERY ===
        queries.append(SearchQuery(query=f'"{name}" "phone" OR "tel" OR "mobile" OR "cell"'))
        queries.append(SearchQuery(query=f'"{name}" "address" OR "location" OR "office"'))
        if domain:
            queries.append(SearchQuery(query=f'"{name}" "phone" OR "tel" OR "address"', site=domain, site_optional=True))

        # === SOCIAL PROFILE DISCOVERY ===
        # LinkedIn
        queries.append(SearchQuery(query=f'"{name}"', site="linkedin.com", inurl="in"))
        queries.append(SearchQuery(query=f'"{name}"', site="linkedin.com", inurl="pub"))
        # Twitter/X
        queries.append(SearchQuery(query=f'"{name}"', site="x.com"))
        queries.append(SearchQuery(query=f'"{name}"', site="twitter.com"))
        # Facebook
        queries.append(SearchQuery(query=f'"{name}"', site="facebook.com"))
        # GitHub (for tech persons)
        queries.append(SearchQuery(query=f'"{name}"', site="github.com"))
        # Instagram
        queries.append(SearchQuery(query=f'"{name}"', site="instagram.com"))

        # === DOCUMENT DISCOVERY ===
        # PDF documents (CV, reports, papers)
        queries.append(SearchQuery(query=f'"{name}" CV OR resume OR curriculum', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{name}"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{name}"', filetype="doc"))
        queries.append(SearchQuery(query=f'"{name}"', filetype="docx"))
        # Spreadsheets (employee lists, directories)
        queries.append(SearchQuery(query=f'"{name}"', filetype="xlsx"))
        queries.append(SearchQuery(query=f'"{name}"', filetype="csv"))
        # Presentations
        queries.append(SearchQuery(query=f'"{name}"', filetype="pptx"))
        queries.append(SearchQuery(query=f'"{name}"', filetype="ppt"))

        # === SENSITIVE DOCUMENT PATTERNS ===
        queries.append(SearchQuery(query=f'"{name}" "confidential" OR "internal" OR "private"', filetype="pdf"))
        queries.append(SearchQuery(query=f'intitle:"index of" "{name}"'))

        # === DIRECTORY/STAFF LISTINGS ===
        queries.append(SearchQuery(query=f'"{name}" "staff" OR "team" OR "directory" OR "employee"'))
        if domain:
            queries.append(SearchQuery(query=f'"{name}" "team" OR "staff" OR "about"', site=domain, site_optional=True))

        # === PROFESSIONAL CONTEXT ===
        queries.append(SearchQuery(query=f'"{name}" CEO OR CFO OR CTO OR director OR manager OR founder'))
        queries.append(SearchQuery(query=f'"{name}" biography OR profile OR background'))
        if context:
            queries.append(SearchQuery(query=f'"{name}" "{context}"'))

        # === THIRD-PARTY DIRECTORIES ===
        queries.append(SearchQuery(query=f'"{name}"', site="whitepages.com"))
        queries.append(SearchQuery(query=f'"{name}"', site="spokeo.com"))
        queries.append(SearchQuery(query=f'"{name}"', site="pipl.com"))

        # === PASTE SITES (leaked data) ===
        queries.append(SearchQuery(query=f'"{name}"', site="pastebin.com"))

        # === NEWS & PRESS ===
        queries.append(SearchQuery(query=f'"{name}" news OR press OR interview', after_date="2022-01-01"))

        return queries

    @staticmethod
    def create_osint_company_dorks(company_name: str, domain: Optional[str] = None) -> List[SearchQuery]:
        """
        Create comprehensive OSINT-style Google dorks for a COMPANY.
        Uses full operator set for corporate intelligence.

        Args:
            company_name: Company/organization name
            domain: Optional company domain (e.g., example.com)

        Returns:
            List of SearchQuery objects for OSINT company investigation
        """
        queries: List[SearchQuery] = []

        # === EMAIL DISCOVERY ===
        if domain:
            queries.append(SearchQuery(query=f'"{company_name}" "email" OR "contact"', site=domain, site_optional=True))
            queries.append(SearchQuery(query=f'"*@{domain}"'))
            queries.append(SearchQuery(query=f'intext:"@{domain}"', excluded_words=[f"site:{domain}"]))
            queries.append(SearchQuery(query=f'"email" OR "contact"', site=domain, site_optional=True, filetype="pdf"))

        # === PHONE & ADDRESS ===
        queries.append(SearchQuery(query=f'"{company_name}" "phone" OR "tel" OR "address" OR "headquarters"'))
        if domain:
            queries.append(SearchQuery(query=f'"phone" OR "tel" OR "contact"', site=domain, site_optional=True))

        # === EMPLOYEE DIRECTORIES & STAFF ===
        if domain:
            queries.append(SearchQuery(query=f'"employee" OR "staff" OR "team"', site=domain, site_optional=True, filetype="xlsx"))
            queries.append(SearchQuery(query=f'"employee" OR "staff"', site=domain, site_optional=True, filetype="csv"))
            queries.append(SearchQuery(query=f'"team" OR "leadership" OR "management"', site=domain, site_optional=True))
        queries.append(SearchQuery(query=f'"{company_name}" "employee" OR "staff" OR "directory"', filetype="xlsx"))
        queries.append(SearchQuery(query=f'"{company_name}" "employee" OR "staff"', filetype="csv"))

        # === INTERNAL DOCUMENTS ===
        queries.append(SearchQuery(query=f'"{company_name}" "confidential" OR "internal"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{company_name}" "confidential" OR "internal"', filetype="pptx"))
        queries.append(SearchQuery(query=f'intitle:"index of" "{company_name}"'))
        if domain:
            queries.append(SearchQuery(query=f'intitle:"index of"', site=domain, site_optional=True))

        # === FINANCIAL & REPORTS ===
        queries.append(SearchQuery(query=f'"{company_name}" "annual report"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{company_name}" "investor presentation"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{company_name}" "financial statement" OR "10-K" OR "10-Q"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{company_name}" "quarterly report"', filetype="pdf"))

        # === GOVERNANCE & LEADERSHIP ===
        queries.append(SearchQuery(query=f'"{company_name}" "board of directors" OR "management team"'))
        queries.append(SearchQuery(query=f'"{company_name}" CEO OR CFO OR CTO OR "executive"'))
        if domain:
            queries.append(SearchQuery(query=f'"about" OR "leadership" OR "team"', site=domain, site_optional=True, inurl="about"))

        # === PRESS & NEWS ===
        queries.append(SearchQuery(query=f'"{company_name}" "press release"', after_date="2023-01-01"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="reuters.com"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="bloomberg.com"))

        # === BUSINESS DIRECTORIES ===
        queries.append(SearchQuery(query=f'"{company_name}"', site="crunchbase.com"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="linkedin.com", inurl="company"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="glassdoor.com"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="yellowpages.com"))
        queries.append(SearchQuery(query=f'"{company_name}"', site="dnb.com"))

        # === PASTE SITES (leaked data) ===
        if domain:
            queries.append(SearchQuery(query=f'"{domain}"', site="pastebin.com"))

        return queries

    @staticmethod
    def create_osint_org_dorks(org_name: str, domain: Optional[str] = None) -> List[SearchQuery]:
        """
        Create OSINT dorks for a generic ORGANIZATION (non-profit, government, etc.).

        Args:
            org_name: Organization name
            domain: Optional organization domain

        Returns:
            List of SearchQuery objects
        """
        queries: List[SearchQuery] = []

        # === CONTACT INFO ===
        queries.append(SearchQuery(query=f'"{org_name}" "email" OR "contact" OR "phone"'))
        if domain:
            queries.append(SearchQuery(query=f'"contact" OR "email"', site=domain, site_optional=True))
            queries.append(SearchQuery(query=f'"*@{domain}"'))

        # === STAFF & LEADERSHIP ===
        queries.append(SearchQuery(query=f'"{org_name}" "staff" OR "team" OR "board" OR "director"'))
        if domain:
            queries.append(SearchQuery(query=f'"team" OR "staff" OR "about"', site=domain, site_optional=True))

        # === DOCUMENTS ===
        queries.append(SearchQuery(query=f'"{org_name}"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{org_name}" "annual report" OR "report"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{org_name}"', filetype="xlsx"))

        # === INTERNAL/SENSITIVE ===
        queries.append(SearchQuery(query=f'"{org_name}" "internal" OR "confidential"', filetype="pdf"))
        queries.append(SearchQuery(query=f'intitle:"index of" "{org_name}"'))

        # === PUBLIC RECORDS ===
        queries.append(SearchQuery(query=f'"{org_name}" "public record" OR "filing" OR "registration"'))

        # === SOCIAL ===
        queries.append(SearchQuery(query=f'"{org_name}"', site="linkedin.com"))
        queries.append(SearchQuery(query=f'"{org_name}"', site="facebook.com"))
        queries.append(SearchQuery(query=f'"{org_name}"', site="twitter.com"))

        return queries

    @staticmethod
    def create_osint_institution_dorks(institution_name: str, domain: Optional[str] = None) -> List[SearchQuery]:
        """
        Create OSINT dorks for an INSTITUTION (university, hospital, government agency).

        Args:
            institution_name: Institution name
            domain: Optional institution domain (often .edu, .gov)

        Returns:
            List of SearchQuery objects
        """
        queries: List[SearchQuery] = []

        # === CONTACT & DIRECTORY ===
        queries.append(SearchQuery(query=f'"{institution_name}" "directory" OR "staff" OR "faculty"'))
        if domain:
            queries.append(SearchQuery(query=f'"directory" OR "faculty" OR "staff"', site=domain, site_optional=True))
            queries.append(SearchQuery(query=f'"contact" OR "email"', site=domain, site_optional=True))
            queries.append(SearchQuery(query=f'"*@{domain}"'))

        # === DOCUMENTS ===
        queries.append(SearchQuery(query=f'"{institution_name}"', filetype="pdf"))
        queries.append(SearchQuery(query=f'"{institution_name}" "report" OR "publication"', filetype="pdf"))
        if domain:
            queries.append(SearchQuery(query=f'"report" OR "publication"', site=domain, site_optional=True, filetype="pdf"))

        # === SENSITIVE/INTERNAL ===
        queries.append(SearchQuery(query=f'"{institution_name}" "internal" OR "confidential"', filetype="pdf"))
        queries.append(SearchQuery(query=f'intitle:"index of"', site=domain if domain else None))

        # === SPREADSHEETS (employee/student lists) ===
        queries.append(SearchQuery(query=f'"{institution_name}"', filetype="xlsx"))
        queries.append(SearchQuery(query=f'"{institution_name}"', filetype="csv"))

        # === SOCIAL ===
        queries.append(SearchQuery(query=f'"{institution_name}"', site="linkedin.com"))
        queries.append(SearchQuery(query=f'"{institution_name}"', site="facebook.com"))

        return queries

    @staticmethod
    def create_osint_role_dorks(role: str, organization: Optional[str] = None, domain: Optional[str] = None) -> List[SearchQuery]:
        """
        Create OSINT dorks for a ROLE/POSITION (e.g., "CEO", "Minister of Finance").

        Args:
            role: Role or position title
            organization: Optional organization context
            domain: Optional domain

        Returns:
            List of SearchQuery objects
        """
        queries: List[SearchQuery] = []

        base_query = f'"{role}"' if not organization else f'"{role}" "{organization}"'

        # === PERSON DISCOVERY ===
        queries.append(SearchQuery(query=f'{base_query}'))
        queries.append(SearchQuery(query=f'{base_query}', site="linkedin.com"))
        queries.append(SearchQuery(query=f'{base_query} biography OR profile'))

        # === CONTACT INFO ===
        queries.append(SearchQuery(query=f'{base_query} "email" OR "contact" OR "phone"'))
        if domain:
            queries.append(SearchQuery(query=f'"{role}" "contact" OR "email"', site=domain, site_optional=True))

        # === DOCUMENTS ===
        queries.append(SearchQuery(query=f'{base_query}', filetype="pdf"))
        queries.append(SearchQuery(query=f'{base_query} "announcement" OR "appointment"'))

        # === NEWS ===
        queries.append(SearchQuery(query=f'{base_query} news OR interview', after_date="2023-01-01"))

        return queries

    @staticmethod
    def create_osint_dorks_by_type(
        name: str,
        target_type: str,
        domain: Optional[str] = None,
        context: Optional[str] = None
    ) -> List[SearchQuery]:
        """
        Unified entry point to create OSINT dorks based on target type.

        Args:
            name: Target name (person, company, etc.)
            target_type: One of "person", "company", "org", "institution", "role", "auto"
            domain: Optional domain for targeted searches
            context: Optional additional context

        Returns:
            List of SearchQuery objects
        """
        tt = (target_type or "auto").lower().strip()

        if tt == "person":
            return AdvancedSearchBuilder.create_osint_person_dorks(name, domain, context)
        elif tt == "company":
            return AdvancedSearchBuilder.create_osint_company_dorks(name, domain)
        elif tt == "org":
            return AdvancedSearchBuilder.create_osint_org_dorks(name, domain)
        elif tt == "institution":
            return AdvancedSearchBuilder.create_osint_institution_dorks(name, domain)
        elif tt == "role":
            return AdvancedSearchBuilder.create_osint_role_dorks(name, context, domain)
        else:
            # "auto" or unknown: combine general queries from person + org
            queries: List[SearchQuery] = []
            # Basic universal queries
            queries.append(SearchQuery(query=f'"{name}"'))
            queries.append(SearchQuery(query=f'"{name}"', site="linkedin.com"))
            queries.append(SearchQuery(query=f'"{name}" "email" OR "contact" OR "phone"'))
            queries.append(SearchQuery(query=f'"{name}"', filetype="pdf"))
            queries.append(SearchQuery(query=f'"{name}"', filetype="xlsx"))
            queries.append(SearchQuery(query=f'"{name}" biography OR profile OR about'))
            if domain:
                queries.append(SearchQuery(query=f'"{name}"', site=domain))
                queries.append(SearchQuery(query=f'"*@{domain}"'))
            if context:
                queries.append(SearchQuery(query=f'"{name}" "{context}"'))
            return queries
