"""
Advanced search query builder with Google search operators.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SearchQuery:
    """Represents an advanced search query with operators."""
    query: str
    exact_phrases: Optional[List[str]] = None
    excluded_words: Optional[List[str]] = None
    site: Optional[str] = None
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
        parts = []
        
        # Base query
        if self.query:
            parts.append(self.query)
        
        # Exact phrase matches (quoted)
        for phrase in self.exact_phrases: # type: ignore
            parts.append(f'"{phrase}"')
        
        # Excluded words (with minus sign)
        for word in self.excluded_words: # type: ignore
            parts.append(f"-{word}")
        
        # Site restriction
        if self.site:
            parts.append(f"site:{self.site}")
        
        # File type filter
        if self.filetype:
            parts.append(f"filetype:{self.filetype}")
        
        # Title keyword
        if self.intitle:
            parts.append(f'intitle:"{self.intitle}"')
        
        # URL keyword
        if self.inurl:
            parts.append(f"inurl:{self.inurl}")
        
        # Date range filters
        if self.after_date:
            parts.append(f"after:{self.after_date}")
        if self.before_date:
            parts.append(f"before:{self.before_date}")
        
        query_string = " ".join(parts)
        logger.debug(f"Built search query: {query_string}")
        return query_string


class AdvancedSearchBuilder:
    """Build sophisticated search queries for targeted information retrieval."""
    
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
        
        # Wikipedia
        queries.append(SearchQuery(
            query=f"{name}",
            exact_phrases=[name],
            site="wikipedia.org"
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
        
        # Wikipedia
        queries.append(SearchQuery(
            query=f"{org_name}",
            exact_phrases=[org_name],
            site="wikipedia.org"
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
    def create_news_query(topic: str, recent_only: bool = True) -> List[SearchQuery]:
        """
        Create queries optimized for news articles.
        
        Args:
            topic: News topic
            recent_only: Whether to limit to recent articles
            
        Returns:
            List of SearchQuery objects for news search
        """
        news_sites = [
            "nytimes.com",
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "bbc.com",
            "theguardian.com",
            "apnews.com"
        ]
        
        queries = []
        
        # General news query
        query = SearchQuery(query=f"{topic} news")
        if recent_only:
            query.after_date = "2023-01-01"
        queries.append(query)
        
        # Major news sites
        for site in news_sites[:3]:  # Limit to top 3 to avoid too many queries
            query = SearchQuery(
                query=topic,
                site=site
            )
            if recent_only:
                query.after_date = "2023-01-01"
            queries.append(query)
        
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
