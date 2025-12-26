# router.py
from enum import Enum
from typing import Dict, List

class QueryType(Enum):
    SQL_ONLY = "sql"
    DOCUMENT_ONLY = "document"
    HYBRID = "hybrid"

class QueryRouter:
    def __init__(self, llm):
        self.llm = llm
        # Keywords that indicate DATABASE query
        self.keywords_sql = ["receivables", "aging", "database", "table"]
        # Keywords that indicate DOCUMENT query (HIGHER PRIORITY)
        self.keywords_doc = ["uploaded", "document", "file", "sheet", "attendance", "excel", "pdf", "show me", "summarize"]
    
    def route(self, query: str, has_uploaded_docs: bool) -> QueryType:
        """Determine query routing based on content and context"""
        query_lower = query.lower()
        
        # If no documents uploaded, must use SQL
        if not has_uploaded_docs:
            return QueryType.SQL_ONLY
        
        # Count keyword matches
        doc_score = sum(1 for kw in self.keywords_doc if kw in query_lower)
        sql_score = sum(1 for kw in self.keywords_sql if kw in query_lower)
        
        # Priority rules:
        # 1. Document keywords take precedence
        if doc_score > 0:
            return QueryType.DOCUMENT_ONLY
        
        # 2. SQL keywords without doc keywords
        if sql_score > 0:
            return QueryType.SQL_ONLY
        
        # 3. Check for comparison/hybrid queries
        if any(word in query_lower for word in ["compare", "difference", "match", "versus", "vs"]):
            return QueryType.HYBRID
        
        # 4. Default: If documents exist, search them first
        return QueryType.DOCUMENT_ONLY
