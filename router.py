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
        
        # ✅ NEW: If files uploaded, ALWAYS check them first
        if has_uploaded_docs:
            # Check for explicit pre-loaded sheet queries
            preloaded_keywords = [
                "sales target", "monthly sales", "january", "february", 
                "service throughput", "customer satisfaction", "receivables",
                "debtors", "skyline", "otc", "insurance"
            ]
            
            # If query mentions pre-loaded sheets, use those
            if any(kw in query_lower for kw in preloaded_keywords):
                return QueryType.SQL_ONLY  # Actually means "pre-loaded sheets"
            
            # Otherwise, query uploaded document
            return QueryType.DOCUMENT_ONLY
        
        # No uploaded docs - use pre-loaded sheets
        return QueryType.SQL_ONLY
