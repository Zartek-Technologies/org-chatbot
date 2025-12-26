# hybrid_engine.py
from typing import Dict, Any
from router import QueryRouter, QueryType


class HybridQueryEngine:
    def __init__(self, sql_engine, doc_processor, llm, router):
        self.sql_engine = sql_engine
        self.doc_processor = doc_processor
        self.llm = llm
        self.router = router
    
    def query(self, user_query: str, session_id: str, has_docs: bool) -> str:
        """Execute hybrid query across SQL and documents"""
        route_type = self.router.route(user_query, has_docs)
        
        if route_type == QueryType.SQL_ONLY:
            source_msg = "🗄️ **Source: Database**\n\n"
            result = self._query_sql(user_query)
            return source_msg + result
        
        elif route_type == QueryType.DOCUMENT_ONLY:
            source_msg = "📄 **Source: Uploaded Documents**\n\n"
            result = self._query_documents(user_query, session_id)
            return source_msg + result
        
        else:  # HYBRID
            source_msg = "🔀 **Source: Database + Documents**\n\n"
            result = self._query_hybrid(user_query, session_id)
            return source_msg + result
    
    def _query_sql(self, query: str) -> str:
        """Query structured database"""
        try:
            response = self.sql_engine.query(query)
            return str(response)
        except Exception as e:
            return f"❌ Database query error: {str(e)}"
    
    def _query_documents(self, query: str, session_id: str) -> str:
        """Query uploaded documents"""
        try:
            response = self.doc_processor.query_documents(query, session_id)
            return str(response)
        except Exception as e:
            return f"❌ Document query error: {str(e)}"
    
    def _query_hybrid(self, query: str, session_id: str) -> str:
        """Combine SQL and document results"""
        sql_result = self._query_sql(query)
        doc_result = self._query_documents(query, session_id)
        
        synthesis_prompt = f"""
        User Question: {query}
        
        Database Results:
        {sql_result}
        
        Document Results:
        {doc_result}
        
        Synthesize a comprehensive answer combining both sources.
        Clearly indicate which information comes from which source.
        """
        
        response = self.llm.complete(synthesis_prompt)
        return response.text
