# intelligent_sheet_router.py
"""
Intelligent sheet routing - maps user questions to relevant data sources.
Uses keyword matching + LLM intent classification + role-based filtering.
"""

from typing import List, Tuple, Optional
from sheet_registry import SHEET_REGISTRY, SheetSource, DataCategory, get_available_sheets
import re
from enum import Enum


class UserRole(Enum):
    """User roles for access control"""
    CEO = "CEO"
    CFO_GM = "CFO/GM"
    SALES_GM = "Sales GM"
    SALES_CONSULTANT = "Sales Consultant"
    SERVICE_MANAGER = "Service Manager"
    FINANCE = "Finance"
    CONSULTANT = "Consultant"
    BACK_OFFICE = "Back Office"


# ============================================================================
# ROLE-BASED ACCESS CONTROL
# ============================================================================

ROLE_SHEET_ACCESS = {
    UserRole.CEO: [
        # Full access to all sheets
        "sales_target", "sales_jan", "sales_feb", "sales_mar", "sales_apr",
        "sales_may", "sales_jun", "sales_jul", "sales_aug", "sales_sep",
        "sales_oct", "sales_nov", "demo_stock",
        "service_view", "incentive",
        "crm_dashboard", "crm_pes",'mis_report'
        "debtors",
        "skyline_otc", "skyline_insurance", "skyline_starease", "skyline_ora","attendance" 
    ],
    
    UserRole.CFO_GM: [
        "debtors", "skyline_otc", "skyline_insurance", "skyline_starease",'mis_report'
        "skyline_ora", "sales_target", "service_view", "demo_stock","attendance" 
    ],
    
    UserRole.SALES_GM: [
        "sales_target", "sales_jan", "sales_feb", "sales_mar", "sales_apr",'mis_report'
        "sales_may", "sales_jun", "sales_jul", "sales_aug", "sales_sep",
        "sales_oct", "sales_nov", "demo_stock", "crm_dashboard","attendance" 
    ],
    
    UserRole.SALES_CONSULTANT: [
        "sales_jan", "sales_feb", "sales_mar", "sales_apr", "sales_may",
        "sales_jun", "sales_jul", "sales_aug", "sales_sep", "sales_oct",
        "sales_nov", "demo_stock"
    ],
    
    UserRole.SERVICE_MANAGER: [
        "service_view", "incentive", "crm_dashboard"
    ],
    
    UserRole.FINANCE: [
        "debtors", "sales_target", "skyline_otc", "skyline_insurance",'mis_report'
    ],
    
    UserRole.CONSULTANT: [
        "sales_jan", "sales_feb", "sales_mar", "sales_apr", "sales_may",
        "sales_jun", "sales_jul", "sales_aug", "sales_sep", "sales_oct",
        "sales_nov"
    ],
    
    UserRole.BACK_OFFICE: [
        "debtors", "crm_dashboard", "crm_pes", "incentive"
    ],
}


# ============================================================================
# INTELLIGENT ROUTER
# ============================================================================

class IntelligentSheetRouter:
    """
    Maps user questions to relevant data sources using:
    1. Keyword matching (fast, deterministic)
    2. Semantic similarity (optional, requires LLM)
    3. Role-based filtering
    """
    
    def __init__(self, llm=None, user_role: UserRole = UserRole.CEO):
        """
        Args:
            llm: Optional LLM for intent classification (Groq)
            user_role: User's role for access control
        """
        self.llm = llm
        self.user_role = user_role
        self.allowed_sheets = ROLE_SHEET_ACCESS.get(user_role, [])
        self.registry = SHEET_REGISTRY
    
    def route(self, user_query: str) -> List[SheetSource]:
        """
        Main routing function - returns list of relevant sheets.
        """
        
        # Step 1: Try keyword matching (fast)
        sheets = self._keyword_match(user_query)
        
        if sheets:
            return sheets
        
        # Step 2: If no match, try LLM intent classification (slow)
        if self.llm:
            sheets = self._llm_intent_match(user_query)
            if sheets:
                return sheets
        
        # ✅ Step 3: Check if query is irrelevant (don't return any sheets)
        query_lower = user_query.lower()
        
        # If query has business keywords but no match, return empty
        business_keywords = ['who', 'what', 'how many', 'show', 'list', 'get', 'find']
        if any(kw in query_lower for kw in business_keywords):
            print(f"⚠️ No matching sheets found for: {user_query}")
            return []  # ✅ Return empty instead of all sheets
        
        # Step 4: Fallback for generic queries - return nothing
        print(f"⚠️ No matching sheets found for: {user_query}")
        return []  # ✅ Changed from self._get_accessible_sheets()
    
    def _keyword_match(self, query: str) -> List[SheetSource]:
        """
        Find sheets by keyword matching against:
        - Sheet keywords
        - Sheet description
        - Category name
        
        Returns sheets ordered by relevance score
        """
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = {}
        
        for sheet_id, sheet in self.registry.items():
            # Check if user has access
            if sheet_id not in self.allowed_sheets:
                continue
            
            # Check if sheet is available
            if not sheet.is_available():
                continue
            
            score = 0
            
            # ============ KEYWORD SCORING ============
            for keyword in sheet.keywords:
                if keyword in query_lower:
                    score += 10  # Exact keyword match
                elif any(word in query_words for word in keyword.split()):
                    score += 5   # Partial keyword match
            
            # ============ DESCRIPTION SCORING ============
            if sheet.description:
                desc_lower = sheet.description.lower()
                desc_words = desc_lower.split()
                
                # Check how many query words appear in description
                matching_words = len(query_words & set(desc_words))
                score += matching_words * 2
            
            # ============ CATEGORY SCORING ============
            if sheet.category:
                category_lower = sheet.category.value.lower()
                if any(word in category_lower for word in query_words):
                    score += 3
            
            # ============ STORE SCORE ============
            if score > 0:
                scores[sheet_id] = (score, sheet)
        
        # Sort by score (highest first)
        sorted_sheets = sorted(
            scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        
        # Return sheets only if score is meaningful
        if sorted_sheets and sorted_sheets[0][1][0] >= 3:
            return [sheet for _, (score, sheet) in sorted_sheets]
        
        return []
    
    def _llm_intent_match(self, user_query: str) -> List[SheetSource]:
        """
        Use LLM to classify intent when keywords don't match.
        
        This is more expensive (API call) but handles complex queries.
        """
        
        accessible_sheets = [
            f"- {s.sheet_id}: {s.name} ({s.description})"
            for s in self._get_accessible_sheets()
        ]
        
        prompt = f"""
You are a data assistant helping users find the right data sources.

User Question: "{user_query}"

Available Data Sources:
{chr(10).join(accessible_sheets)}

Which data source(s) are most relevant for this question?

Return ONLY the sheet IDs (comma-separated), nothing else.
Example: sales_target,sales_jan,sales_feb

If none are relevant, return: NONE
"""
        
        try:
            response = self.llm.complete(prompt)
            sheet_ids = response.text.strip().upper()
            
            if sheet_ids == "NONE":
                return []
            
            # Parse response
            sheet_ids = [s.strip().lower() for s in sheet_ids.split(",")]
            
            result = []
            for sheet_id in sheet_ids:
                if sheet_id in self.registry and sheet_id in self.allowed_sheets:
                    result.append(self.registry[sheet_id])
            
            return result
            
        except Exception as e:
            print(f"❌ LLM routing failed: {str(e)}")
            return []
    
    def _get_accessible_sheets(self) -> List[SheetSource]:
        """Get all sheets user can access (filtered by role)"""
        sheets = []
        for sheet_id in self.allowed_sheets:
            if sheet_id in self.registry:
                sheet = self.registry[sheet_id]
                if sheet.is_available():
                    sheets.append(sheet)
        return sheets
    
    def explain_routing(self, user_query: str) -> str:
        """
        Explain why certain sheets were selected.
        Useful for debugging and transparency.
        """
        
        sheets = self.route(user_query)
        
        if not sheets:
            return f"❌ No matching sheets found for: '{user_query}'"
        
        explanation = f"📊 Query: '{user_query}'\n\n"
        explanation += f"✅ Matching sheets ({len(sheets)}):\n"
        
        for sheet in sheets:
            explanation += f"  • {sheet.name}\n"
            explanation += f"    ID: {sheet.sheet_id}\n"
            explanation += f"    Category: {sheet.category.value if sheet.category else 'N/A'}\n"
        
        return explanation


# ============================================================================
# DEMO / TESTING
# ============================================================================

def test_routing():
    """Test the router with sample queries"""
    
    print("\n" + "="*70)
    print("TESTING INTELLIGENT SHEET ROUTER")
    print("="*70)
    
    router = IntelligentSheetRouter(user_role=UserRole.CEO)
    
    test_queries = [
        "What is the sales target achievement MTD?",
        "Show me January sales by consultant",
        "How many service requests pending?",
        "Customer satisfaction scores",
        "Receivables aging analysis",
        "Demo vehicle inventory status",
        "March and April sales comparison",
        "Show OTC transactions",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        sheets = router.route(query)
        
        if sheets:
            print(f"   ✅ Found {len(sheets)} sheet(s):")
            for sheet in sheets:
                print(f"      • {sheet.name} ({sheet.sheet_id})")
        else:
            print("   ❌ No matching sheets")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_routing()
