# sheet_registry.py
"""
Centralized registry of all pre-loaded data sources.
No hardcoding - all paths come from environment variables.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class DataCategory(Enum):
    """Classification of data sources"""
    SALES = "Sales"
    SERVICE = "Service"
    CRM = "CRM"
    ACCOUNTS = "Accounts"
    SKYLINE = "Skyline System"
    DEMO = "Demo Stock"
    INCENTIVE = "Incentive"
    ATTENDANCE = "Attendance"


@dataclass
class SheetSource:
    """Metadata for a pre-loaded Excel sheet"""
    
    sheet_id: str                    # Unique identifier (e.g., "sales_target")
    name: str                        # Display name for UI
    file_path: str                   # Full path to Excel file
    sheet_name: Optional[str] = None # Sheet name within file (if multiple)
    category: Optional[DataCategory] = None
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    refresh_frequency: str = "Manual"
    columns_info: Dict[str, str] = field(default_factory=dict)
    icon: str = "📊"                 # Emoji for UI
    
    def is_available(self) -> bool:
        """Check if file exists"""
        return self.file_path and os.path.exists(self.file_path)
    
    def __repr__(self):
        return f"Sheet({self.name}, available={self.is_available()})"


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def _get_path(env_var: str, description: str = "") -> Optional[str]:
    """Safely get path from environment"""
    path = os.getenv(env_var)
    if path and not os.path.exists(path):
        print(f"⚠️ {description} not found: {path}")
    return path


# ============================================================================
# SALES DATA SOURCES
# ============================================================================

SALES_SOURCES = {
    "sales_target": SheetSource(
        sheet_id="sales_target",
        name="Sales Target & Actual (MTD)",
        file_path=_get_path("EXCEL_SALES_TARGET_PATH", "Sales Target"),
        sheet_name="Sheet1",
        category=DataCategory.SALES,
        keywords=["target", "achievement", "mtd", "sales target", "actual", "retail", "scc"],
        description="Monthly sales targets vs actual achievement - Retail and SCC divisions",
        refresh_frequency="Daily",
        columns_info={
            "Model": "Vehicle model name",
            "Target": "Monthly target units",
            "Actual": "Actual sales units",
            "Achievement%": "MTD achievement percentage",
            "Category": "Retail or SCC"
        },
        icon="🎯"
    ),
    
    "sales_jan": SheetSource(
        sheet_id="sales_jan",
        name="January Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_JAN_PATH", "January Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["january", "jan", "sales", "retail", "scc", "month"],
        description="Detailed January sales data - vehicle details, consultants, amounts",
        refresh_frequency="Monthly",
        columns_info={
            "ConsultantName": "Sales consultant name",
            "VehicleModel": "Vehicle model",
            "SalesAmount": "Sale value in currency",
            "Date": "Sale date"
        },
        icon="📅"
    ),
    
    "sales_feb": SheetSource(
        sheet_id="sales_feb",
        name="February Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_FEB_PATH", "February Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["february", "feb", "sales"],
        description="Detailed February sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_mar": SheetSource(
        sheet_id="sales_mar",
        name="March Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_MAR_PATH", "March Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["march", "mar", "sales"],
        description="Detailed March sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_apr": SheetSource(
        sheet_id="sales_apr",
        name="April Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_APR_PATH", "April Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["april", "apr", "sales"],
        description="Detailed April sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_may": SheetSource(
        sheet_id="sales_may",
        name="May Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_MAY_PATH", "May Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["may", "sales"],
        description="Detailed May sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_jun": SheetSource(
        sheet_id="sales_jun",
        name="June Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_JUN_PATH", "June Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["june", "jun", "sales"],
        description="Detailed June sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_jul": SheetSource(
        sheet_id="sales_jul",
        name="July Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_JUL_PATH", "July Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["july", "jul", "sales"],
        description="Detailed July sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_aug": SheetSource(
        sheet_id="sales_aug",
        name="August Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_AUG_PATH", "August Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["august", "aug", "sales"],
        description="Detailed August sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_sep": SheetSource(
        sheet_id="sales_sep",
        name="September Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_SEP_PATH", "September Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["september", "sep", "sales"],
        description="Detailed September sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_oct": SheetSource(
        sheet_id="sales_oct",
        name="October Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_OCT_PATH", "October Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["october", "oct", "sales"],
        description="Detailed October sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "sales_nov": SheetSource(
        sheet_id="sales_nov",
        name="November Retail & SCC Sales",
        file_path=_get_path("EXCEL_SALES_NOV_PATH", "November Sales"),
        sheet_name="Sales",
        category=DataCategory.SALES,
        keywords=["november", "nov", "sales"],
        description="Detailed November sales data",
        refresh_frequency="Monthly",
        icon="📅"
    ),
    
    "demo_stock": SheetSource(
        sheet_id="demo_stock",
        name="Demo Stock Report",
        file_path=_get_path("EXCEL_DEMO_STOCK_PATH", "Demo Stock"),
        sheet_name="Stock",
        category=DataCategory.DEMO,
        keywords=["demo", "stock", "inventory", "demo vehicle", "unpaid demo", "demo age"],
        description="Demo vehicle inventory status and aging",
        refresh_frequency="Weekly",
        columns_info={
            "VehicleID": "Demo vehicle unique identifier",
            "Model": "Vehicle model name",
            "AgeInDays": "Days in inventory",
            "Status": "Current status (Saleable, On Hold, etc)"
        },
        icon="🚗"
    ),
}


# ============================================================================
# SERVICE DATA SOURCES
# ============================================================================

SERVICE_SOURCES = {
    "service_view": SheetSource(
        sheet_id="service_view",
        name="Service Throughput",
        file_path=_get_path("EXCEL_SERVICE_VIEW_PATH", "Service View"),
        sheet_name="Service",
        category=DataCategory.SERVICE,
        keywords=["service", "throughput", "service request", "request count", "completed"],
        description="Service request count, completion rate, and throughput metrics",
        refresh_frequency="Daily",
        columns_info={
            "Month": "Month of data",
            "TotalRequests": "Total service requests received",
            "CompletedRequests": "Completed service requests",
            "PendingRequests": "Requests still pending",
            "Throughput": "Throughput percentage"
        },
        icon="🔧"
    ),
    
    "incentive": SheetSource(
        sheet_id="incentive",
        name="Incentive Master",
        file_path=_get_path("EXCEL_INCENTIVE_PATH", "Incentive Master"),
        sheet_name="Incentive",
        category=DataCategory.INCENTIVE,
        keywords=["incentive", "bonus", "technician", "engineer", "commission"],
        description="Technician and engineer incentive structure",
        refresh_frequency="Monthly",
        icon="💰"
    ),
}


# ============================================================================
# CRM DATA SOURCES
# ============================================================================

CRM_SOURCES = {
    "crm_dashboard": SheetSource(
        sheet_id="crm_dashboard",
        name="CRM Dashboard & Satisfaction",
        file_path=_get_path("EXCEL_CRM_DASHBOARD_PATH", "CRM Dashboard"),
        sheet_name="Dashboard",
        category=DataCategory.CRM,
        keywords=["satisfaction", "customer satisfaction", "score", "csat", "crm", "feedback"],
        description="Customer satisfaction scores, CSAT metrics, and feedback data",
        refresh_frequency="Weekly",
        columns_info={
            "CustomerName": "Customer full name",
            "SatisfactionScore": "CSAT score (1-5 or 1-10)",
            "Feedback": "Customer feedback text",
            "Date": "Survey date"
        },
        icon="😊"
    ),
    
    "crm_pes": SheetSource(
        sheet_id="crm_pes",
        name="CRM PES Data",
        file_path=_get_path("EXCEL_CRM_PES_PATH", "CRM PES"),
        sheet_name="PES",
        category=DataCategory.CRM,
        keywords=["pes", "crm", "customer"],
        description="CRM PES (Product Education Support) data",
        refresh_frequency="Weekly",
        icon="📚"
    ),
}


# ============================================================================
# ACCOUNTS / FINANCE DATA SOURCES
# ============================================================================

ACCOUNTS_SOURCES = {
    "debtors": SheetSource(
        sheet_id="debtors",
        name="Debtors Aging Report",
        file_path=_get_path("EXCEL_DEBTORS_PATH", "Debtors"),
        sheet_name="Debtors",
        category=DataCategory.ACCOUNTS,
        keywords=["receivables", "debtors", "aging", "pending", "outstanding", "dues", "receivable"],
        description="Receivables aging analysis - pending amounts and aging buckets (0-30, 30-60, 60-90, 90+ days)",
        refresh_frequency="Monthly",
        columns_info={
            "PartyName": "Customer/Party name",
            "Amount": "Receivable amount",
            "DaysOutstanding": "Number of days overdue",
            "AgingBucket": "Aging category (0-30/30-60/60-90/90+)",
            "InvoiceDate": "Original invoice date"
        },
        icon="💳"
    ),
    
    # ✅ NEW: MIS Report
    "mis_report": SheetSource(
        sheet_id="mis_report",
        name="MIS Report 2025",
        file_path=_get_path("EXCEL_MIS_PATH", "MIS Report"),
        sheet_name=None,  # Will auto-detect or load first sheet
        category=DataCategory.ACCOUNTS,
        keywords=[
            "mis", "management information", "financial summary", 
            "consolidated report", "monthly report", "financial analysis",
            "profit loss", "p&l", "revenue", "expenses", "income statement",
            "balance sheet", "cash flow", "financial metrics", "kpi"
        ],
        description="Management Information System (MIS) - Consolidated financial reports, P&L, revenue analysis, expense tracking, and key performance indicators for 2025",
        refresh_frequency="Monthly",
        columns_info={
            "Month": "Reporting month",
            "Revenue": "Total revenue",
            "Expenses": "Total expenses",
            "Profit": "Net profit",
            "KPI": "Key performance indicators"
        },
        icon="📊"
    ),
}


# ============================================================================
# SKYLINE SYSTEM DATA SOURCES
# ============================================================================

SKYLINE_SOURCES = {
    "skyline_otc": SheetSource(
        sheet_id="skyline_otc",
        name="Skyline OTC",
        file_path=_get_path("EXCEL_SKYLINE_OTC_PATH", "Skyline OTC"),
        sheet_name="OTC",
        category=DataCategory.SKYLINE,
        keywords=["otc", "over the counter", "skyline", "vas", "revenue"],
        description="Over-the-counter transactions in SKYLINE system",
        refresh_frequency="Daily",
        icon="🏦"
    ),
    
    "skyline_insurance": SheetSource(
        sheet_id="skyline_insurance",
        name="Skyline Insurance",
        file_path=_get_path("EXCEL_SKYLINE_INSURANCE_PATH", "Skyline Insurance"),
        sheet_name="Insurance",
        category=DataCategory.SKYLINE,
        keywords=["insurance", "skyline"],
        description="Insurance transactions in SKYLINE system",
        refresh_frequency="Daily",
        icon="📋"
    ),
    
    "skyline_starease": SheetSource(
        sheet_id="skyline_starease",
        name="Skyline StarEase",
        file_path=_get_path("EXCEL_SKYLINE_STAREASE_PATH", "Skyline StarEase"),
        sheet_name="StarEase",
        category=DataCategory.SKYLINE,
        keywords=["starease", "skyline"],
        description="StarEase transactions in SKYLINE system",
        refresh_frequency="Daily",
        icon="⭐"
    ),
    
    "skyline_ora": SheetSource(
        sheet_id="skyline_ora",
        name="Skyline ORA",
        file_path=_get_path("EXCEL_SKYLINE_ORA_PATH", "Skyline ORA"),
        sheet_name="ORA",
        category=DataCategory.SKYLINE,
        keywords=["ora", "skyline"],
        description="ORA transactions in SKYLINE system",
        refresh_frequency="Daily",
        icon="📊"
    ),
}

ATTENDANCE_SOURCES = {
    "attendance": SheetSource(
        sheet_id="attendance",
        name="Attendance Report",
        file_path=_get_path("ATTENDANCE_REPORT_PATH", "Attendance"),
        sheet_name=None,  # Will auto-detect first sheet
        category=DataCategory.ATTENDANCE,
        keywords=[
            "attendance", "absent", "leave", "leaves", "present", 
            "working days", "attendance report", "employee attendance",
            "who was absent", "who attended", "least attendance",
            "most attendance", "attendance record"
        ],
        description="Employee attendance records - present, absent, leaves, working days",
        refresh_frequency="Daily",
        columns_info={
            "EmployeeName": "Employee name",
            "Department": "Department/Team",
            "PresentDays": "Days present",
            "AbsentDays": "Days absent",
            "LeaveDays": "Leave days taken",
            "TotalWorkingDays": "Total working days in period"
        },
        icon="📅"
    ),
}
# ============================================================================
# MASTER REGISTRY
# ============================================================================

SHEET_REGISTRY = {
    **SALES_SOURCES,
    **SERVICE_SOURCES,
    **CRM_SOURCES,
    **ACCOUNTS_SOURCES,
    **SKYLINE_SOURCES,
    **ATTENDANCE_SOURCES
}


def get_sheets_by_category(category: DataCategory) -> List[SheetSource]:
    """Get all sheets in a category"""
    return [s for s in SHEET_REGISTRY.values() if s.category == category]


def get_available_sheets() -> List[SheetSource]:
    """Get only sheets that exist on disk"""
    return [s for s in SHEET_REGISTRY.values() if s.is_available()]


def print_registry_summary():
    """Print summary of all available sheets"""
    print("\n" + "="*70)
    print("SHEET REGISTRY SUMMARY")
    print("="*70)
    
    for category in DataCategory:
        sheets = get_sheets_by_category(category)
        if sheets:
            print(f"\n{category.value}:")
            for sheet in sheets:
                status = "✅" if sheet.is_available() else "❌"
                print(f"  {status} {sheet.name}")
                print(f"     ID: {sheet.sheet_id}")
                print(f"     Path: {sheet.file_path}")
    
    print("\n" + "="*70)
    print(f"Total sheets: {len(SHEET_REGISTRY)}")
    print(f"Available: {len(get_available_sheets())}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_registry_summary()
