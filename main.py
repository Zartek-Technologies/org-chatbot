# main.py
import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uuid

# ✅ NEW IMPORTS
from intelligent_sheet_router import IntelligentSheetRouter, UserRole
from document_processor import DocumentProcessor
from hybrid_engine import HybridQueryEngine

load_dotenv()

st.set_page_config(
    page_title="Enterprise Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
# ✅ NEW: Store user role
if "user_role" not in st.session_state:
    st.session_state.user_role = UserRole.CEO  # Default to CEO, change based on login


@st.cache_resource
def init_system():
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1024
    )
    Settings.llm = llm
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Optional Database Connection
    sql_engine = None
    db_enabled = False
    
    try:
        db_user = os.getenv('POSTGRES_USER')
        db_password = os.getenv('POSTGRES_PASSWORD')
        db_host = os.getenv('POSTGRES_HOST')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DATABASE')
        
        if all([db_user, db_password, db_host, db_name]):
            engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            
            engine.connect()
            
            sql_database = SQLDatabase(
                engine,
                include_tables=["receivables_aging"],
                sample_rows_in_table_info=0
            )
            
            sql_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                tables=["receivables_aging"],
                llm=llm,
                embed_model=embed_model
            )
            
            db_enabled = True
            st.toast("✅ Database connected successfully!")
        else:
            st.info("ℹ️ Database not configured. Running in document-only mode.")
    
    except Exception as e:
        st.warning(f"⚠️ Database connection failed: {str(e)}\nRunning in document-only mode.")
    
    # ✅ NEW: Pass user role to document processor
    user_role = st.session_state.user_role
    doc_processor = DocumentProcessor(embed_model, llm, user_role=user_role)
    
    router = IntelligentSheetRouter(llm, user_role=user_role)
    hybrid_engine = HybridQueryEngine(sql_engine, doc_processor, llm, router)
    
    return hybrid_engine, doc_processor, llm, db_enabled


hybrid_engine, doc_processor, llm, db_enabled = init_system()

# UI Layout
st.title("🤖 Enterprise Multi-Source Chatbot")
st.markdown("Query databases, uploaded documents, or pre-loaded sheets!")

# ✅ NEW: Show current user role
st.sidebar.info(f"👤 **User Role:** {st.session_state.user_role.value}")

# Sidebar for file uploads (PDFs only now - Excel sheets are pre-loaded)
with st.sidebar:
    st.header("📄 Upload Additional Documents")
    st.caption("Excel sheets are pre-loaded. Upload PDFs for additional context.")
    
    uploaded_file = st.file_uploader(
        "Upload PDF or Image (optional)",
        type=["pdf", "png", "jpg", "jpeg"],
        key="file_uploader"
    )
    
    if uploaded_file and uploaded_file not in st.session_state.uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                file_extension = uploaded_file.name.lower()
                
                if file_extension.endswith('.pdf'):
                    # ✅ Process PDF
                    doc_processor.process_pdf(
                        uploaded_file, 
                        st.session_state.session_id
                    )
                    st.success(f"✅ Processed PDF: {uploaded_file.name}")
                    
                    # ✅ NEW: Verify it's queryable
                    if st.session_state.session_id in doc_processor.vector_indexes:
                        st.info("📄 PDF is now queryable! Ask questions about it.")
                    
                elif file_extension.endswith(('.xlsx', '.xls')):
                    # ✅ Process uploaded Excel
                    doc_processor.process_excel(
                        uploaded_file,
                        session_id=f"uploaded_{st.session_state.session_id}",
                        sheet_id=None,
                        metadata=None
                    )
                    st.success(f"✅ Processed Excel: {uploaded_file.name}")
                    
                else:
                    # ✅ Process image
                    doc_processor.process_image(
                        uploaded_file, 
                        st.session_state.session_id
                    )
                    st.success(f"✅ Processed Image: {uploaded_file.name}")
                
                st.session_state.uploaded_files.append(uploaded_file)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

    
    if st.session_state.uploaded_files:
        st.subheader("📁 Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.text(f"✓ {file.name}")
    
    st.divider()
    
    # ✅ NEW: Show pre-loaded sheets
    st.header("📊 Pre-Loaded Data Sources")
    st.caption("These sheets are always available (based on your role):")
    
    from sheet_registry import get_available_sheets
    available_sheets = [s for s in get_available_sheets() 
                       if s.sheet_id in doc_processor.router.allowed_sheets]
    
    if available_sheets:
        for sheet in available_sheets[:5]:  # Show first 5
            st.text(f"{sheet.icon} {sheet.name}")
        if len(available_sheets) > 5:
            st.caption(f"... and {len(available_sheets)-5} more")
    
    st.divider()
    st.header("💡 Example Queries")
    
    examples = {
        "Sales": [
            "What is the sales target achievement?",
            "Show January sales by consultant",
        ],
        "Service": [
            "How many service requests pending?",
        ],
        "Finance": [
            "Receivables aging analysis",
            "Show debtors older than 90 days",
        ],
        "Demo": [
            "Demo vehicle inventory status",
        ]
    }
    
    for category, queries in examples.items():
        st.subheader(category)
        for q in queries:
            if st.button(q, key=f"ex_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                has_uploaded = len(st.session_state.uploaded_files) > 0
                
                # ✅ STEP 1: Check if query is about uploaded file
                if has_uploaded:
                    # Keywords that indicate uploaded file
                    uploaded_keywords = [
                        "uploaded", "this file", "this document", 
                        "performance", "product deck", "annexure"
                    ]
                    
                    query_lower = prompt.lower()
                    is_uploaded_query = any(kw in query_lower for kw in uploaded_keywords)
                    
                    # Or if query is generic and file was just uploaded
                    recent_upload = (
                        len(st.session_state.messages) < 2 and 
                        has_uploaded
                    )
                    
                    if is_uploaded_query or recent_upload:
                        # Query uploaded file
                        if st.session_state.session_id in doc_processor.vector_indexes:
                            response = doc_processor._query_vector_index(
                                prompt,
                                st.session_state.session_id
                            )
                        else:
                            response = "❌ No uploaded document found. Please upload a file first."
                    else:
                        # Query pre-loaded sheets
                        response = doc_processor.query_documents(
                            prompt, 
                            st.session_state.session_id
                        )
                else:
                    # No uploaded files - query pre-loaded sheets
                    response = doc_processor.query_documents(
                        prompt, 
                        st.session_state.session_id
                    )
                
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })