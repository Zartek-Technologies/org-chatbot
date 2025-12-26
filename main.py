import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uuid

from router import QueryRouter, QueryType
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

@st.cache_resource
def init_system():
    llm = Groq(
        model="openai/gpt-oss-120b",
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
        # Check if all required DB env vars are present
        db_user = os.getenv('POSTGRES_USER')
        db_password = os.getenv('POSTGRES_PASSWORD')
        db_host = os.getenv('POSTGRES_HOST')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DATABASE')
        
        if all([db_user, db_password, db_host, db_name]):
            engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            
            # Test connection
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
    
    doc_processor = DocumentProcessor(embed_model, llm)
    router = QueryRouter(llm)
    hybrid_engine = HybridQueryEngine(sql_engine, doc_processor, llm, router)
    
    return hybrid_engine, doc_processor, llm, db_enabled

hybrid_engine, doc_processor, llm, db_enabled = init_system()

# UI Layout
st.title("🤖 Enterprise Multi-Source Chatbot")
st.markdown("Query databases, uploaded documents, or both!")

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Upload Excel, PDF, or Image",
        type=["xlsx", "xls", "pdf", "png", "jpg", "jpeg"],
        key="file_uploader"
    )
    
    if uploaded_file and uploaded_file not in st.session_state.uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                file_extension = uploaded_file.name.lower()
                
                if file_extension.endswith(('.xlsx', '.xls')):
                    doc_processor.process_excel(uploaded_file, st.session_state.session_id)
                    st.success(f"✅ Processed Excel: {uploaded_file.name}")
                    
                elif file_extension.endswith('.pdf'):
                    doc_processor.process_pdf(uploaded_file, st.session_state.session_id)
                    st.success(f"✅ Processed PDF: {uploaded_file.name}")
                    
                else:
                    doc_processor.process_image(uploaded_file, st.session_state.session_id)
                    st.success(f"✅ Processed Image: {uploaded_file.name}")
                
                st.session_state.uploaded_files.append(uploaded_file)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    if st.session_state.uploaded_files:
        st.subheader("📄 Uploaded Files")
        for file in st.session_state.uploaded_files:
            st.text(f"✓ {file.name}")
    
    st.divider()
    st.header("🎯 Query Source")
    
    # Adjust options based on DB availability
    if db_enabled:
        query_options = ["Auto (Smart Routing)", "Uploaded Documents Only", "Database Only", "Both"]
    else:
        query_options = ["Uploaded Documents Only"]
        st.caption("⚠️ Database mode unavailable")
    
    query_source = st.radio(
        "Choose data source:",
        query_options,
        index=0,
        help="Auto: Automatically detects which source to use" if db_enabled else "Database not configured"
    )
    
    st.divider()
    st.header("💡 Example Queries")
    
    examples = {}
    
    if db_enabled:
        examples["Database"] = [
            "Total receivables pending?",
            "Receivables aging > 360 days?"
        ]
    
    examples["Documents"] = [
        "Summarize the uploaded Excel",
        "Show attendance for sreejith",
        "What's in the sheet?"
    ]
    
    if db_enabled:
        examples["Hybrid"] = [
            "Compare uploaded data with database"
        ]
    
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
                has_docs = len(st.session_state.uploaded_files) > 0
                
                # Check if database query is requested but DB is disabled
                if not db_enabled and query_source in ["Database Only", "Both"]:
                    response = "❌ Database is not configured. Please upload documents or configure database credentials in secrets."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    if query_source == "Uploaded Documents Only":
                        if not has_docs:
                            response = "❌ No documents uploaded yet. Please upload a file first."
                        else:
                            response = "📄 **Querying Uploaded Documents...**\n\n"
                            doc_result = doc_processor.query_documents(prompt, st.session_state.session_id)
                            response += str(doc_result)
                    
                    elif query_source == "Database Only":
                        response = "🗄️ **Querying Database...**\n\n"
                        db_result = hybrid_engine._query_sql(prompt)
                        response += str(db_result)
                    
                    elif query_source == "Both":
                        if not has_docs:
                            response = "❌ No documents uploaded. Querying database only...\n\n"
                            response += str(hybrid_engine._query_sql(prompt))
                        else:
                            response = hybrid_engine._query_hybrid(prompt, st.session_state.session_id)
                    
                    else:  # Auto routing
                        response = hybrid_engine.query(prompt, st.session_state.session_id, has_docs)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
