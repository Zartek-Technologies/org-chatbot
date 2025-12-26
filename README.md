# Enterprise Chatbot

A Streamlit-based chatbot that can query both structured databases (PostgreSQL) and unstructured documents (Excel, PDF, Images) using LLMs (Groq), LlamaIndex, and Pandas.

## Features

- **Hybrid Querying**: Intelligently routes queries to SQL database, uploaded documents, or both.
- **Document Analysis**:
  - **Excel**: Automatically interprets headers, cleans data, and generates Pandas code to answer natural language questions.
  - **PDF**: Extracts text and performs retrieval-augmented generation (RAG).
  - **Images**: Processes image metadata (OCR integration ready).
- **SQL Integration**: Queries PostgreSQL databases using natural language.
- **Smart Routing**: automatically determines the best data source for your question.

## Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- Groq API Key

## Installation

1.  **Clone the repository** (if not already done):
    ```bash
    git clone https://github.com/alfik-z/org-chatbot.git
    cd org-chatbot
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file in the project root by copying the example:
    ```bash
    cp .env.example .env
    # Or manually create .env and copy contents from .env.example
    ```

2.  Edit `.env` and fill in your credentials:
    ```ini
    GROQ_API_KEY=your_actual_api_key
    POSTGRES_USER=your_postgres_user
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DATABASE=your_database_name
    ```

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run main.py
    ```

2.  **Access the web interface**:
    The app should automatically open in your browser at `http://localhost:8501`.

3.  **Interact**:
    - Upload documents (Excel, PDF) via the sidebar.
    - Select your query source (Auto, Documents, Database, or Both).
    - Ask questions like:
        - "Summarize the uploaded Excel file"
        - "Show total receivables from the database"
        - "Compare the uploaded attendance sheet with the database records"

## Project Structure

- `main.py`: Application entry point and UI logic.
- `document_processor.py`: Handles file uploads (Excel/PDF) and creates query engines.
- `pandas_executor.py`: Securely executes generated Pandas code for Excel analysis.
- `hybrid_engine.py`: Orchestrates logic between SQL and Document engines.
- `router.py`: Determines user intent (SQL vs Doc vs Hybrid).

