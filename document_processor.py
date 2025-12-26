# document_processor.py
import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from typing import List, Dict
import PyPDF2
from io import BytesIO
from pandas_executor import PandasCodeExecutor


class DocumentProcessor:
    def __init__(self, embed_model, llm):
        self.embed_model = embed_model
        self.llm = llm
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
        self.pandas_engines = {}
        self.vector_indexes = {}
        self.file_types = {}
        self.dataframes = {}
        
        # ✅ NEW: Initialize executor
        self.executor = PandasCodeExecutor()

    def _format_pandas_output(self, result):
        import pandas as pd

        # 1) Scalar values
        if isinstance(result, (int, float, str)):
            return f"**Result:** {result}"

        # 2) DataFrame – generic formatting
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return "No results found."

            df_display = result.copy()
            
            # Smart truncation
            max_rows = 10
            total_rows = len(result)
            truncated = False
            
            if total_rows > max_rows:
                df_display = df_display.head(max_rows)
                truncated = True
            
            # Check for "wide" dataframes (many columns)
            if len(df_display.columns) > 8:
                # If too wide, show a summary view or just select key columns if possible
                # For now, we'll try to keep it readable by converting to markdown but maybe transposing if single row
                if len(df_display) == 1:
                    df_display = df_display.T
                    return f"### 📋 Detailed Record\n\n{df_display.to_markdown(header=['Value'])}"
                else:
                    return f"### 📋 Results ({total_rows} rows)\n\n_Table is too wide to display fully. Showing first {max_rows} rows and first 8 columns._\n\n{df_display.iloc[:, :8].to_markdown(index=False)}"

            # Standard table format
            markdown_table = df_display.to_markdown(index=False)
            
            footer = f"\n\n_Showing {max_rows} of {total_rows} rows._" if truncated else ""
            return f"### 📋 Results\n\n{markdown_table}{footer}"

        # 3) Series – key/value bullets
        if isinstance(result, pd.Series):
            if len(result) == 0:
                return "No results found."
            
            # If it's a count or scalar-like series
            if len(result) == 1:
                return f"**{result.index[0]}:** {result.iloc[0]}"
                
            items = [f"- **{idx}**: {val}" for idx, val in result.items()]
            if len(items) > 10:
                items = items[:10] + [f"... and {len(result)-10} more"]
                
            body = "\n".join(items)
            return f"### 📋 Results\n\n{body}"

        # 4) List/Array
        if isinstance(result, (list, tuple, set)):
            if not result:
                return "No results."
            
            items = list(result)
            if len(items) > 10:
                display_items = items[:10]
                footer = f"\n... and {len(items)-10} more"
            else:
                display_items = items
                footer = ""
                
            formatted = "\n".join([f"- {item}" for item in display_items])
            return f"### 📋 List\n\n{formatted}{footer}"

        # 5) Fallback
        return f"**Result:**\n\n{str(result)}"



    def process_excel(self, file, session_id: str):
        """Process Excel files with improved header detection"""
        
        df_raw = pd.read_excel(file, header=None)
        
        # ✅ IMPROVED: Better header detection
        header_row = 0
        max_score = 0
        
        for idx in range(min(5, len(df_raw))):
            row = df_raw.iloc[idx]
            non_null = row.notna().sum()
            
            # Score based on non-null count and data type variety
            if non_null > 2:
                score = non_null
                sample_vals = row.dropna().astype(str).str.lower()
                keyword_matches = sum(1 for kw in ['name', 'id', 'date', 'model', 'no', 'customer', 'sl', 'age', 'days'] 
                                    if any(kw in val for val in sample_vals))
                score += keyword_matches * 2
                
                if score > max_score:
                    max_score = score
                    header_row = idx
        
        # Re-read with correct header
        df = pd.read_excel(file, header=header_row)
        
        # Clean column names
        df.columns = df.columns.map(str)
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all').dropna(axis=1, how='all')

        threshold = 0.8  # If column is 80%+ empty, drop it
        df = df.loc[:, (df.isna().sum() / len(df)) < threshold]

        df = df.loc[:, df.nunique() > 1]
        
        # Fill NaN
        for col in df.select_dtypes(include=['object']).columns:
            df.loc[:, col] = df[col].fillna('')
        
        print(f"✅ Loaded DataFrame with columns: {list(df.columns)}")
        print(f"✅ Shape: {df.shape}")
        print(f"✅ Sample:\n{df.head(2)}")
        
        # Store DataFrame
        self.dataframes[session_id] = df
        
        # ✅ CHANGED: Improved instruction with explicit schema
        instruction_str = f"""
        You are an expert Data Analyst working with a pandas DataFrame 'df'.
    
    **DATA PROFILE:**
    - Rows: {len(df)}
    - Columns: {', '.join(df.columns.tolist())}
    
    **COLUMN TYPES:**
    {df.dtypes.to_string()}
    
    **YOUR GOAL:**
    Translate the user's natural language query into a SINGLE valid pandas python expression.
    
    **CRITICAL RULES:**
    1. **SELECTIVITY:** Never return the full DataFrame (`df`) unless explicitly asked to "show data".
       - Prefer selecting specific columns relevant to the question.
       - Example: `df[['Model', 'Price']]` instead of `df`.
    2. **AGGREGATION:** If the user asks for "how many" or "total", use `len()`, `.sum()`, or `.count()`.
    3. **FILTERING:** converting text to lowercase for comparison is safer: `df[df['Col'].str.lower() == 'value']`.
    4. **LIMITS:** If returning raw rows, always consider using `.head(10)` if the result might be huge, unless the user wants all.
    5. **NO PRINTS:** Do not use `print()`. The last expression evaluation is the result.
    6. **IMPORTS:** Do NOT import anything. `pd` is available.
    
    **EXAMPLES:**
    - "List all sales consultants" -> `df['SC'].unique()`
    - "Show details for VIN 123" -> `df[df['VIN'] == '123']`
    - "Count of cars by model" -> `df.groupby('Model').size().sort_values(ascending=False)`
    - "Show the first 5 rows" -> `df.head(5)`
    
    Return ONLY the python expression.
    """
        
        query_engine = PandasQueryEngine(
            df=df,
            llm=self.llm,
            verbose=True,
            instruction_str=instruction_str,
            synthesize_response=False
        )
        
        self.pandas_engines[session_id] = query_engine
        self.file_types[session_id] = "excel"
        
        return query_engine
    
    def process_pdf(self, file, session_id: str) -> VectorStoreIndex:
        """Process PDF - unchanged"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            documents = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    doc = Document(
                        text=text,
                        metadata={"source": file.name, "page": page_num + 1, "type": "pdf_page"}
                    )
                    documents.append(doc)
            
            if not documents:
                raise ValueError("No text extracted from PDF")
            
            index = VectorStoreIndex.from_documents(documents, embed_model=self.embed_model)
            self.vector_indexes[session_id] = index
            self.file_types[session_id] = "pdf"
            
            return index
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def process_image(self, file, session_id: str) -> VectorStoreIndex:
        """Process images - unchanged"""
        from PIL import Image
        
        try:
            image = Image.open(file)
            
            text = f"""
            Image File: {file.name}
            Size: {image.size[0]} x {image.size[1]} pixels
            Format: {image.format}
            Mode: {image.mode}
            
            Note: This is an image file. For text extraction, OCR can be integrated.
            """
            
            document = Document(
                text=text,
                metadata={"source": file.name, "type": "image", "width": image.size[0], "height": image.size[1]}
            )
            
            index = VectorStoreIndex.from_documents([document], embed_model=self.embed_model)
            self.vector_indexes[session_id] = index
            self.file_types[session_id] = "image"
            
            return index
            
        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")
    
    def query_documents(self, query: str, session_id: str):
        if session_id not in self.file_types:
            return "No documents uploaded yet."

        file_type = self.file_types[session_id]

        if file_type == "excel":
            engine = self.pandas_engines.get(session_id)
            df = self.dataframes.get(session_id)

            if not engine or df is None:
                return "Excel file not properly loaded."

            try:
                llm_response = engine.query(query)
                print(f"🔍 LLM Response: {llm_response}")
                print(f"🔍 type(llm_response): {type(llm_response)}")
                print(f"🔍 llm_response:\n{llm_response}")
                result = llm_response.response
                # llm_response is already the pandas OUTPUT (df, scalar, etc.)
                return self._format_pandas_output(result)

            except Exception as e:
                print(f"⚠️ PandasQueryEngine error: {str(e)}")
                return self._fallback_query(query, session_id)

    
    def _fallback_query(self, query: str, session_id: str) -> str:
        """Dynamic fallback that provides schema info and prompts for specific questions"""
        df = self.dataframes.get(session_id)
        if df is None:
            return "DataFrame not available."
        
        # Generic helpful response showing available data points
        columns = df.columns.tolist()
        col_str = ", ".join(columns[:10])
        if len(columns) > 10:
            col_str += f", ... ({len(columns)-10} more)"
            
        return f"""
 I couldn't process that query exactly.

**Available Data Columns:**
{col_str}

**Try asking:**
- "Show the first 5 rows"
- "Count by [Column Name]"
- "Filter where [Column Name] is [Value]"
"""
