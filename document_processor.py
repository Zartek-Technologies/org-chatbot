# document_processor.py
import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from typing import List, Dict
import PyPDF2
from io import BytesIO
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os

# ✅ NEW IMPORTS
from sheet_registry import SHEET_REGISTRY, SheetSource, get_available_sheets
from intelligent_sheet_router import IntelligentSheetRouter, UserRole


class DocumentProcessor:
    def __init__(self, embed_model, llm, user_role: UserRole = UserRole.CEO):
        self.embed_model = embed_model
        self.llm = llm
        self.user_role = user_role
        
        # ✅ NEW: Initialize router
        self.router = IntelligentSheetRouter(llm, user_role)
        
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1024,     
            chunk_overlap=200    
        )
        
        # Data storage
        self.pandas_engines = {}
        self.vector_indexes = {}
        self.file_types = {}
        self.dataframes = {}
        self.sheet_metadata = {}  # ✅ NEW: Store metadata
        
        # ✅ NEW: Preload all sheets at startup
        self._preload_all_sheets()

    def _preload_all_sheets(self):
        """Load all available sheets at startup"""
        
        print("\n" + "="*70)
        print("📁 PRE-LOADING EXCEL SHEETS")
        print("="*70)
        
        loaded_count = 0
        failed_count = 0
        skipped_count = 0
        
        for sheet_id, sheet_source in SHEET_REGISTRY.items():
            # Check if user has access to this sheet
            if sheet_id not in self.router.allowed_sheets:
                skipped_count += 1
                continue
            
            if not sheet_source.is_available():
                print(f"  ⚠️  {sheet_source.name}: File not found")
                print(f"       Path: {sheet_source.file_path}")
                failed_count += 1
                continue
            
            try:
                with open(sheet_source.file_path, 'rb') as f:
                    # Use a session ID that includes the sheet ID
                    session_id = f"preloaded_{sheet_id}"
                    
                    self.process_excel(
                        f,
                        session_id=session_id,
                        sheet_id=sheet_id,
                        metadata=sheet_source
                    )
                    print(f"  ✅ {sheet_source.name}")
                    loaded_count += 1
                    
            except Exception as e:
                print(f"  ❌ {sheet_source.name}: {str(e)}")
                failed_count += 1
        
        print("="*70)
        print(f"✅ Loaded: {loaded_count} | ❌ Failed: {failed_count} | ⚠️ Skipped: {skipped_count}")
        print("="*70 + "\n")
        
        if loaded_count == 0:
            print("\n⚠️  WARNING: No sheets loaded! Check your .env file paths.\n")


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
    
    def _process_multi_sheet_excel(
        self, 
        excel_file, 
        session_id: str, 
        sheet_id: str, 
        metadata: SheetSource,
        sheet_names: list
    ) -> None:
        """
        Process Excel files with multiple sheets (like MIS reports).
        Creates separate query engines for each sheet.
        """
        
        print(f"\n🔄 Processing multi-sheet Excel: {metadata.name}")
        
        # Prioritize certain sheets (common MIS structure)
        priority_sheets = [
            "Summary", "Dashboard", "Overview", "P&L", "Profit & Loss",
            "Revenue", "Expenses", "Balance Sheet", "Cash Flow"
        ]
        
        # Try to find priority sheet first
        target_sheets = []
        for priority in priority_sheets:
            for sheet_name in sheet_names:
                if priority.lower() in sheet_name.lower():
                    target_sheets.insert(0, sheet_name)  # Add to front
                    break
        
        # Add remaining sheets
        for sheet_name in sheet_names:
            if sheet_name not in target_sheets:
                target_sheets.append(sheet_name)
        
        # Limit to first 5 sheets to avoid overload
        target_sheets = target_sheets[:5]
        
        print(f"📋 Will process sheets: {target_sheets}")
        
        # Process each sheet
        all_dfs = {}
        
        for idx, sheet_name in enumerate(target_sheets):
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Clean the dataframe
                df = self._clean_dataframe(df)
                
                if df.empty or len(df.columns) < 2:
                    print(f"  ⚠️ Skipped {sheet_name}: Too little data")
                    continue
                
                print(f"  ✅ Loaded sheet '{sheet_name}': {df.shape}")
                
                # Store with unique session ID per sheet
                sheet_session_id = f"{session_id}_{sheet_name}"
                self.dataframes[sheet_session_id] = df
                
                # Create metadata for this specific sheet
                sheet_metadata = SheetSource(
                    sheet_id=f"{sheet_id}_{sheet_name}",
                    name=f"{metadata.name} - {sheet_name}",
                    file_path=metadata.file_path,
                    sheet_name=sheet_name,
                    category=metadata.category,
                    keywords=metadata.keywords + [sheet_name.lower()],
                    description=f"{metadata.description} - {sheet_name} sheet",
                    icon=metadata.icon
                )
                
                self.sheet_metadata[sheet_session_id] = sheet_metadata
                
                # Build instruction
                instruction_str = self._build_instruction(df, sheet_metadata)
                
                # Create query engine
                query_engine = PandasQueryEngine(
                    df=df,
                    llm=self.llm,
                    verbose=False,
                    instruction_str=instruction_str,
                    synthesize_response=False,
                    output_processor=self._create_output_processor(df)
                )
                
                self.pandas_engines[sheet_session_id] = query_engine
                all_dfs[sheet_name] = df
                
            except Exception as e:
                print(f"  ❌ Failed to process sheet '{sheet_name}': {str(e)}")
        
        # ✅ NEW: Also create a COMBINED view
        if all_dfs:
            self._create_combined_mis_view(session_id, all_dfs, metadata)
        
        self.file_types[session_id] = "excel_multi_sheet"
        
        print(f"✅ Processed {len(all_dfs)} sheets from {metadata.name}\n")


    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe"""
        
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.map(str).str.strip()
        
        # Remove columns with too many NaNs (>80%)
        threshold = 0.8
        df = df.loc[:, (df.isna().sum() / len(df)) < threshold]
        
        # Remove columns with only one unique value
        df = df.loc[:, df.nunique() > 1]
        
        # Fill NaN in text columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')
        
        return df


    def _create_output_processor(self, df):
        """Create output processor with df in closure"""
        def output_processor(output: str):
            cleaned_code = self._clean_generated_code(output)
            try:
                local_vars = {"df": df, "pd": pd}
                exec(f"result = {cleaned_code}", {}, local_vars)
                return local_vars.get("result")
            except Exception as e:
                print(f"⚠️ Code execution failed: {str(e)}")
                return None
        return output_processor


    def _create_combined_mis_view(self, session_id: str, all_dfs: dict, metadata: SheetSource):
        """
        Create a combined summary view of all MIS sheets.
        Useful for queries like "Give me overall summary".
        """
        
        # Create summary DataFrame
        summary_rows = []
        
        for sheet_name, df in all_dfs.items():
            summary_rows.append({
                "Sheet": sheet_name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Column_Names": ", ".join(df.columns.tolist()[:5])
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Store summary
        summary_session_id = f"{session_id}_SUMMARY"
        self.dataframes[summary_session_id] = summary_df
        
        # Create summary metadata
        summary_metadata = SheetSource(
            sheet_id=f"{metadata.sheet_id}_summary",
            name=f"{metadata.name} - Summary",
            file_path=metadata.file_path,
            category=metadata.category,
            keywords=metadata.keywords + ["summary", "overview"],
            description=f"Summary view of all sheets in {metadata.name}",
            icon="📋"
        )
        
        self.sheet_metadata[summary_session_id] = summary_metadata
        
        print(f"  ✅ Created combined summary view")



    def process_excel(self, file, session_id: str, sheet_id: str = None, metadata: SheetSource = None):
        """Process Excel files with improved header detection"""
        try:
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            
            print(f"\n📊 Excel file has {len(sheet_names)} sheet(s): {sheet_names}")
            
            # If MIS or multi-sheet file, process intelligently
            if len(sheet_names) > 1 and metadata and metadata.sheet_id == "mis_report":
                return self._process_multi_sheet_excel(
                    excel_file, 
                    session_id, 
                    sheet_id, 
                    metadata, 
                    sheet_names
                )
        except Exception as e:
            print(f"⚠️ Could not detect sheets: {str(e)}")
        df_raw = pd.read_excel(file, header=None)
        
        # ✅ IMPROVED: Better header detection
        header_row = 0
        max_score = 0
        
        for idx in range(min(5, len(df_raw))):
            row = df_raw.iloc[idx]
            non_null = row.notna().sum()
            
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

        threshold = 0.8
        df = df.loc[:, (df.isna().sum() / len(df)) < threshold]
        df = df.loc[:, df.nunique() > 1]
        
        # Fill NaN
        for col in df.select_dtypes(include=['object']).columns:
            df.loc[:, col] = df[col].fillna('')
        
        print(f"✅ Loaded DataFrame with columns: {list(df.columns)}")
        print(f"✅ Shape: {df.shape}")
        
        # ✅ Store DataFrame and metadata
        self.dataframes[session_id] = df
        self.sheet_metadata[session_id] = metadata
        
        # ✅ Build instruction with metadata
        instruction_str = self._build_instruction(df, metadata)
        
        # ✅ IMPORTANT: Add output processor that cleans markdown
        def output_processor(output: str):
            """Process LLM output and clean markdown artifacts"""
            # Clean the code
            cleaned_code = self._clean_generated_code(output)
            
            # Execute the cleaned code
            try:
                local_vars = {"df": df, "pd": pd}
                exec(f"result = {cleaned_code}", {}, local_vars)
                return local_vars.get("result")
            except Exception as e:
                print(f"⚠️ Code execution failed: {str(e)}")
                print(f"Generated code: {cleaned_code}")
                return None
        
        query_engine = PandasQueryEngine(
            df=df,
            llm=self.llm,
            verbose=True,
            instruction_str=instruction_str,
            synthesize_response=False,
            output_processor=output_processor  # ✅ Use custom processor
        )
        
        self.pandas_engines[session_id] = query_engine
        self.file_types[session_id] = "excel"
        
        return query_engine

    def _build_instruction(self, df: pd.DataFrame, metadata: SheetSource = None) -> str:
        """Build context-aware instruction for LLM"""
        
        instruction = f"""
    You are a Python code generator for pandas DataFrame queries.

    CRITICAL: Return ONLY the Python expression. NO markdown, NO code blocks, NO explanations.

    """
        
        # Add metadata context if available
        if metadata:
            instruction += f"""
    DATA SOURCE: {metadata.name}
    DESCRIPTION: {metadata.description}
    """
        
        instruction += f"""
    DATA PROFILE:
    - Rows: {len(df)}
    - Columns: {', '.join(df.columns.tolist()[:15])}
    {'  ... and ' + str(len(df.columns) - 15) + ' more columns' if len(df.columns) > 15 else ''}

    COLUMN TYPES:
    {df.dtypes.head(20).to_string()}
    """
        
        # Add column info from metadata if available
        if metadata and metadata.columns_info:
            instruction += "\n\nCOLUMN MEANINGS:\n"
            for col, desc in list(metadata.columns_info.items())[:10]:
                if col in df.columns:
                    instruction += f"- {col}: {desc}\n"
        
        # Show sample data
        instruction += f"""

    SAMPLE DATA (first 3 rows):
    {df.head(3).to_string()}

    YOUR TASK:
    Generate a SINGLE pandas expression to answer the user's query.

    CRITICAL RULES:
    1. Return ONLY the Python expression - NO markdown, NO ```python blocks, NO explanations
    2. Use ONLY columns that exist in the DataFrame
    3. If the query cannot be answered with available columns, return: None
    4. For "how many" questions, return a count (use len() or .count())
    5. For "who/what/which" questions, return names/values (use .value_counts() or .unique())
    6. For "show/list" questions, return relevant rows (use df[...] with filters)
    7. Always limit results to reasonable size (use .head(20) if returning many rows)

    WRONG (DO NOT DO THIS):
    df['Column'].value_counts()

    RIGHT (DO THIS):
    df['Column'].value_counts().head(10)

    IMPORTANT: The expression you return will be executed directly. Do NOT include:
    - Markdown code blocks (``````)
    - Comments (# ...)
    - Print statements
    - Multiple lines
    - Variable assignments (x = ...)

    Return ONLY a single executable pandas expression.
    """
        
        return instruction
    def _clean_generated_code(self, code_str: str) -> str:
        """
        Clean LLM-generated code by removing markdown artifacts
        """
        import re
        
        # Remove markdown code blocks
        # code_str = re.sub(r'```
        code_str = re.sub(r'```\s*', '', code_str)
        
        # Remove leading/trailing whitespace
        code_str = code_str.strip()
        
        # If multiple lines, try to get just the actual code
        lines = code_str.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if len(code_lines) == 1:
            return code_lines[0]
        elif len(code_lines) > 1:
            # Return the last non-comment line (usually the actual expression)
            return code_lines[-1]
        
        return code_str
    
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
        """Query with intelligent sheet selection"""
        
        # ✅ FIX: Check if any sheets are loaded
        if not self.pandas_engines:
            return """
    ❌ **No data sources available.**

    **Possible reasons:**
    1. File paths in `.env` may not exist on your system
    2. Sheets failed to load during startup

    **To fix:**
    1. Check your `.env` file paths are correct
    2. Run `python sheet_registry.py` to verify paths
    3. Check console for any loading errors

    **Available sheets:** Run `python sheet_registry.py` to see status
    """
        
        # ✅ NEW: Use router to find relevant sheets
        relevant_sheets = self.router.route(query)
        
        if not relevant_sheets:
            # Show available sheets user can access
            available = [s.name for s in self.router._get_accessible_sheets()]
            return f"""
    ❌ **No matching data source found for your query.**

    **Your available data sources ({len(available)}):**
    {chr(10).join(['• ' + name for name in available[:10]])}
    {f'... and {len(available)-10} more' if len(available) > 10 else ''}

    **Try asking about:**
    - Sales targets and achievements
    - Monthly sales data (January-November)
    - Service throughput
    - Customer satisfaction
    - Receivables/debtors aging
    """
        
        print(f"\n🎯 Query: {query}")
        print(f"📊 Routing to {len(relevant_sheets)} sheet(s): {[s.name for s in relevant_sheets]}")
        
        results = []
        
        # Query each relevant sheet
        for sheet_source in relevant_sheets:
            # Find session_id for this sheet
            sheet_session = f"preloaded_{sheet_source.sheet_id}"
            
            if sheet_session in self.pandas_engines:
                result = self._query_single_sheet(query, sheet_session, sheet_source)
                
                if result and "No results" not in result:
                    results.append({
                        "sheet": sheet_source.name,
                        "sheet_id": sheet_source.sheet_id,
                        "result": result
                    })
        
        # Combine results
        return self._format_combined_results(results, query, relevant_sheets)
    
    def _intelligent_format(self, result, query: str, metadata: SheetSource, df=None) -> str:
        """
        Use LLM to intelligently format results based on query intent.
        Works for any data type - no hardcoding!
        """
        import pandas as pd
        
        # ✅ STEP 1: Convert result to string representation
        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return "No matching data found."
            
            # For large dataframes, sample for LLM
            if len(result) > 50:
                result_str = result.head(50).to_string()
                result_str += f"\n\n... ({len(result)} total rows)"
            else:
                result_str = result.to_string()
                
        elif isinstance(result, pd.Series):
            if len(result) == 0:
                return "No matching data found."
            result_str = result.to_string()
            
        elif isinstance(result, (list, tuple)):
            result_str = "\n".join([str(item) for item in result[:50]])
            if len(result) > 50:
                result_str += f"\n... and {len(result)-50} more items"
        else:
            result_str = str(result)
        
        # If result is very simple, just return it formatted
        if len(result_str) < 100 and isinstance(result, (int, float, str)):
            return f"**Result:** {result}"
        
        # ✅ STEP 2: Ask LLM to format intelligently
        format_prompt = f"""
    You are formatting query results for a business user.

    USER QUESTION: "{query}"
    DATA SOURCE: {metadata.name if metadata else "Unknown"}
    QUERY RESULT:
    {result_str}

    TASK:
    Format this data in the most appropriate way for the user's question.

    GUIDELINES:
    1. **Tables** - Use markdown tables for structured data (rankings, comparisons, lists)
    2. **Numbers** - Highlight key metrics prominently
    3. **Lists** - Use bullet points for simple lists
    4. **Insights** - Add brief context if helpful (e.g., "Top 5 performers:")
    5. **Clean** - Remove technical artifacts (dtype, index names, etc.)

    Provide ONLY the formatted output, no explanations.
    """
        
        try:
            from llama_index.core.llms import ChatMessage
            
            messages = [
                ChatMessage(role="system", content="You format data results clearly for business users. Output should be markdown-formatted, clean, and professional."),
                ChatMessage(role="user", content=format_prompt)
            ]
            
            response = self.llm.chat(messages)
            formatted = response.message.content
            
            return formatted
            
        except Exception as e:
            print(f"⚠️ LLM formatting failed: {str(e)}")
            
            # Fallback to basic pandas formatting
            return self._format_pandas_output(result)
        
    def _query_vector_index(self, query: str, session_id: str) -> str:
        """
        Query uploaded PDFs/Images using vector search.
        """
        
        index = self.vector_indexes.get(session_id)
        
        if not index:
            return "❌ No uploaded document found in session."
        
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5  # Return top 5 chunks
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine(retriever=retriever)
            
            # Query
            response = query_engine.query(query)
            
            return f"**📄 Source: Uploaded Document**\n\n{response.response}"
            
        except Exception as e:
            return f"❌ Query failed: {str(e)}"
    
    def _query_single_sheet(self, query: str, session_id: str, metadata: SheetSource) -> str:
        """Query a single sheet and return formatted results"""
        
        engine = self.pandas_engines.get(session_id)
        if not engine:
            matching_sessions = [
            sid for sid in self.pandas_engines.keys() 
            if sid.startswith(session_id)
        ]
            if matching_sessions:
        # Query all sheets and combine
                results = []
                for sid in matching_sessions[:3]:  # Limit to 3 sheets
                    sheet_engine = self.pandas_engines[sid]
                    sheet_meta = self.sheet_metadata.get(sid)
                
                    try:
                        response = sheet_engine.query(query)
                        result = response.response if hasattr(response, 'response') else response
                        
                        if result and str(result).strip():
                            results.append({
                                "sheet": sheet_meta.name if sheet_meta else "Unknown",
                                "result": result
                            })
                    except:
                        continue
            
                if results:
                    # Combine multi-sheet results
                    combined = ""
                    for item in results:
                        combined += f"### {item['sheet']}\n{item['result']}\n\n"
                    return combined
            
            return None

# ✅ Single sheet processing (existing code)
        try:
            df = self.dataframes.get(session_id)
            llm_response = engine.query(query)
            
            if hasattr(llm_response, 'response'):
                result = llm_response.response
            else:
                result = llm_response
            
            formatted = self._intelligent_format(result, query, metadata, df)
            return formatted
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Query error for {metadata.name}: {error_msg}")
            return None

    
    def _format_combined_results(self, results: list, query: str, relevant_sheets: list) -> str:
        """
        Use LLM to intelligently combine and synthesize results from multiple sheets.
        Works for ANY query type - no hardcoding!
        """
        
        if not results:
            sheet_names = [s.name for s in relevant_sheets]
            return f"❌ No results found in the following data sources:\n\n" + "\n".join([f"• {name}" for name in sheet_names])
        
        # Single source - return directly with basic formatting
        if len(results) == 1:
            return f"**📊 Source:** {results[0]['sheet']}\n\n{results[0]['result']}"
    
    # ✅ MULTIPLE SOURCES: Let LLM synthesize intelligently
    # Build context for LLM
        context = f"""
    You are analyzing data from multiple sources to answer a user's question.

    USER QUESTION: "{query}"

    DATA FROM MULTIPLE SOURCES:
    """
        
        for i, item in enumerate(results, 1):
            context += f"\n--- SOURCE {i}: {item['sheet']} ---\n"
            context += f"{item['result']}\n"
        
        # Ask LLM to synthesize
        synthesis_prompt = f"""
    {context}

    TASK:
    Analyze the data from all sources above and provide a comprehensive, well-formatted answer to the user's question.

    INSTRUCTIONS:
    1. **Combine data intelligently** - If data is across multiple months/regions, aggregate it
    2. **Format clearly** - Use markdown tables for structured data, bullet points for lists
    3. **Highlight insights** - Show top performers, trends, totals, averages as relevant
    4. **Be specific** - Include actual numbers and names from the data
    5. **No technical jargon** - Write for business users, not developers

    Provide a complete, business-ready answer.
    """
        
        try:
            from llama_index.core.llms import ChatMessage
            
            messages = [
                ChatMessage(role="system", content="You are a business intelligence assistant that synthesizes data from multiple sources into clear, actionable insights."),
                ChatMessage(role="user", content=synthesis_prompt)
            ]
            
            response = self.llm.chat(messages)
            synthesized = response.message.content
            
            # Add source attribution footer
            source_list = "\n".join([f"- {item['sheet']}" for item in results])
            footer = f"\n\n---\n**📊 Data Sources ({len(results)}):**\n{source_list}"
            
            return synthesized + footer
            
        except Exception as e:
            print(f"⚠️ LLM synthesis failed: {str(e)}")
            
            # Fallback: Show results from each sheet separately
            combined = f"**📊 Results from {len(results)} data sources:**\n\n"
            
            for i, item in enumerate(results[:10], 1):
                combined += f"### {i}. {item['sheet']}\n"
                combined += item["result"]
                combined += "\n\n"
            
            if len(results) > 10:
                combined += f"\n_... and {len(results)-10} more data sources_\n"
            
            return combined
    
    def _fallback_query(self, query: str, session_id: str) -> str:
        """Dynamic fallback that provides schema info and prompts for specific questions"""
        df = self.dataframes.get(session_id)
        if df is None:
            return "DataFrame not available."
        
        columns = df.columns.tolist()
        col_str = ", ".join(columns[:10])
        if len(columns) > 10:
            col_str += f", ... ({len(columns)-10} more)"
            
        return f"""
❌ I couldn't process that query exactly.

**Available Data Columns:**
{col_str}

**Try asking:**
- "Show the first 5 rows"
- "Count by [Column Name]"
- "Filter where [Column Name] is [Value]"
"""
