import streamlit as st
import pandas as pd
import json
import tempfile
import time
import os
import numpy as np
from paddleocr import PaddleOCR
from hugchat import hugchat
from hugchat.login import Login
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
import sqlite3
import logging
import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
MAX_WORKERS = 4
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png"]

# Initialize OCR engine (singleton)
@st.cache_resource
def get_ocr_engine():
    try:
        logger.info("Initializing OCR engine")
        return PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )
    except Exception as e:
        logger.error(f"OCR initialization failed: {e}")
        raise

# Initialize Chatbot (singleton)
@st.cache_resource
def get_chatbot():
    try:
        logger.info("Initializing Chatbot")
        sign = Login("syedzoya", "Zoya@huggingface18")
        cookies = sign.login()
        cookie_dir = tempfile.mkdtemp()
        sign.saveCookiesToDir(cookie_dir)
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        chatbot.new_conversation()
        return chatbot
    except Exception as e:
        logger.error(f"Chatbot initialization failed: {e}")
        raise

# Data Model
class InvoiceData(BaseModel):
    invoice_number: Optional[str] = Field(description="The invoice number")
    date: Optional[str] = Field(description="The invoice date")
    vendor_name: Optional[str] = Field(description="The vendor's name")
    from_details: Optional[str] = Field(description="The 'from' details")
    to: Optional[str] = Field(description="The 'to' details")
    total_amount: Optional[str] = Field(description="The total amount")
    tax_amount: Optional[str] = Field(description="The tax amount")
    line_items: Optional[List[str]] = Field(description="List of line items")
    payment_terms: Optional[str] = Field(description="Payment terms")
    due_date: Optional[str] = Field(description="Due date")
    filename: Optional[str] = Field(description="Original filename")

# Initialize parser
parser = PydanticOutputParser(pydantic_object=InvoiceData)

# Database Setup
def setup_database():
    """Initialize database connection and tables"""
    try:
        conn = sqlite3.connect("scribes.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_filename ON files (filename)
        """)
        conn.commit()
        return conn
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

# Document Processing Functions with Retry Logic
def extract_text_with_retry(image_path, ocr_engine):
    """Extract text with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            result = ocr_engine.ocr(image_path, cls=True)
            text = "\n".join(
                line[1][0] for line in result[0] 
                if line and len(line) > 1 and isinstance(line[1], tuple) and line[1]
            )
            return text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning(f"OCR failed after {MAX_RETRIES} attempts for {image_path}")
                return ""
            time.sleep(1)

def structure_data_with_retry(text, filename, chatbot, parser):
    """Structure data with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            prompt = f"""
            you are given an extracted text from image, analyze it carefully and structure it with max accuracy.
            {parser.get_format_instructions()}
            Document Text:
            {text}
            """
            
            response = chatbot.chat(prompt)
            response_text = getattr(response, 'text', str(response))
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
                
            parsed = parser.parse(response_text)
            return {**parsed.dict(), "filename": filename}
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning(f"Structuring failed after {MAX_RETRIES} attempts for {filename}")
                return {
                    "filename": filename,
                    "error": str(e),
                    "raw_text": text[:1000]
                }
            time.sleep(1)

def process_single_document(file, ocr_engine, chatbot, parser):
    """Process a single document with comprehensive error handling"""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        
        # Extract text
        extracted_text = extract_text_with_retry(tmp_path, ocr_engine)
        if not extracted_text:
            return {
                "filename": file.name,
                "status": "failed",
                "error": "No text extracted"
            }
        
        # Structure data
        structured_data = structure_data_with_retry(extracted_text, file.name, chatbot, parser)
        structured_data["status"] = "processed"
        return structured_data
        
    except Exception as e:
        logger.error(f"Unexpected error processing {file.name}: {e}")
        return {
            "filename": file.name,
            "status": "failed",
            "error": f"Processing error: {str(e)}"
        }
    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass

def process_all_documents(files, ocr_engine, chatbot, parser):
    """Process all documents with guaranteed completion"""
    results = []
    
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(files))) as executor:
        futures = {
            executor.submit(
                process_single_document,
                file,
                ocr_engine,
                chatbot,
                parser
            ): file.name for file in files
        }
        
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                filename = futures[future]
                logger.error(f"Thread error processing {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": f"Thread error: {str(e)}"
                })
    
    return results

# UI Components
def show_processing_progress(progress, status):
    """Display processing progress"""
    progress_bar = st.progress(progress)
    status_text = st.empty()
    status_text.markdown(f"**Status:** {status}")
    return progress_bar, status_text

def display_results(results):
    """Display processing results"""
    df = pd.DataFrame(results)
    
    # Count stats
    processed = df[df['status'] == 'processed'].shape[0]
    failed = df[df['status'] == 'failed'].shape[0]
    
    # Show summary
    st.success(f"Processing complete! {processed} succeeded, {failed} failed")
    
    # Display data
    st.dataframe(df)
    
    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results",
        data=csv,
        file_name="processing_results.csv",
        mime="text/csv"
    )
    
    # Save to DB option
    if st.button("Save to Database") and processed > 0:
        try:
            conn = setup_database()
            with conn:
                conn.executemany(
                    "INSERT INTO files (filename, data) VALUES (?, ?)",
                    [(r['filename'], json.dumps(r)) for r in results if r['status'] == 'processed']
                )
            st.success(f"Saved {processed} documents to database")
        except Exception as e:
            st.error(f"Database save failed: {e}")

# Main App Functions
def app_dashboard():
    """Dashboard page showing key statistics with clean, aesthetic visualizations"""
    st.title("📊 Dashboard")
    st.write("Overview of data in the database.")

    try:
        conn = setup_database()
        total_count = conn.execute("SELECT COUNT(*) as count FROM files").fetchone()[0]
        
        if total_count > 0:
            st.metric("Total Files Processed", total_count)
            
            # Vendor analysis
            vendors = conn.execute("""
                SELECT DISTINCT json_extract(data, '$.vendor_name') as vendor 
                FROM files 
                WHERE json_extract(data, '$.vendor_name') IS NOT NULL
            """).fetchall()
            
            if vendors:
                vendor_list = [v[0] for v in vendors if v[0]]
                st.subheader("Vendor Summary")
                st.caption(f"{len(vendor_list)} unique vendors")
                
                vendor_counts = conn.execute("""
                    SELECT json_extract(data, '$.vendor_name') as vendor, COUNT(*) as count
                    FROM files
                    WHERE json_extract(data, '$.vendor_name') IS NOT NULL
                    GROUP BY vendor
                    ORDER BY count DESC
                    LIMIT 8
                """).fetchall()
                
                if vendor_counts:
                    vendor_df = pd.DataFrame(vendor_counts, columns=["Vendor", "Documents"])
                    
                    # Apply 538 style with pastel palette
                    plt.style.use('fivethirtyeight')
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Pastel color palette
                    pastel_colors = [
                        '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
                        '#80b1d3', '#fdb462', '#b3de69', '#fccde5'
                    ]
                    
                    # Create vertical bar plot
                    bars = ax.bar(vendor_df["Vendor"], vendor_df["Documents"],
                                 color=pastel_colors[:len(vendor_df)],
                                 edgecolor='white', linewidth=0.7)
                    
                    # Customize plot
                    ax.set_title('Top Vendors', fontsize=16, pad=20, weight='bold')
                    ax.set_ylabel('Number of Documents', fontsize=12)
                    ax.tick_params(axis='x', rotation=45, labelsize=10)
                    ax.tick_params(axis='y', labelsize=10)
                    
                    # Remove unnecessary chart junk
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels on top of bars if space permits
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', fontsize=10)
                    
                    st.pyplot(fig)

            # Amount trend visualization
            st.markdown("---")
            st.subheader("Invoice Amount Trend")
            
            recent_files = conn.execute("""
                SELECT 
                    json_extract(data, '$.date') as date,
                    json_extract(data, '$.total_amount') as total_amount
                FROM files
                WHERE json_extract(data, '$.date') IS NOT NULL
                AND json_extract(data, '$.total_amount') IS NOT NULL
                ORDER BY date ASC
            """).fetchall()
            
            if recent_files:
                trend_df = pd.DataFrame(recent_files, columns=["Date", "Amount"])
                trend_df['Date'] = pd.to_datetime(trend_df['Date'], errors='coerce')
                trend_df['Amount'] = pd.to_numeric(trend_df['Amount'], errors='coerce')
                trend_df = trend_df.dropna().sort_values(by="Date")
                
                # Apply 538 style
                plt.style.use('fivethirtyeight')
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Create line plot with subtle styling
                ax.plot(trend_df['Date'], trend_df['Amount'],
                       linewidth=2.5, color='#30a2da', alpha=0.9,
                       marker='o', markersize=5, markeredgecolor='white',
                       markerfacecolor='#30a2da')
                
                # Customize plot
                ax.set_title('Invoice Amounts Over Time', fontsize=16, pad=20, weight='bold')
                ax.set_ylabel('Amount ($)', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Format x-axis dates cleanly
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(trend_df)//6)))
                plt.xticks(rotation=45, fontsize=8)  # Smaller font size for x-axis labels
                ax.set_xlabel("Date (Months)", fontsize=10)  # Add proper units to x-axis label
                
                # Clean up the plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig)
                
                # Show clean metrics in columns
                st.markdown("### Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Highest Invoice", f"${trend_df['Amount'].max():,.0f}")
                with col2:
                    st.metric("Average Invoice", f"${trend_df['Amount'].mean():,.0f}")
                with col3:
                    st.metric("Total Volume", f"${trend_df['Amount'].sum():,.0f}")
            else:
                st.info("No invoice data available for trend analysis")
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        
def app_upload_files():
    """File upload and processing with guaranteed completion"""
    st.title("📄 Upload Files")
    st.write("Upload and process multiple documents simultaneously")

    uploaded_files = st.file_uploader(
        "Choose files to process",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Initializing processing engine..."):
            try:
                ocr_engine = get_ocr_engine()
                chatbot = get_chatbot()
                setup_database()
                
                progress_bar, status_text = show_processing_progress(0, "Starting...")
                
                # Process files
                results = process_all_documents(uploaded_files, ocr_engine, chatbot, parser)
                
                # Update UI
                progress_bar.progress(1.0)
                status_text.markdown("**Status:** Processing complete!")
                
                # Show results
                display_results(results)
                
            except Exception as e:
                st.error(f"System initialization failed: {e}")
                logger.exception("System initialization error")

def app_search_files():
    """Page for searching files"""
    st.title("🔍 Search Database")
    st.write("Search for files in the database using various criteria.")

    try:
        conn = setup_database()
        # Get sample record to determine available fields
        sample = conn.execute("SELECT data FROM files LIMIT 1").fetchone()
        
        if not sample:
            st.warning("No files found in the database")
            return
            
        sample_data = json.loads(sample[0])
        attributes = list(sample_data.keys())
        
        tab1, tab2 = st.tabs(["Search by Attribute", "Search by Date Range"])

        with tab1:
            st.subheader("Search by Attribute")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_attribute = st.selectbox("Select attribute", attributes)
            
            with col2:
                search_value = st.text_input("Enter search value", value="")

            exact_match = st.checkbox("Exact match", value=False)
            
            if st.button("Search", key="search_attr"):
                if exact_match:
                    query = """
                    SELECT id, filename, data 
                    FROM files 
                    WHERE json_extract(data, '$."{}"') = ?
                    """.format(selected_attribute.replace('"', '""'))
                    params = (search_value,)
                else:
                    query = """
                    SELECT id, filename, data 
                    FROM files 
                    WHERE json_extract(data, '$."{}"') LIKE ?
                    """.format(selected_attribute.replace('"', '""'))
                    params = (f"%{search_value}%",)
                
                results = conn.execute(query, params).fetchall()
                
                if results:
                    formatted = [{
                        "id": r[0],
                        "filename": r[1],
                        **json.loads(r[2])
                    } for r in results]
                    
                    df = pd.DataFrame(formatted)
                    st.success(f"Found {len(df)} matching files.")
                    st.dataframe(df[[selected_attribute, 'filename']])
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="search_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No matching files found.")

        with tab2:
            st.subheader("Search by Date Range")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Start date")
            
            with col2:
                end_date = st.date_input("End date")
            
            if st.button("Search", key="search_date"):
                if start_date and end_date:
                    results = conn.execute("""
                        SELECT id, filename, data 
                        FROM files 
                        WHERE json_extract(data, '$.date') >= ? AND json_extract(data, '$.date') <= ?
                        """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))).fetchall()
                    
                    if results:
                        formatted = [{
                            "id": r[0],
                            "filename": r[1],
                            **json.loads(r[2])
                        } for r in results]
                        
                        df = pd.DataFrame(formatted)
                        st.success(f"Found {len(df)} files within the date range.")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="date_search_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No files found within the date range.")
                else:
                    st.warning("Please select both start and end dates.")
                    
    except Exception as e:
        st.error(f"Error searching database: {e}")

# Main App
def main():
    # App configuration
    st.set_page_config(
        page_title="SCRIBES",
        page_icon="filogo.png",
        layout="wide"
    )
    
    # Header with logo
    logo_path = "scribes.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -20px;">
                <img src="data:image/png;base64,{encoded_image}" width="300">
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.title("SCRIBES")
    st.subheader("AI-Powered Document Processing")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Document Management")
    
    # Sidebar image
    ddimg_path = "ddimg.jpg"
    if os.path.exists(ddimg_path):
        with open(ddimg_path, "rb") as image_file:
            ddimg_encoded = base64.b64encode(image_file.read()).decode()
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="data:image/png;base64,{ddimg_encoded}" width="200">
            </div>
            """,
            unsafe_allow_html=True
        )
    
    app_mode = st.sidebar.selectbox(
        "Select a page",
        ["Upload Files", "Dashboard", "Search Database"]
    )
    
    # Page routing
    if app_mode == "Upload Files":
        app_upload_files()
    elif app_mode == "Dashboard":
        app_dashboard()
    elif app_mode == "Search Database":
        app_search_files()
    
    # Footer
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Powered by <a href="https://falconinformatics.com" target="_blank">Falcon Informatics</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
