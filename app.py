import streamlit as st
from PIL import Image
import pandas as pd
import io
import time
from datetime import datetime, timedelta
import scraper_web
import llm_service
import importlib
import re
import os
import json
import openai
from streamlit_local_storage import LocalStorage
import logging

# Force reload for development
importlib.reload(scraper_web)
importlib.reload(llm_service)

# Load peashooter icon
import base64

# Load peashooter icon path
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "image.png")
peashooter_icon = Image.open(icon_path)

# Function to convert image to base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_img_as_base64(icon_path)

# Page Configuration
st.set_page_config(
    page_title="新阿姆斯特朗旋風噴射阿姆斯特朗砲",
    page_icon=peashooter_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Local Storage
localS = LocalStorage()

# Initialize Session State
if 'current_results' not in st.session_state:
    st.session_state.current_results = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'stop_scan' not in st.session_state:
    st.session_state.stop_scan = False
if 'current_job_config' not in st.session_state:
    st.session_state.current_job_config = None
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = 'config' # config, metadata_fetching, metadata_review, processing, completed

# NEW: Define default keywords constant
DEFAULT_KEYWORDS = [
    "販毒", "毒品", "製毒", "運毒",
    "詐貸", "詐欺", "詐騙集團", "車手", "水房", "假投資", "假交友",
    "逃稅", "漏稅", "逃漏稅", 
    "賭博罪", "地下賭場", "線上博弈",
    "黑幫", "幫派", "組織犯罪",
    "走私", "私煙", "私酒", "私藥", "偽藥",
    "貪污", "收賄", "圖利", "貪腐",
    "侵佔", "挪用", "掏空",
    "人口販運", "人口販賣",
    "洗錢", "制裁", "資恐",
    "證交法", "洗錢防制法", "刑法", "廢棄物清理法", "食安法",
    "內線", "虛擬貨幣",
]

if 'available_keywords' not in st.session_state:
    # Try to load from local storage
    stored_keywords = localS.getItem("user_keywords")
    if stored_keywords and isinstance(stored_keywords, list):
        st.session_state.available_keywords = stored_keywords
    else:
        # Initialize with default and save to localStorage
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()
        localS.setItem("user_keywords", st.session_state.available_keywords)
if 'selected_keywords' not in st.session_state:
    st.session_state.selected_keywords = st.session_state.available_keywords.copy()
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "Perplexity"

# Helper function to check for Chinese characters
def has_chinese(text):
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# Helper function for Excel export
def to_excel(df):
    try:
        output = io.BytesIO()
        # Use default engine (usually openpyxl) to avoid compatibility issues
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            for i, col in enumerate(df.columns):
                # Handle potential non-string types for length calculation
                try:
                    max_len = df[col].astype(str).map(len).max()
                    if pd.isna(max_len):
                        max_len = 10
                except:
                    max_len = 10
                column_len = max(max_len, len(str(col))) + 2
                column_len = min(column_len, 72)
                if hasattr(worksheet, 'set_column'):
                    worksheet.set_column(i, i, column_len)
        return output.getvalue()
    except Exception as e:
        print(f"Export error: {e}")
        return None

# Helper function to prepare Stage 1 export data (filter columns)
def prepare_stage1_export_df(df):
    """
    準備 Stage 1 的匯出資料，只保留必要欄位。
    """
    required_columns = ['keyword', 'title', 'link', 'pub_date', 'source']
    # Filter columns that exist in the dataframe
    return df[[col for col in required_columns if col in df.columns]]

# Helper function to process analysis results into DataFrame
def process_results_to_df(results):
    expanded_results = []
    for result in results:
        names = result.get('names', [])
        if isinstance(names, str):
            names = [names]
        
        # Create a row for each name
        for name in names:
            if name and name.strip():
                expanded_results.append({
                    "人物姓名": name.strip(),
                    "新聞摘要": result.get('summary', ''),
                    "資料來源": result.get('source_name', ''),
                    "連結": result.get('source_url', ''),
                    # Hidden fields for export
                    "original_text_pattern": result.get('original_text_pattern', ''),
                    "type": result.get('type', ''),
                    "reason": result.get('reason', ''),
                    "keyword": result.get('keyword', ''),
                    "pub_date": result.get('pub_date', '')
                })
    
    if expanded_results:
        return pd.DataFrame(expanded_results)
    return pd.DataFrame()

# Title and Description (Custom HTML for alignment)
st.markdown(
    f"""
    <div style="display: flex; flex-direction: row; align-items: stretch; gap: 20px; margin-bottom: 20px;">
        <div style="flex: 0 0 180px;">
            <img src="data:image/png;base64,{img_base64}" style="width: 180px; height: 180px; object-fit: contain;">
        </div>
        <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between; height: 180px;">
            <h1 style="margin: 0; padding: 0; font-size: 3.5rem; line-height: 1.2;">新阿姆斯特朗旋風噴射阿姆斯特朗砲</h1>
            <p style="margin: 0; font-size: 1.5rem; font-weight: 500; align-self: flex-start;">趕快做完  回家喝奶茶！</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS to hide "Press Enter to apply"
st.markdown(
    """
    <style>
    /* Hide 'Press Enter to apply' instructions */
    [data-testid="InputInstructions"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Determine if we are in config stage (allow editing)
    is_config_stage = st.session_state.workflow_stage == 'config'
    
    # Debug Mode Toggle
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    
    # LLM Provider Selection
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Perplexity", "OpenAI", "Google Gemini", "Meta Llama (Together.ai)"],
        index=["Perplexity", "OpenAI", "Google Gemini", "Meta Llama (Together.ai)"].index(st.session_state.llm_provider),
        help="Select your LLM API provider",
    )
    st.session_state.llm_provider = llm_provider
    
    # API Key (No default)
    api_key = st.text_input(
        f"{llm_provider} API Key", 
        type="password", 
        help=f"Enter your {llm_provider} API key here.",
    )
    
    # === Keyword Management Section ===
    st.subheader("Keyword")
    
    # Ensure lists are initialized
    if 'available_keywords' not in st.session_state:
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = []

    # 1. Standard Multiselect for Selection
    # Sync logic:
    # If we want to manipulate the selection programmatically (Add New, Delete),
    # we should operate on st.session_state['kw_multiselect_key'].
    
    if 'kw_multiselect_key' not in st.session_state:
        st.session_state.kw_multiselect_key = st.session_state.selected_keywords

    def on_multiselect_change():
        st.session_state.selected_keywords = st.session_state.kw_multiselect_key

    # The Main Multiselect
    st.multiselect(
        "Selected Keywords",
        options=st.session_state.available_keywords,
        key="kw_multiselect_key", # Binds directly to session state
        on_change=on_multiselect_change,
        help="Select keywords to include in the scan."
    )
    
    st.markdown("---")

    # 2. Add NEW keyword to library
    col_add_input, col_add_btn = st.columns([3, 1])
    
    with col_add_input:
        st.text_input(
            "New Keyword", 
            label_visibility="collapsed", 
            placeholder="Type new keyword...", 
            key="new_keyword_input"
        )
    
    def add_keyword_callback():
        kw = st.session_state.new_keyword_input.strip()
        if kw:
            # Add to available if not present
            if kw not in st.session_state.available_keywords:
                st.session_state.available_keywords.append(kw)
                localS.setItem("user_keywords", st.session_state.available_keywords)
            
            # Add to selected (and update widget key)
            if kw not in st.session_state.selected_keywords:
                st.session_state.selected_keywords.append(kw)
                st.session_state.kw_multiselect_key.append(kw)
            
            # Clear input
            st.session_state.new_keyword_input = ""

    with col_add_btn:
        st.button("Add", on_click=add_keyword_callback, use_container_width=True)

    # 3. Delete from Library (Batch Delete)
    col_del_select, col_del_btn = st.columns([3, 1])
    
    with col_del_select:
        # Use multiselect for batch deletion
        st.multiselect(
            "Delete from library",
            options=st.session_state.available_keywords,
            placeholder="Select keywords to delete...",
            label_visibility="collapsed",
            key="delete_keywords_select"
        )
        
    def delete_keywords_callback():
        keywords_to_delete = st.session_state.delete_keywords_select
        if keywords_to_delete:
            # Filter out deleted keywords from available_keywords
            st.session_state.available_keywords = [
                k for k in st.session_state.available_keywords
                if k not in keywords_to_delete
            ]
            
            # Filter out deleted keywords from selected_keywords
            st.session_state.selected_keywords = [
                k for k in st.session_state.selected_keywords
                if k not in keywords_to_delete
            ]
            
            # Sync the widget key as well
            st.session_state.kw_multiselect_key = [
                k for k in st.session_state.kw_multiselect_key
                if k not in keywords_to_delete
            ]
            
            # Update local storage once
            localS.setItem("user_keywords", st.session_state.available_keywords)
            
            # Clear the delete selection widget
            st.session_state.delete_keywords_select = []

    with col_del_btn:
        st.button("Delete", type="primary", on_click=delete_keywords_callback, use_container_width=True)

    # Date Range
    today = datetime.now().date()
    last_week = today - timedelta(days=7)
    
    date_range = st.date_input(
        "Select Date Range",
        value=(last_week, today),
        max_value=today,
        format="YYYY/MM/DD"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        # Define callback for Start Scan
        def start_scan_callback():
            # Snapshot configuration to isolate from sidebar changes
            # NOTE: API Key is intentionally excluded from snapshot to allow dynamic updates
            st.session_state.current_job_config = {
                'keywords': st.session_state.selected_keywords,
                'date_range': date_range,
                'provider': st.session_state.llm_provider
            }
            
            # Transition to metadata fetching state
            st.session_state.workflow_stage = 'metadata_fetching'
            
            # Reset results
            st.session_state.current_results = [] 
            st.session_state.analysis_results = []
            st.session_state.stop_scan = False

        st.button("Start Scan", type="primary", use_container_width=True, on_click=start_scan_callback)
        
    with col2:
        # Stop button should be enabled during processing
        stop_btn = st.button("Stop", type="secondary", use_container_width=True)

    if stop_btn:
        st.session_state.stop_scan = True
    
    

# ==========================================
# Main Logic - State Machine
# ==========================================

# Get current stage (default to config if not set, though initialization handles this)
stage = st.session_state.workflow_stage

# --- Stage 0: Config (Default) ---
if stage == 'config':
    st.info("👈 Please configure your search parameters in the sidebar and click 'Start Scan'.")

# --- Stage 1.0: Metadata Fetching (Transient) ---
# This stage is triggered by "Start Scan" and automatically transitions to 'metadata_review'
if stage == 'metadata_fetching':
    # Ensure we have a valid config
    if not st.session_state.current_job_config or not st.session_state.current_job_config['keywords']:
        st.warning("Please select or enter at least one keyword.")
        st.session_state.workflow_stage = 'config'
        st.rerun()
    
    config = st.session_state.current_job_config
    
    st.subheader("Stage 1: Metadata Fetching")
    status_container = st.status("Fetching metadata...", expanded=True)
    
    # Handle date range from config
    config_date_range = config['date_range']
    if isinstance(config_date_range, tuple):
        start_date = config_date_range[0]
        end_date = config_date_range[1] if len(config_date_range) > 1 else start_date
    else:
        start_date = config_date_range
        end_date = config_date_range

    try:
        all_rss_items = []
        log_messages = []
        
        # Create UI elements for progress
        progress_bar = st.progress(0)
        log_container = st.empty() # For persistent logs
        
        config_keywords = config['keywords']
        total_keywords = len(config_keywords)
        
        for idx, keyword in enumerate(config_keywords):
            # Check for stop signal
            if st.session_state.stop_scan:
                status_container.update(label="Scan stopped by user.", state="error")
                break

            # Update status
            status_container.write(f"正在抓取 [{keyword}] 相關新聞... ({idx+1}/{total_keywords})")
            
            # Fetch for single keyword
            rss_items = scraper_web.fetch_rss_items([keyword], start_date=start_date, end_date=end_date)
            
            # Filter: Must have Chinese characters in title
            rss_items = [n for n in rss_items if has_chinese(n['title'])]
            
            # Add to master list
            all_rss_items.extend(rss_items)
            
            # Update logs
            timestamp = datetime.now().strftime("%H:%M:%S")
            if rss_items:
                log_msg = f"✅ [{timestamp}] [{keyword}] 抓取完成，共 {len(rss_items)} 篇。"
            else:
                log_msg = f"⚠️ [{timestamp}] [{keyword}] 抓取完成，未發現相關新聞。"
            
            log_messages.append(log_msg)
            
            # Render logs (cumulative)
            log_html = "<div style='height: 150px; overflow-y: auto; background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.8em;'>"
            for msg in log_messages:
                color = "green" if "✅" in msg else "orange"
                log_html += f"<div style='color: {color}; margin-bottom: 4px;'>{msg}</div>"
            log_html += "</div>"
            
            log_container.markdown(log_html, unsafe_allow_html=True)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_keywords)
            
            # Small delay to prevent rate limiting and allow UI update
            time.sleep(0.5)
        
        # Save results
        st.session_state.current_results = all_rss_items
        
        # AI Title Screening
        # Use dynamic api_key from sidebar input
        config_api_key = api_key 
        config_provider = config['provider']
        
        if config_api_key and all_rss_items and not st.session_state.stop_scan:
            try:
                status_container.write("正在進行 AI 標題快篩...")
                screened_items = llm_service.screen_titles(
                    all_rss_items, 
                    config_keywords, 
                    config_api_key, 
                    provider=config_provider
                )
                
                if screened_items:
                    removed_count = len(all_rss_items) - len(screened_items)
                    st.session_state.current_results = screened_items
                    status_container.write(f"標題快篩完成。已過濾 {removed_count} 篇不相關新聞。")
                    log_container.markdown(f"<div style='color: blue; margin-bottom: 4px;'>🤖 [AI 快篩] 保留 {len(screened_items)} / {len(all_rss_items)} 篇新聞</div>", unsafe_allow_html=True)
                else:
                    st.warning("標題快篩後沒有保留任何新聞。")
                    st.session_state.current_results = []

            except Exception as e:
                st.error(f"標題快篩發生錯誤，將顯示所有新聞: {e}")
                logging.error(f"Title screening failed: {e}")

        status_container.update(label=f"Metadata fetch complete. Found {len(st.session_state.current_results)} items.", state="complete", expanded=False)
        
        # Transition to next stage (Fall-through)
        st.session_state.workflow_stage = 'metadata_review'
        stage = 'metadata_review' # Update local variable for fall-through execution

    except Exception as e:
        st.error(f"Error during metadata fetch: {e}")
        st.session_state.workflow_stage = 'config'

# --- Stage 1.5: Metadata Review (Interactive) ---
if stage == 'metadata_review':
    st.subheader("Stage 1: Raw Data Review")
    
    if not st.session_state.current_results:
        st.warning("No items found. Please try different keywords or date range.")
        if st.button("Back to Config"):
            st.session_state.workflow_stage = 'config'
            st.rerun()
    else:
        # Display Metadata DataFrame
        df_metadata = pd.DataFrame(st.session_state.current_results)
        
        # Ensure columns exist before selecting
        available_cols = df_metadata.columns.tolist()
        display_cols = [col for col in ['keyword', 'title', 'pub_date', 'source', 'link'] if col in available_cols]
        
        if display_cols:
            st.dataframe(df_metadata[display_cols], use_container_width=True)
        else:
            st.dataframe(df_metadata, use_container_width=True)
        
        # Export Button
        df_export_stage1 = prepare_stage1_export_df(df_metadata)
        excel_data_stage1 = to_excel(df_export_stage1)
        
        if excel_data_stage1:
            file_name_stage1 = f"negative_news_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.download_button(
                label="📥 Export to Excel",
                data=excel_data_stage1,
                file_name=file_name_stage1,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_stage1"
            )
        
        st.info(f"Found {len(st.session_state.current_results)} potential items. Click 'Continue' to fetch content and analyze, or 'Reset' to start over.")
        
        col_cont, col_reset = st.columns([1, 1])
        with col_cont:
            if st.button("Continue (Fetch Content & Analyze)", type="primary", use_container_width=True):
                st.session_state.workflow_stage = 'processing'
                stage = 'processing' # Update local variable for fall-through
        with col_reset:
            if st.button("Reset", type="secondary", use_container_width=True):
                st.session_state.current_results = []
                st.session_state.workflow_stage = 'config'
                st.rerun()

# --- Stage 2: Processing (Transient) ---
if stage == 'processing':
    st.divider()
    st.markdown("## Stage 2: Content Fetching & Analysis")
    
    # Progress tracking
    status_container = st.status("Processing...", expanded=True)
    progress_bar = st.progress(0)
    
    try:
        total_items = len(st.session_state.current_results)
        results = []
        processed_items = []
        
        # Use config from snapshot
        config = st.session_state.current_job_config
        # Use dynamic api_key from sidebar input
        config_api_key = api_key
        config_provider = config.get('provider', "Perplexity")

        if not config_api_key:
            st.error("Missing API Key. Please enter your API Key in the sidebar.")
            if st.button("Back to Config"):
                st.session_state.workflow_stage = 'config'
                st.rerun()
        else:
            # Initialize Scrollable Log Container
            log_container = st.empty()
            logs = []
            
            def update_logs(message):
                timestamp = datetime.now().strftime("%H:%M:%S")
                logs.append(f"[{timestamp}] {message}")
                log_html = f"""
                <div style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; font-family: monospace; font-size: 0.9em;">
                    {'<br>'.join(logs)}
                    <script>
                        var objDiv = document.getElementById("log_container");
                        objDiv.scrollTop = objDiv.scrollHeight;
                    </script>
                </div>
                """
                log_container.markdown(log_html, unsafe_allow_html=True)

            for idx, item in enumerate(st.session_state.current_results):
                # Check for stop signal
                if st.session_state.stop_scan:
                    update_logs("⚠️ Scan stopped by user.")
                    status_container.update(label="Scan stopped by user.", state="error")
                    break

                try:
                    # Update Status & Logs
                    current_progress = (idx + 1) / total_items
                    status_message = f"Processing item {idx + 1}/{total_items}: {item['title']}..."
                    update_logs(status_message)
                    progress_bar.progress(current_progress)
                    
                    # 2.1 Fetch Content
                    item = scraper_web.fetch_content_single(item)
                    processed_items.append(item)
                    
                    # 2.2 Analyze
                    item_results, debug_info = llm_service.analyze_single_item(
                        item, 
                        config_api_key,
                        provider=config_provider
                    )
                    
                    # Debug Inspector
                    if debug_mode:
                        with st.expander(f"🕵️ Debug: Check Input Content for '{item['title']}'"):
                            st.markdown(f"**Method:** `{debug_info.get('scraping_method', 'Unknown')}`")
                            st.markdown("### Input Text Preview")
                            st.text(debug_info.get('input_content', 'No content')[:1000])
                            st.markdown("### LLM Raw Response")
                            st.code(debug_info.get('raw_response', 'No response'), language='json')
                    
                    if item_results:
                        results.extend(item_results)
                        update_logs(f"✅ Extracted {len(item_results[0]['names'])} entities from '{item['title']}'")
                        
                except Exception as e:
                    error_msg = f"Error processing item {idx + 1}: {e}"
                    st.error(error_msg)
                    logging.error(f"Error in Stage 2 loop for item {item.get('title', 'Unknown')}: {e}")
                    update_logs(f"❌ {error_msg}")
                    continue
            
            # Update session state
            st.session_state.current_results = processed_items
            st.session_state.analysis_results = results
            
            # Process and save DataFrame
            df_results = process_results_to_df(results)
            st.session_state.analysis_results_df = df_results
            
            progress_bar.progress(1.0)
            status_container.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # Transition to completed (Fall-through)
            st.session_state.workflow_stage = 'completed'
            stage = 'completed'

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        if st.button("Retry"):
            st.rerun()
        if st.button("Reset"):
            st.session_state.workflow_stage = 'config'
            st.rerun()

# --- Stage 3: Completed (Results Display) ---
if stage == 'completed':
    st.divider()
    st.markdown("## Stage 2: Content Fetching & Analysis")
    st.subheader("Analysis Complete")
    
    # Reset Button
    if st.button("Start New Scan", type="secondary"):
        st.session_state.current_results = []
        st.session_state.analysis_results = []
        if 'analysis_results_df' in st.session_state:
            del st.session_state['analysis_results_df']
        st.session_state.workflow_stage = 'config'
        st.rerun()

    # Display Results
    if 'analysis_results_df' in st.session_state and not st.session_state.analysis_results_df.empty:
        st.subheader("3. Intelligence Report")
        
        df_display = st.session_state.analysis_results_df
        
        st.data_editor(
            df_display[["人物姓名", "新聞摘要", "資料來源", "連結"]],
            column_config={
                "連結": st.column_config.LinkColumn(
                    "原始連結",
                    display_text="點擊查看"
                ),
                "新聞摘要": st.column_config.TextColumn(
                    "新聞摘要",
                    width="large"
                )
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Excel Export
        excel_data_stage2 = to_excel(df_display)
        if excel_data_stage2:
            file_name_stage2 = f"negative_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label="📥 Export Analysis Results to Excel",
                data=excel_data_stage2,
                file_name=file_name_stage2,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_stage2"
            )
    else:
        st.info("Analysis complete. No negative news entities were found.")

# Footer
st.markdown("---")
st.markdown("Sophistication, radiance and elegance | Jennifer")
