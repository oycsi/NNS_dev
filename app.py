import streamlit as st
from PIL import Image
import pandas as pd
import io
import time
from datetime import datetime, timedelta
import scraper_web
import llm_service
import re
import os
import json
import openai
from streamlit_local_storage import LocalStorage
import logging
import sys

# === Nuclear Logging Fix ===
# 1. Define a completely silent handler
class SafeHandler(logging.Handler):
    def emit(self, record):
        pass

# 2. Force reset root logger to use ONLY the silent handler
logging.basicConfig(handlers=[SafeHandler()], level=logging.INFO, force=True)

# 3. Aggressively clean up all third-party loggers
# This prevents libraries like trafilatura from independently writing to closed stderr
for name in logging.root.manager.loggerDict:
    logger_obj = logging.getLogger(name)
    logger_obj.handlers = []
    logger_obj.propagate = True

# 4. Use module-level logger (will propagate to our Safe Root)
logger = logging.getLogger(__name__)

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

# UI Optimization: Left-aligned Spinner
st.markdown("""
<style>
    /* 強制 Spinner 靠左對齊 */
    div[data-testid="stSpinner"] {
        justify-content: flex-start;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Local Storage
localS = LocalStorage()

# Initialize Session State
if 'current_results' not in st.session_state:
    st.session_state.current_results = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'stop_scan' not in st.session_state:
    st.session_state.stop_scan = False
if 'processing_index' not in st.session_state:
    st.session_state.processing_index = 0
if 'error_log' not in st.session_state:
    st.session_state.error_log = []
if 'export_bytes' not in st.session_state:
    st.session_state.export_bytes = None
if 'current_job_config' not in st.session_state:
    st.session_state.current_job_config = None
if 'workflow_stage' not in st.session_state:
    st.session_state.workflow_stage = 'config' # config, metadata_fetching, metadata_review, processing, completed

# NEW: Define default keywords constant
DEFAULT_KEYWORDS = [
    "毒品",
    "詐欺", "詐騙集團", "車手", "水房", "假投資", "虛擬貨幣",
    "逃漏稅", 
    "賭場",
    "幫派", "組織犯罪",
    "走私",
    "貪污", "收賄", "圖利", "貪腐",
    "侵佔", "挪用", "掏空",
    "人口販賣",
    "洗錢", "制裁", "資恐",
    "證交法", "洗錢防制法", "刑法", "廢棄物清理法", "食安法",
    
]

DEFAULT_NEG_KEYWORDS = {
    "涉案": 3, "被告": 3, "嫌犯": 3, "被起訴": 3, "被判刑": 3, "被逮捕": 3, "被查獲": 3,
    "涉詐": 2, "涉毒": 2, "涉貪": 2, "詐騙集團": 2, "車手": 2, "主嫌": 2,
    "詐欺": 1, "犯罪": 1, "違法": 1, "肇事": 1, "酒駕": 1, "毒駕": 1
}
DEFAULT_COM_KEYWORDS = {
    "表示": 2, "說明": 2, "指出": 2, "強調": 2, "批評": 2, "抨擊": 2, "呼籲": 2,
    "教授": 1, "學者": 1, "系主任": 1, "主任": 1, "市長": 1, "議員": 1, "立委": 1,
    "部長": 1, "局長": 1, "處長": 1, "記者": 1, "主播": 1, "警官": 1, "檢察官": 1, "律師": 1
}

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
if 'dedup_threshold' not in st.session_state:
    st.session_state.dedup_threshold = 0.7

# Callbacks for threshold synchronization
def update_threshold_from_slider():
    st.session_state.dedup_threshold = st.session_state.threshold_slider / 100.0

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
        logger.error(f"Export error: {e}")
        return None

# Helper function to prepare Stage 1 export data (filter columns)
def prepare_stage1_export_df(df):
    """
    準備 Stage 1 的匯出資料，只保留必要欄位。
    """
    required_columns = ['keyword', 'title', 'link', 'pub_date', 'source']
    # Filter columns that exist in the dataframe
    return df[[col for col in required_columns if col in df.columns]]

def parse_pubdate(pubdate_raw):
    if not pubdate_raw: return ""
    if isinstance(pubdate_raw, datetime): return pubdate_raw.strftime("%Y-%m-%d %H:%M")
    s = str(pubdate_raw).strip()
    # Try clean ISO
    if 'T' in s:
        try: return datetime.strptime(s.split('+')[0].split('.')[0], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M")
        except: pass
    # Fallback to pandas
    try:
        dt = pd.to_datetime(pubdate_raw)
        if pd.notnull(dt): return dt.strftime("%Y-%m-%d %H:%M")
    except: pass
    return ""

def generate_snapshot_excel(results):
    rows = []
    for res in results:
        pd_str = parse_pubdate(res.get('pub_date', ''))
        # names may be list or string, ensure flatness
        names = res.get('names', [])
        if isinstance(names, str): names = [names]
        for name in names:
            rows.append({
                "keyword": res.get('keyword', ''),
                "pubdate": pd_str,
                "name": name,
                "title": res.get('title', ''),  # Use title enriched in Stage 2
                "url": res.get('source_url', '')
            })
    df = pd.DataFrame(rows, columns=["keyword", "pubdate", "name", "title", "url"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
        ws = writer.sheets['Results']
        for i, col in enumerate(df.columns):
            ws.column_dimensions[chr(65+i)].width = 20
    return output.getvalue()

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
        key="apikey"
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
            # Build filter_config from session state widgets if enabled
            filter_config = None
            if st.session_state.get("enable_custom_weights"):
                # Get DataFrames from session state (editor output)
                df_neg = st.session_state.get("editor_neg_keywords")
                df_com = st.session_state.get("editor_com_keywords")
                
                # Convert DataFrame to Dict: {"涉案": 3, ...}
                neg_w = {}
                if df_neg is not None and not df_neg.empty:
                    # Clean up: remove empty keys, convert types
                    neg_w = {
                        str(row["關鍵字"]).strip(): int(row["權重"]) 
                        for _, row in df_neg.iterrows() 
                        if str(row["關鍵字"]).strip()
                    }
                    
                com_w = {}
                if df_com is not None and not df_com.empty:
                    com_w = {
                        str(row["關鍵字"]).strip(): int(row["權重"]) 
                        for _, row in df_com.iterrows() 
                        if str(row["關鍵字"]).strip()
                    }

                mult = st.session_state.get("filter_threshold_mult", 1.5)
                filter_config = {
                    "negative_keywords": neg_w,
                    "commentator_keywords": com_w,
                    "threshold_multiplier": mult
                }

            # Snapshot configuration to isolate from sidebar changes
            # NOTE: API Key is intentionally excluded from snapshot to allow dynamic updates
            st.session_state.current_job_config = {
                'keywords': st.session_state.selected_keywords,
                'date_range': date_range,
                'provider': st.session_state.llm_provider,
                'filter_config': filter_config
            }
            
            # Transition to metadata fetching state
            st.session_state.workflow_stage = 'metadata_fetching'
            
            # Reset results
            st.session_state.current_results = [] 
            st.session_state.analysis_results = []
            st.session_state.processing_index = 0
            st.session_state.error_log = []
            st.session_state.export_bytes = None
            st.session_state.stop_scan = False

        st.button("Start Scan", type="primary", use_container_width=True, on_click=start_scan_callback)
        
    with col2:
        # Stop button should be enabled during processing
        stop_btn = st.button("Stop", type="secondary", use_container_width=True)

    if stop_btn:
        st.session_state.stop_scan = True
    
    

    # 5. Advanced Settings (Filtering & Thresholds)
    with st.expander("進階設定 (Advanced Settings)"):
        st.markdown("### 1. 去重設定")
        st.slider(
            "本地相似度閾值 (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.dedup_threshold * 100),
            key='threshold_slider',
            step=1,
            on_change=update_threshold_from_slider,
            help="設定本地端字串比對的相似度門檻。數值越高，比對越嚴格。"
        )

        st.markdown("---")
        st.markdown("### 2. 權重篩選設定")
        
        # Default Weights are defined at module level
        
        enable_custom_weights = st.checkbox("啟用自訂權重篩選 (Enable Custom Weights)", value=False, key="enable_custom_weights")
        
        # UI for adjusting weights
        if enable_custom_weights:
            st.caption("請在下方表格直接編輯關鍵字與權重。支援新增、刪除與修改。")
            
            # Initialize Default DataFrames in Session State if strictly needed specific initialization
            # But simpler: Construct on fly if not in editor state? 
            # Actually st.data_editor handles persistence via key well.
            # We just need to parse the dicts to DF for 'data' argument.
            # To prevent reset, we use a cached function or just create it. 
            # Note: If we pass a new DF every run, data_editor might reset if key is not enough? 
            # Experimental feature: If data changes, it might reset. modify behavior is safer.
            # Let's initialize in session_state ONCE.

            if 'df_neg_keywords' not in st.session_state:
                st.session_state.df_neg_keywords = pd.DataFrame(
                    list(DEFAULT_NEG_KEYWORDS.items()), columns=["關鍵字", "權重"]
                )
            if 'df_com_keywords' not in st.session_state:
                st.session_state.df_com_keywords = pd.DataFrame(
                    list(DEFAULT_COM_KEYWORDS.items()), columns=["關鍵字", "權重"]
                )

            # 1. Negative Keywords
            with st.expander("負面行為權重 (Negative Weights)", expanded=False):
                edited_neg_df = st.data_editor(
                    st.session_state.df_neg_keywords,
                    num_rows="dynamic",
                    column_config={
                        "關鍵字": st.column_config.TextColumn(required=True),
                        "權重": st.column_config.NumberColumn(min_value=0, max_value=10, step=1, required=True)
                    },
                    key="editor_neg_keywords",
                    use_container_width=True
                )
            
            # 2. Commentator Keywords
            with st.expander("評論角色權重 (Commentator Weights)", expanded=False):
                edited_com_df = st.data_editor(
                    st.session_state.df_com_keywords,
                    num_rows="dynamic",
                    column_config={
                        "關鍵字": st.column_config.TextColumn(required=True),
                        "權重": st.column_config.NumberColumn(min_value=0, max_value=10, step=1, required=True)
                    },
                    key="editor_com_keywords",
                    use_container_width=True
                )
            
            # 3. Threshold Multiplier
            st.slider("篩選門檻倍數 (Multiplier)", 1.0, 3.0, 1.5, 0.1, key="filter_threshold_mult", help="保留條件：負面分數 > 評論分數 * 倍數。倍數越高越嚴格。")

        st.markdown("---")
        debug_mode = st.checkbox("啟用除錯模式 (Debug Mode)", value=False, key="debug_mode_checkbox")

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
        
        # Save raw fetch results first
        st.session_state.current_results = all_rss_items

        if st.session_state.current_results and not st.session_state.stop_scan:
            
            # --- 1. Mandatory Local Deduplication (Exact + Fuzzy) ---
            with st.spinner("正在進行初步去重 (本地運算)..."):
                # Step 1.1: Exact Dedup (normalize_title)
                original_count = len(st.session_state.current_results)
                dedup_map = {}
                for item in st.session_state.current_results:
                    norm_t = llm_service.normalize_title(item['title'])
                    if norm_t not in dedup_map:
                        dedup_map[norm_t] = item
                exact_items = list(dedup_map.values())
                exact_removed = original_count - len(exact_items)
                
                # Step 1.2: Fuzzy Dedup (Local Mode)
                threshold = st.session_state.get('dedup_threshold', 0.7)
                unique_results, fuzzy_removed = llm_service.cluster_similar_titles(
                    exact_items, 
                    apikey=None, # Force Local Mode
                    threshold=threshold
                )
                
                st.session_state.current_results = unique_results
                total_removed = exact_removed + fuzzy_removed
                
                if total_removed > 0:
                    st.session_state.screening_status = f"✅ 本地去重完成！已移除 {total_removed} 筆 (完全重複: {exact_removed}, 相似: {fuzzy_removed})。"
                    log_container.markdown(f"<div style='color: gray; margin-bottom: 4px;'>� [本地去重] 已移除 {total_removed} 筆 (完全重複:{exact_removed}, 相似:{fuzzy_removed})</div>", unsafe_allow_html=True)

            # --- 2. Conditional AI Title Screening ---
            config = st.session_state.current_job_config
            # Get API Key: Try session first, then config fallback
            api_key = st.session_state.get('apikey') 
            if not api_key:
                api_key = config.get('apikey') # Fallback if stored in config
            
            if api_key:
                 try:
                    status_container.write("正在進行 AI 標題快篩...")
                    with st.spinner("AI 正在分析標題關聯性..."):
                        screened_items = llm_service.screen_titles(
                            st.session_state.current_results, 
                            config['keywords'], 
                            api_key, 
                            provider=config.get('provider', 'Perplexity')
                        )
                        removed_count_ai = len(st.session_state.current_results) - len(screened_items)
                        st.session_state.current_results = screened_items
                        
                        ai_msg = f"AI title screening complete. Removed {removed_count_ai} items."
                        status_container.write(f"{ai_msg} Remaining: {len(screened_items)}.")
                        
                        # Update persistent status if items were removed
                        if removed_count_ai > 0:
                            current_status = st.session_state.get('screening_status', '')
                            st.session_state.screening_status = f"{current_status} 🤖 AI 快篩移除 {removed_count_ai} 筆。"
                            log_container.markdown(f"<div style='color: blue; margin-bottom: 4px;'>� [AI 快篩] 自動移除 {removed_count_ai} 筆不相關新聞</div>", unsafe_allow_html=True)
                            
                 except Exception as e:
                    st.error(f"AI Screening error: {e}")
                    logger.error(f"Title screening failed: {e}")
            else:
                status_container.write("Skipping AI title screening (no API key provided).")

        # --- 3. Transition to Metadata Review ---
        if st.session_state.current_results:
            status_container.update(label=f"Metadata fetch complete. Found {len(st.session_state.current_results)} items.", state="complete", expanded=False)
            st.session_state.workflow_stage = 'metadata_review'
            st.rerun()
        else:
            status_container.update(label="No items found.", state="error")
            st.warning("No items found. Please try different keywords or date range.")
            if st.button("Back to Config"):
                st.session_state.workflow_stage = 'config'
                st.rerun()

    except Exception as e:
        st.error(f"Error during metadata fetch: {e}")
        # Allow retry/reset by showing back button or keeping in error state
        if st.button("Back to Config", key="error_back"):
            st.session_state.workflow_stage = 'config'
            st.rerun()

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
        
        if "screening_status" not in st.session_state:
            st.session_state.screening_status = ""
            
        st.info(f"Found {len(st.session_state.current_results)} potential items. Click 'Continue' to fetch content and analyze, or 'Reset' to start over.")
        
        # Action Buttons in 4 columns
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            btn_start = st.button("🚀 開始內容分析", type="primary", use_container_width=True)
            
        with col_btn2:
            btn_rescreen = st.button("重做標題快篩", use_container_width=True, help="使用 AI 重新過濾標題")

        with col_btn3:
            btn_dedup = st.button("合併相似標題", use_container_width=True, help="合併語意相近的新聞標題")
                            
        with col_btn4:
            btn_reset = st.button("Reset", type="secondary", use_container_width=True)

        # Process button actions outside columns
        if btn_start:
            st.session_state.workflow_stage = 'processing'
            st.session_state.processing_index = 0
            st.session_state.screening_status = "" 
            st.rerun()

        if btn_rescreen:
            apikey = st.session_state.get('apikey', '') 
            if not apikey:
                st.session_state.screening_status = "⚠️ 請在側邊欄輸入 API Key 以使用此功能"
            elif st.session_state.current_results:
                with st.spinner("AI 正在重新篩選標題..."):
                    try:
                        config = st.session_state.current_job_config
                        screeneditems = llm_service.screen_titles(
                            st.session_state.current_results, 
                            config['keywords'], 
                            apikey, 
                            config.get('provider', 'Perplexity')
                        )
                        removed_count = len(st.session_state.current_results) - len(screeneditems)
                        st.session_state.current_results = screeneditems
                        st.session_state.screening_status = f"✅ 標題快篩完成！篩選掉 {removed_count} 筆，保留 {len(screeneditems)} 筆"
                        st.rerun()
                    except Exception as e:
                        st.session_state.screening_status = f"❌ 標題快篩失敗：{e}"
                        logger.error(f"Redo screening failed: {e}")

        if btn_dedup:
            if st.session_state.current_results:
                with st.spinner("正在進行語意分析與合併..."):
                    try:
                        apikey = st.session_state.get('apikey', '')
                        provider = st.session_state.current_job_config.get('provider', 'OpenAI')
                        original_count = len(st.session_state.current_results)
                        deduped_items, removed_count = llm_service.cluster_similar_titles(
                            st.session_state.current_results, 
                            apikey, 
                            provider
                        )
                        st.session_state.current_results = deduped_items
                        st.session_state.screening_status = f"✅ 合併完成！原始 {original_count} 筆，移除 {removed_count} 筆，剩餘 {len(deduped_items)} 筆。"
                        st.rerun()
                    except Exception as e:
                        st.session_state.screening_status = f"❌ 合併失敗：{e}"
                        logger.error(f"Semantic deduplication failed: {e}")

        if btn_reset:
            st.session_state.current_results = []
            st.session_state.workflow_stage = 'config'
            st.session_state.screening_status = "" 
            st.rerun()
                
        # Persistent Status Message (Left-aligned)
        if st.session_state.screening_status:
            st.markdown(f"""
                <div style='text-align: left; padding: 12px; background-color: #f0f2f6; border-radius: 8px; border-left: 5px solid #0068c9; margin-top: 10px; font-family: sans-serif; font-size: 0.95em;'>
                    {st.session_state.screening_status}
                </div>
            """, unsafe_allow_html=True)

# --- Stage 2: Processing (Transient) ---
if stage == 'processing':
    st.divider()
    st.markdown("## Stage 2: Content Fetching & Analysis")
    
    # Progress tracking
    status_container = st.status("Processing...", expanded=True)
    progress_bar = st.progress(0)
    
    try:
        items_proc = st.session_state.current_results
        total_items = len(items_proc)
        results = st.session_state.analysis_results
        
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

            # === Batch Logic ===
            start_idx = st.session_state.processing_index
            end_idx = min(start_idx + 100, total_items)
            status_container.write(f"Processing Batch: {start_idx+1} to {end_idx} (Total: {total_items})")

            for idx in range(start_idx, end_idx):
                item = items_proc[idx]
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
                    if 'full_text' not in item:
                        item = scraper_web.fetch_content_single(item)
                        st.session_state.current_results[idx] = item
                    
                    # 2.2 Analyze
                    item_results, debug_info = llm_service.analyze_single_item(
                        item, 
                        config_api_key,
                        provider=config_provider,
                        filter_config=config.get('filter_config')
                    )

                    if debug_info.get("error"):
                        error_msg = f"LLM Error: {debug_info['error']}"
                        st.error(f"{item['title']}: {error_msg}")
                        st.session_state.error_log.append(error_msg)
                        update_logs(f"❌ {error_msg}")
                    
                    # Debug Inspector
                    if debug_mode:
                        with st.expander(f"🕵️ Debug: Check Input Content for '{item['title']}'"):
                            st.markdown(f"**Method:** `{debug_info.get('scraping_method', 'Unknown')}`")
                            st.markdown("### Input Text Preview")
                            st.text(debug_info.get('input_content', 'No content')[:1000])
                            st.markdown("### LLM Raw Response")
                            st.code(debug_info.get('raw_response', 'No response'), language='json')
                    
                    if item_results:
                        for res in item_results: res['title'] = item['title']
                        results.extend(item_results)
                        update_logs(f"✅ Extracted {len(item_results[0]['names'])} entities from '{item['title']}'")
                    
                    # Checkpoint
                    st.session_state.analysis_results = results
                    st.session_state.processing_index = idx + 1
                    st.session_state.export_bytes = generate_snapshot_excel(results)
                        
                except Exception as e:
                    error_msg = f"Error processing item {idx + 1}: {e}"
                    st.error(error_msg)
                    st.session_state.error_log.append(error_msg)
                    logger.error(f"Error in Stage 2 loop for item {item.get('title', 'Unknown')}: {e}")
                    update_logs(f"❌ {error_msg}")
                    st.session_state.processing_index = idx + 1
                    continue
            
            if st.session_state.stop_scan:
                status_container.update(label="Paused.", state="error")
                
                col_pause1, col_pause2 = st.columns(2)
                with col_pause1:
                    if st.button("▶️ Resume Scan", type="primary", use_container_width=True):
                        st.session_state.stop_scan = False
                        st.rerun()
                
                with col_pause2:
                    if st.session_state.analysis_results:
                        # Refresh export bytes with latest results
                        st.session_state.export_bytes = generate_snapshot_excel(st.session_state.analysis_results)
                        fname = f"negative_news_snapshot_paused_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        st.download_button(
                            label="📥 Download Snapshot",
                            data=st.session_state.export_bytes,
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="snapshot_dl_paused",
                            use_container_width=True
                        )
            elif st.session_state.processing_index < total_items:
                st.rerun()
            else:
                progress_bar.progress(1.0)
                st.session_state.workflow_stage = 'completed'
                stage = 'completed'
                st.rerun()

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
        st.session_state.export_bytes = None
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

    # Snapshot Export (Always Available if data exists)
    if st.session_state.export_bytes:
        fname = f"negative_news_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            label="📥 Download Snapshot",
            data=st.session_state.export_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="snapshot_dl"
        )

# Footer
st.markdown("---")
st.markdown("Sophistication, radiance and elegance | Jennifer")
