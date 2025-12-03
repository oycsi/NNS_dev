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
if 'scraped_news' not in st.session_state:
    st.session_state.scraped_news = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False
if 'stop_scan' not in st.session_state:
    st.session_state.stop_scan = False
if 'available_keywords' not in st.session_state:
    # Try to load from local storage
    stored_keywords = localS.getItem("user_keywords")
    if stored_keywords and isinstance(stored_keywords, list):
        st.session_state.available_keywords = stored_keywords
    else:
        st.session_state.available_keywords = [
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
if 'selected_keywords' not in st.session_state:
    st.session_state.selected_keywords = st.session_state.available_keywords.copy()
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "Perplexity"
if 'scan_completed' not in st.session_state:
    st.session_state.scan_completed = False
if 'expander_state' not in st.session_state:
    st.session_state.expander_state = False
if 'show_aggregation' not in st.session_state:
    st.session_state.show_aggregation = False
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Helper function to check for Chinese characters
def has_chinese(text):
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', text))

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
    
    # LLM Provider Selection
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Perplexity", "OpenAI", "Google Gemini", "Meta Llama (Together.ai)"],
        index=["Perplexity", "OpenAI", "Google Gemini", "Meta Llama (Together.ai)"].index(st.session_state.llm_provider),
        help="Select your LLM API provider"
    )
    st.session_state.llm_provider = llm_provider
    
    # API Key (No default)
    api_key = st.text_input(
        f"{llm_provider} API Key", 
        type="password", 
        help=f"Enter your {llm_provider} API key here."
    )
    
    # === Keyword Management Section ===
    st.subheader("Keyword")
    
    # Multiselect for keyword management (at the top)
    selected = st.multiselect(
        "Selected Keywords",
        options=st.session_state.available_keywords,
        default=st.session_state.selected_keywords,
        help="Select/deselect keywords by clicking. Remove by clicking the X on each tag."
    )
    
    # Check if selection changed and update
    if selected != st.session_state.selected_keywords:
        st.session_state.selected_keywords = selected
        st.rerun()
    
    # Add new keyword input (second position)
    st.text("")  # Empty text for spacing
    st.markdown("<p style='margin-bottom: 0.5rem; font-size: 0.875rem; font-weight: 400;'>Keyword Management</p>", unsafe_allow_html=True)
    col_add_input, col_add_btn = st.columns([6, 1])
    with col_add_input:
        new_keyword = st.text_input(
            "Add New Keyword", 
            label_visibility="collapsed", 
            placeholder="Add new keyword...", 
            key="new_keyword_input"
        )
    with col_add_btn:
        add_kw_btn = st.button("➕", key="add_new_keyword", help="Add keyword", use_container_width=True)
        
    if add_kw_btn and new_keyword:
        keyword_stripped = new_keyword.strip()
        if keyword_stripped:
            # Add to available keywords if not exists
            if keyword_stripped not in st.session_state.available_keywords:
                st.session_state.available_keywords.append(keyword_stripped)
                # Save to local storage
                localS.setItem("user_keywords", st.session_state.available_keywords)
            # Add to selected keywords if not already selected
            if keyword_stripped not in st.session_state.selected_keywords:
                st.session_state.selected_keywords.append(keyword_stripped)
                st.rerun()
    
    # Delete from available keywords (third position)
    if st.session_state.available_keywords:
        col_del_select, col_del_btn = st.columns([6, 1])
        with col_del_select:
            keyword_to_remove = st.selectbox(
                "Remove from library",
                options=st.session_state.available_keywords,
                index=None,
                placeholder="Select a keyword to delete",
                label_visibility="collapsed",
                key="remove_keyword_select",
                help="Permanently remove keyword from available list"
            )
        with col_del_btn:
            if st.button("🗑️", key="remove_from_available", help="Delete from library", use_container_width=True):
                if keyword_to_remove and keyword_to_remove in st.session_state.available_keywords:
                    st.session_state.available_keywords.remove(keyword_to_remove)
                    # Save to local storage
                    localS.setItem("user_keywords", st.session_state.available_keywords)
                    # Also remove from selected if present
                    if keyword_to_remove in st.session_state.selected_keywords:
                        st.session_state.selected_keywords.remove(keyword_to_remove)
                    st.rerun()
    
    # Final list for scraping
    final_keywords = st.session_state.selected_keywords
    
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
        start_btn = st.button("Start Scan", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("Stop", type="secondary", use_container_width=True)

    if stop_btn:
        st.session_state.stop_scan = True

# Main Logic

# 1. Scraping Phase
if start_btn:
    st.session_state.scraped_news = [] # Reset on new scan
    st.session_state.analysis_results = []
    st.session_state.stop_scan = False
    st.session_state.scan_completed = False  # Reset completion flag
    st.session_state.show_aggregation = False  # Will be set to True after scan completes
    st.session_state.show_analysis = False  # Reset analysis section
    
    if not final_keywords:
        st.warning("Please select or enter at least one keyword.")
    else:
        st.subheader("1. News Aggregation & Screening")
        status_container = st.status("Starting scan...", expanded=True)
        progress_bar = st.progress(0)
        
        all_news_items = []
        
        # Handle date range
        if isinstance(date_range, tuple):
            start_date = date_range[0]
            end_date = date_range[1] if len(date_range) > 1 else start_date
        else:
            start_date = date_range
            end_date = date_range

        # Phase 1: Global Collection
        master_rss_items = []
        
        for idx, keyword in enumerate(final_keywords):
            # Check for stop signal
            if st.session_state.stop_scan:
                break
                
            # Phase 1: RSS Fetching
            status_container.write(f"抓取關鍵字 : {keyword} ...")
            try:
                rss_items = scraper_web.fetch_rss_items([keyword], start_date=start_date, end_date=end_date)
                
                # Filter: Must have Chinese characters in title
                rss_items = [n for n in rss_items if has_chinese(n['title'])]
                
                status_container.write(f"已完成 關鍵字抓取: {keyword} (共抓取 {len(rss_items)} 篇)")
                master_rss_items.extend(rss_items)
                
            except Exception as e:
                st.error(f"Error scanning {keyword}: {e}")
            
            # Update progress (Phase 1 accounts for 50% of progress)
            progress = (idx + 1) / len(final_keywords) * 0.5
            progress_bar.progress(progress)

        # Deduplicate Master List
        unique_items = []
        seen_links = set()
        for item in master_rss_items:
            if item['link'] not in seen_links:
                unique_items.append(item)
                seen_links.add(item['link'])
        
        status_container.write(f"全域抓取完成。共收集 {len(unique_items)} 篇新聞。")
        
        if not st.session_state.stop_scan and unique_items:
            # Phase 1.5: Global Batch Screening
            if api_key:
                status_container.write(f"正在進行全域篩選 (共 {len(unique_items)} 篇)...")
                screened_items = llm_service.screen_titles(
                    unique_items, 
                    final_keywords, 
                    api_key, 
                    provider=st.session_state.llm_provider
                )
                status_container.write(f"全域篩選完成。保留 {len(screened_items)} 篇新聞。")
            else:
                status_container.write(f"跳過篩選 (無 API Key)。保留 {len(unique_items)} 篇。")
                screened_items = unique_items
            
            progress_bar.progress(0.75)
            
            if screened_items:
                # Phase 2: Global Content Fetching
                status_container.write(f"正在全域下載內文 (共 {len(screened_items)} 篇)...")
                all_news_items = scraper_web.fetch_content_batch(screened_items)
                status_container.write(f"全域內文下載完成。")
                progress_bar.progress(1.0)
            else:
                all_news_items = []
        else:
            all_news_items = unique_items # If stopped or empty, just keep what we have (though content might be missing)
            
        # Handle stop scan scenario
        if st.session_state.stop_scan:
            status_container.update(label="Scanning stopped by user.", state="error")
            st.session_state.scraped_news = all_news_items
            st.session_state.scan_completed = True 
            st.session_state.show_aggregation = True 
            
            if not all_news_items:
                st.warning("No relevant news found (scan stopped).")
            else:
                st.success(f"Found {len(all_news_items)} relevant news items (scan stopped).")
                
        elif not st.session_state.stop_scan:
            status_container.update(label="Scraping & Screening complete!", state="complete", expanded=True)
            st.session_state.scraped_news = all_news_items
            st.session_state.scan_completed = True  # Mark scan as completed
            st.session_state.show_aggregation = True  # Show aggregation permanently
            
            if not all_news_items:
                st.warning("No relevant news found.")
            else:
                st.success(f"Found {len(all_news_items)} relevant news items.")
                
        # Rerun removed to keep logs visible
        # st.rerun()

# Display News Aggregation section when it should be shown (Persistent Section)
# We only show this if we are NOT currently running a new scan (start_btn is False)
if st.session_state.show_aggregation and st.session_state.scan_completed:
    st.subheader("1. News Aggregation")
    st.success(f"Found {len(st.session_state.scraped_news)} relevant news items.")
    
# Display Scraped Data & Action Buttons
if st.session_state.scraped_news and st.session_state.scan_completed:
    st.subheader("Raw Data Preview")
    
    # Export raw data button
    raw_df = pd.DataFrame(st.session_state.scraped_news)
    raw_output = io.BytesIO()
    with pd.ExcelWriter(raw_output, engine='xlsxwriter') as writer:
        raw_df.to_excel(writer, index=False, sheet_name='Raw Data')
        
        # Auto-adjust columns width with max 72
        workbook = writer.book
        worksheet = writer.sheets['Raw Data']
        for i, col in enumerate(raw_df.columns):
            column_len = max(raw_df[col].astype(str).map(len).max(), len(col)) + 2
            column_len = min(column_len, 72)  # Cap at 72
            worksheet.set_column(i, i, column_len)
    
    raw_excel_data = raw_output.getvalue()
    
    col_exp, col_export = st.columns([5, 1])
    with col_exp:
        with st.expander("View Scraped News", expanded=st.session_state.expander_state):
            st.dataframe(raw_df)
            # Track if user manually changed expander state
            if st.session_state.expander_state != st.session_state.get('prev_expander_state', False):
                st.session_state.prev_expander_state = st.session_state.expander_state
    with col_export:
        st.download_button(
            label="Export",
            data=raw_excel_data,
            file_name=f"Raw_Data_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary"
        )
        
    st.divider()
    
    col_analyze, col_reset = st.columns([1, 4])
    
    # Use a placeholder to manage button visibility
    button_placeholder = st.empty()
    
    with button_placeholder.container():
        col_btns = st.columns([1, 1])
        with col_btns[0]:
            continue_btn = st.button("Continue", type="primary", use_container_width=True)
        with col_btns[1]:
            reset_main_btn = st.button("Reset", type="secondary", use_container_width=True)
        
    if reset_main_btn:
        st.session_state.scraped_news = []
        st.session_state.analysis_results = []
        st.session_state.is_scanning = False
        st.session_state.stop_scan = False
        st.session_state.scan_completed = False
        st.session_state.show_aggregation = False
        st.session_state.show_analysis = False
        st.session_state.expander_state = False
        st.rerun()
    
    if continue_btn:
        # Clear buttons to prevent duplication/clutter during analysis
        button_placeholder.empty()
        st.session_state.show_analysis = True  # Show analysis section
        
        if not api_key:
            st.error(f"Please enter your {st.session_state.llm_provider} API Key in the sidebar to proceed.")
        else:
            st.subheader("2. AI Intelligence Analysis")
            with st.spinner(f"Analyzing content with {st.session_state.llm_provider}..."):
                try:
                    # Run Analysis
                    results = llm_service.analyze_news(
                        st.session_state.scraped_news, 
                        api_key,
                        provider=st.session_state.llm_provider
                    )
                    st.session_state.analysis_results = results
                    if not results:
                        st.warning("Analysis completed but no valid events with named individuals were found.")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse {st.session_state.llm_provider} response. The API returned invalid JSON.")
                    st.error(f"Error details: {str(e)}")
                    if st.button("Retry", use_container_width=True, type="primary", key="retry_json"):
                        st.rerun()
                except openai.AuthenticationError:
                    st.error(f"❌ Authentication failed. Please check your {st.session_state.llm_provider} API key.")
                    if st.button("Retry", use_container_width=True, type="primary", key="retry_auth"):
                        st.rerun()
                except openai.RateLimitError:
                    st.error(f"❌ Rate limit exceeded for {st.session_state.llm_provider}. Please wait and try again.")
                    if st.button("Retry", use_container_width=True, type="primary", key="retry_rate"):
                        st.rerun()
                except openai.APIError as e:
                    st.error(f"❌ {st.session_state.llm_provider} API error: {str(e)}")
                    if st.button("Retry", use_container_width=True, type="primary", key="retry_api"):
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred during analysis: {str(e)}")
                    st.error("Please check your API key and try again.")
                    import traceback
                    st.code(traceback.format_exc())
                    if st.button("Retry", use_container_width=True, type="primary", key="retry_general"):
                        st.rerun()

# Display Analysis Section Header (persists when editing sidebar)
if st.session_state.show_analysis:
    if not st.session_state.analysis_results:
        st.subheader("2. AI Intelligence Analysis")
        st.info("Analysis in progress or encountered an error. Please check above.")

# Display Analysis Results
if st.session_state.analysis_results:
    st.subheader("3. Intelligence Report")
    
    # Process results to expand names into separate rows
    expanded_results = []
    for result in st.session_state.analysis_results:
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
                    "連結": result.get('source_url', '')
                })
    
    if expanded_results:
        df_display = pd.DataFrame(expanded_results)
        
        # Display Data Grid with Link
        st.data_editor(
            df_display,
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
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Intelligence Report')
            
            # Auto-adjust columns width with max 72
            workbook = writer.book
            worksheet = writer.sheets['Intelligence Report']
            for i, col in enumerate(df_display.columns):
                column_len = max(df_display[col].astype(str).map(len).max(), len(col)) + 2
                column_len = min(column_len, 72)  # Cap at 72
                worksheet.set_column(i, i, column_len)
                
        excel_data = output.getvalue()
        file_name = f"新阿姆斯特朗旋風噴射阿姆斯特朗砲_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        
        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Analysis complete, but no specific named individuals were found in the negative news.")

# Footer
st.markdown("---")
st.markdown("© 2025 😘 | 美麗又大方的珍妮佛 ")
