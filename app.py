import streamlit as st
from PIL import Image
import pandas as pd
import io
import time
from datetime import datetime, timedelta
import scraper_web
import pplx_service
import importlib
import re
import os

# Force reload for development
importlib.reload(scraper_web)
importlib.reload(pplx_service)

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
    page_title="Negative News Smart Scan",
    page_icon=peashooter_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.session_state.available_keywords = [
        "貪污", "詐騙", "洗錢", "制裁", "稅務犯罪", 
        "證券犯罪", "販毒", "人口販運", "走私", "謀殺"
    ]
if 'selected_keywords' not in st.session_state:
    st.session_state.selected_keywords = st.session_state.available_keywords.copy()

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
            <h1 style="margin: 0; padding: 0; font-size: 3rem; line-height: 1.2;">Negative News Smart Scan</h1>
            <p style="margin: 0; font-size: 1.5rem; font-weight: 500; align-self: flex-start;">快點做完  回家喝奶茶！</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key (No default)
    api_key = st.text_input("API Key", type="password", help="Enter your API key here.")
    
    # === Keyword Management Section ===
    st.subheader("Keyword Management")
    
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
                label_visibility="collapsed",
                key="remove_keyword_select",
                help="Permanently remove keyword from available list"
            )
        with col_del_btn:
            if st.button("🗑️", key="remove_from_available", help="Delete from library", use_container_width=True):
                if keyword_to_remove in st.session_state.available_keywords:
                    st.session_state.available_keywords.remove(keyword_to_remove)
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
    
    if not final_keywords:
        st.warning("Please select or enter at least one keyword.")
    else:
        st.subheader("1. News Aggregation")
        status_container = st.status("Scanning news sources...", expanded=True)
        progress_bar = st.progress(0)
        
        all_news_items = []
        
        # Handle date range
        if isinstance(date_range, tuple):
            start_date = date_range[0]
            end_date = date_range[1] if len(date_range) > 1 else start_date
        else:
            start_date = date_range
            end_date = date_range

        for idx, keyword in enumerate(final_keywords):
            # Check for stop signal
            if st.session_state.stop_scan:
                status_container.update(label="Scanning stopped by user.", state="error")
                st.warning("Scan stopped by user.")
                break
                
            status_container.write(f"Scanning for '{keyword}'...")
            
            try:
                news = scraper_web.fetch_news([keyword], start_date=start_date, end_date=end_date)
                
                # Filter: Must have Chinese characters in title
                filtered_news = [n for n in news if has_chinese(n['title'])]
                all_news_items.extend(filtered_news)
                
            except Exception as e:
                st.error(f"Error scanning {keyword}: {e}")
            
            # Update progress
            progress = (idx + 1) / len(final_keywords)
            progress_bar.progress(progress)
            
        if not st.session_state.stop_scan:
            status_container.update(label="Scraping complete!", state="complete", expanded=False)
            st.session_state.scraped_news = all_news_items
            
            if not all_news_items:
                st.warning("No relevant news found.")
            else:
                st.success(f"Found {len(all_news_items)} relevant news items.")

# Display Scraped Data & Action Buttons
if st.session_state.scraped_news:
    st.subheader("Raw Data Preview")
    
    # Export raw data button
    raw_df = pd.DataFrame(st.session_state.scraped_news)
    raw_output = io.BytesIO()
    with pd.ExcelWriter(raw_output, engine='xlsxwriter') as writer:
        raw_df.to_excel(writer, index=False, sheet_name='Raw Data')
    raw_excel_data = raw_output.getvalue()
    
    col_exp, col_export = st.columns([5, 1])
    with col_exp:
        with st.expander("View Scraped News", expanded=False):
            st.dataframe(raw_df)
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
        col_btns = st.columns([1, 4])
        with col_btns[0]:
            continue_btn = st.button("Continue Analysis", type="primary")
        with col_btns[1]:
            reset_main_btn = st.button("Reset", type="secondary")
        
    if reset_main_btn:
        st.session_state.scraped_news = []
        st.session_state.analysis_results = []
        st.session_state.is_scanning = False
        st.session_state.stop_scan = False
        st.rerun()
    
    if continue_btn:
        # Clear buttons to prevent duplication/clutter during analysis
        button_placeholder.empty()
        
        if not api_key:
            st.error("Please enter your Perplexity API Key in the sidebar to proceed.")
        else:
            st.subheader("2. AI Intelligence Analysis")
            with st.spinner("Analyzing content with Perplexity AI..."):
                try:
                    # Run Analysis
                    results = pplx_service.analyze_news(st.session_state.scraped_news, api_key)
                    st.session_state.analysis_results = results
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

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
            
            # Auto-adjust columns width
            workbook = writer.book
            worksheet = writer.sheets['Intelligence Report']
            for i, col in enumerate(df_display.columns):
                column_len = max(df_display[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_len)
                
        excel_data = output.getvalue()
        file_name = f"Negative_News_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        
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
st.markdown("© 2024 Negative News Smart Scan | Powered by Streamlit & Perplexity AI")
