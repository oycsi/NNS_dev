import requests
import urllib.parse
import time
import logging
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import concurrent.futures
import trafilatura
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_rss_items(keywords, start_date, end_date):
    """
    Phase 1: Fetches news titles/metadata from Google News RSS.
    Does NOT fetch full content.
    
    Args:
        keywords (list): List of keywords to search for.
        start_date (date): Start date for the search.
        end_date (date): End date for the search.
        
    Returns:
        list: A list of dictionaries containing news metadata (title, link, pub_date, source, summary).
    """
    all_items = []
    
    # Base URL for Google News RSS
    base_url = "https://news.google.com/rss/search"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Format dates for query (YYYY-MM-DD)
    after_str = start_date.strftime("%Y-%m-%d")
    end_date_inclusive = end_date + timedelta(days=1)
    before_str = end_date_inclusive.strftime("%Y-%m-%d")

    for keyword in keywords:
        try:
            logging.info(f"Fetching RSS for keyword: {keyword}")
            
            query = f"{keyword} after:{after_str} before:{before_str}"
            params = {
                "q": query,
                "hl": "zh-TW",
                "gl": "TW",
                "ceid": "TW:zh-Hant"
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            items = root.findall(".//item")
            logging.info(f"Found {len(items)} RSS items for {keyword}")
            
            for item in items:
                title = item.find("title").text if item.find("title") is not None else "No Title"
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                description = item.find("description").text if item.find("description") is not None else ""
                source = item.find("source").text if item.find("source") is not None else "Unknown"
                
                # Basic deduplication based on link
                if any(n['link'] == link for n in all_items):
                    continue
                    
                news_item = {
                    "keyword": keyword,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                    "summary": description, # Initial summary from RSS
                    "full_text": "" # To be populated in Phase 2
                }
                
                all_items.append(news_item)
                
            # Be nice to the server
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error fetching RSS for {keyword}: {e}")
            
    return all_items

def fetch_content_batch(news_items):
    """
    Phase 2: Fetches full content for a list of news items using Trafilatura.
    
    Args:
        news_items (list): List of news item dictionaries (filtered from Phase 1).
        
    Returns:
        list: The same list with 'full_text' populated. Items where fetch failed or content is too short might be filtered out or marked.
    """
    logging.info(f"Starting content fetch for {len(news_items)} items...")
    
    # Use ThreadPoolExecutor for concurrent content fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_item = {executor.submit(extract_content_trafilatura, item['link']): item for item in news_items}
        
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                content = future.result()
                if content and len(content) > 50:
                    item['full_text'] = content
                else:
                    # Fallback to summary
                    logging.warning(f"Using summary fallback for {item['title']}")
                    summary = item.get('summary', '')
                    # Clean summary HTML
                    if summary:
                        soup = BeautifulSoup(summary, "html.parser")
                        item['full_text'] = soup.get_text()
                    else:
                        item['full_text'] = item['title'] # Last resort
            except Exception as e:
                logging.error(f"Error fetching content for {item['title']}: {e}")
                item['full_text'] = item.get('summary', '') or item['title']

    # Filter out items that are still empty (should be rare now)
    valid_items = [item for item in news_items if item['full_text']]
    logging.info(f"Content fetch complete. {len(valid_items)} valid items retained out of {len(news_items)}.")
    return valid_items

def extract_content_trafilatura(url):
    """
    Extracts main content using Trafilatura.
    Uses requests to fetch HTML first to handle Google News redirects and User-Agent.
    """
    try:
        # Use requests to fetch the page (handles redirects automatically)
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        logging.info(f"Fetched {url} -> {response.url} (Status: {response.status_code})")
        
        # Extract content from the HTML
        result = trafilatura.extract(response.content, include_comments=False, include_tables=False, no_fallback=False)
        if not result:
            logging.warning(f"Trafilatura returned empty result for {response.url}")
        if result:
            return result
        return ""
    except Exception as e:
        logging.error(f"Error extracting content from {url}: {e}")
        return ""

# Legacy function wrapper for backward compatibility if needed (though we should update app.py)
def fetch_news(keywords, start_date, end_date):
    items = fetch_rss_items(keywords, start_date, end_date)
    return fetch_content_batch(items)

if __name__ == "__main__":
    # Test run
    test_keywords = ["詐騙"]
    today = datetime.now().date()
    start_date = today - timedelta(days=1)
    
    print("Phase 1: Fetching RSS...")
    rss_items = fetch_rss_items(test_keywords, start_date, today)
    print(f"Fetched {len(rss_items)} RSS items.")
    
    if rss_items:
        print("Phase 2: Fetching Content for top 3 items...")
        # Test with just a few
        content_items = fetch_content_batch(rss_items[:3])
        for item in content_items:
            print(f"\nTitle: {item['title']}")
            print(f"Content Length: {len(item['full_text'])}")
            print(f"Preview: {item['full_text'][:100]}...")
