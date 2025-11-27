import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import logging
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_news(keywords, start_date, end_date):
    """
    Fetches news from Google News RSS for the given keywords within a date range.
    
    Args:
        keywords (list): List of keywords to search for.
        start_date (date): Start date for the search.
        end_date (date): End date for the search.
        
    Returns:
        list: A list of dictionaries containing news details.
    """
    all_news = []
    
    # Base URL for Google News RSS
    base_url = "https://news.google.com/rss/search"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Format dates for query (YYYY-MM-DD)
    after_str = start_date.strftime("%Y-%m-%d")
    before_str = end_date.strftime("%Y-%m-%d")

    for keyword in keywords:
        try:
            logging.info(f"Fetching news for keyword: {keyword}")
            
            # Construct query with date range operators
            # Note: 'before' is exclusive in some contexts, but for search it usually includes the day or up to it.
            # To be safe for inclusive range, we might want to set before to end_date + 1 day, 
            # but standard search usually treats it as inclusive of the date provided or up to the next.
            # Let's stick to the user provided dates first.
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
            logging.info(f"Found {len(items)} items for {keyword}")
            
            for item in items:
                title = item.find("title").text if item.find("title") is not None else "No Title"
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                description = item.find("description").text if item.find("description") is not None else ""
                source = item.find("source").text if item.find("source") is not None else "Unknown"
                
                # Basic deduplication based on link
                if any(n['link'] == link for n in all_news):
                    continue
                    
                news_item = {
                    "keyword": keyword,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                    "summary": description, # Initial summary from RSS
                    "full_text": "" # To be populated if possible
                }
                
                all_news.append(news_item)
                
            # Be nice to the server
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error fetching news for {keyword}: {e}")
            
    # Use ThreadPoolExecutor for concurrent content fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a dictionary to map futures to news items
        future_to_item = {
            executor.submit(extract_article_content, item['link']): item 
            for item in all_news
        }
        
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                full_text = future.result()
                item['full_text'] = full_text
            except Exception as e:
                logging.error(f"Concurrency error for {item['link']}: {e}")
                item['full_text'] = "Failed to fetch"

    return all_news

def extract_article_content(url):
    """
    Attempts to extract the main content from a news article URL.
    Note: Google News links are redirects. Requests handles redirects automatically.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Simple heuristic to get the main body
        if not text:
            return "Failed to fetch"
            
        return text[:5000] 
        
    except Exception as e:
        logging.error(f"Error extracting content from {url}: {e}")
        return "Failed to fetch"

if __name__ == "__main__":
    # Test run
    test_keywords = ["詐騙"]
    today = datetime.now().date()
    start_date = today - timedelta(days=1)
    news = fetch_news(test_keywords, start_date, today)
    print(f"Fetched {len(news)} items.")
    if news:
        print("First item:", news[0])
