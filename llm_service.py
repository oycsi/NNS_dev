import openai
import json
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LLM Provider Configuration
LLM_PROVIDERS = {
    "Perplexity": {
        "base_url": "https://api.perplexity.ai",
        "model": "sonar-pro",
        "cheap_model": "sonar" # Lightweight model for screening
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "cheap_model": "gpt-4o-mini"
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash-exp",
        "cheap_model": "gemini-2.0-flash-exp" # Gemini Flash is already cheap/fast
    },
    "Meta Llama (Together.ai)": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "cheap_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    }
}

def is_complete_name(name):
    """
    Check if a name appears to be a complete name (at least 2 characters for Chinese, 
    or has both first and last name for English).
    """
    if not name or not name.strip():
        return False
    
    name = name.strip()
    
    # Chinese name: should be at least 2 characters
    if re.search(r'[\u4e00-\u9fff]', name):
        # Count Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', name)
        if len(chinese_chars) < 2:
            return False
            
        # Exclude common journalistic placeholders (Surname + Gender/Role)
        # e.g., 陳男, 林女, 張某, 李嫌, 王氏
        if re.search(r'^[\u4e00-\u9fff][男女某嫌氏]$', name):
            return False
        
        # Exclude "Surname + 姓 + Role/Description" pattern
        # e.g., 林姓公務員, 陳姓男子
        if re.search(r'^[\u4e00-\u9fff]姓.+$', name):
            return False
            
        return True
    
    # English name: should have at least first and last name (space separated)
    if ' ' in name and len(name.split()) >= 2:
        return True
    
    return False

def screen_titles(news_items, keywords, api_key, provider="Perplexity"):
    """
    Phase 1: Batch Title Screening.
    Uses a lightweight model to filter out irrelevant titles before content fetching.
    
    Args:
        news_items (list): List of news item dictionaries (from RSS).
        keywords (list): List of keywords used for search.
        api_key (str): API Key.
        provider (str): LLM provider.
        
    Returns:
        list: Filtered list of news items that passed the screening.
    """
    if not news_items:
        return []
        
    # Validate provider
    if provider not in LLM_PROVIDERS:
        logging.error(f"Invalid provider: {provider}. Defaulting to Perplexity.")
        provider = "Perplexity"
    
    config = LLM_PROVIDERS[provider]
    base_url = config["base_url"]
    model = config.get("cheap_model", config["model"]) # Use cheap model if available
    
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        logging.info(f"Screening titles using {provider} with model {model}")
    except Exception as e:
        logging.error(f"Failed to initialize {provider} client: {e}")
        return news_items # Fail open (return all) if client init fails
        
    screened_items = []
    
    # Process in batches of 20
    batch_size = 20
    for i in range(0, len(news_items), batch_size):
        batch = news_items[i:i+batch_size]
        
        # Prepare context
        titles_text = ""
        for idx, item in enumerate(batch):
            titles_text += f"{idx+1}. {item['title']} (Source: {item['source']})\n"
            
        keyword_str = ", ".join(keywords)
        
        system_prompt = (
            f"You are a risk screening assistant. Your task is to screen news titles for potential negative news related to: {keyword_str}.\n"
            "Criteria for KEEPING a title (Recall-oriented):\n"
            "1. Mentions crime, scandal, regulation, lawsuit, fraud, corruption, or negative business events.\n"
            "2. Ambiguous titles that MIGHT be negative.\n"
            "3. Mentions specific individuals in a potentially negative context.\n\n"
            "Criteria for DISCARDING:\n"
            "1. Clearly positive news (awards, earnings growth, charity).\n"
            "2. Generic market reports (stock prices up/down) without specific scandal.\n"
            "3. Advertisements or promotional content.\n"
            "4. Sports, entertainment, or lifestyle news unrelated to crime/fraud.\n\n"
            "Output Format:\n"
            "Return ONLY a JSON array of integers representing the indices (1-based) of the titles to KEEP.\n"
            "Example: [1, 3, 5, 12]\n"
            "If no titles are relevant, return []"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": titles_text}
                ],
                temperature=0.0 # Deterministic
            )
            
            content = response.choices[0].message.content
            
            # Clean up markdown
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
                
            indices = json.loads(content)
            
            if isinstance(indices, list):
                for idx in indices:
                    if isinstance(idx, int) and 1 <= idx <= len(batch):
                        screened_items.append(batch[idx-1])
            
        except Exception as e:
            logging.error(f"Error during title screening batch {i}: {e}")
            # If screening fails, keep the whole batch to be safe (Fail Open)
            screened_items.extend(batch)
            
    logging.info(f"Screening complete. Kept {len(screened_items)} out of {len(news_items)} items.")
    return screened_items



def analyze_single_item(item, api_key, provider="Perplexity"):
    """
    Analyzes a single news item.
    Returns a list of result dictionaries (usually 0 or 1, but could be multiple if multiple entities found).
    """
    # Validate provider
    if provider not in LLM_PROVIDERS:
        logging.error(f"Invalid provider: {provider}. Defaulting to Perplexity.")
        provider = "Perplexity"
    
    # Get provider configuration
    config = LLM_PROVIDERS[provider]
    base_url = config["base_url"]
    model = config["model"]
    
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        logging.error(f"Failed to initialize {provider} client: {e}")
        return [], {"error": str(e)}

    results = []
    
    # Prepare content
    full_text = item.get('full_text', '')
    
    # Check if we need to use Perplexity Search Mode (Fallback)
    if not full_text or len(full_text) < 100:
        logging.warning(f"Content too short/empty for '{item['title']}'. Switching to Perplexity Search Mode.")
        content_to_use = f"Target News Title: {item['title']}\nSource: {item['source']}\n\n(Instructions: The original link could not be scraped. Please SEARCH for this news event online, read the details, and then extract the negative entities based on your search results.)"
        scraping_method = "Perplexity Search Fallback"
    else:
        content_to_use = full_text[:10000] # Truncate if using full text
        scraping_method = "Trafilatura Scraper"

    # Debug: Log content being analyzed
    logging.info(f"Analyzing content (len={len(content_to_use)}): {content_to_use[:100]}...")
    
    system_prompt = """
# Role: Expert Negative News Entity Extractor (Strict Full-Name Policy) #

# Objective: 你是一位精通中文語義分析的犯罪情報專家。你的任務是分析新聞文本，精準提取「負面新聞人物（犯罪嫌疑人、違法者、被調查對象）」的**完整真實姓名**

# Core Logic (Analysis Steps): 在提取之前，請先在腦中執行以下邏輯判斷：
1. **識別角色**：誰是執法者（主詞）？誰是違法者（受詞）？
2. **檢查完整性**：違法者的名字是全名（如：王小明）還是代稱（如：王男、王嫌）？
3. **過濾結果**：若僅有代稱，**絕對不要**提取，直接回傳空結果。

# Strict Constraints & Guidelines (必須嚴格遵守):

1. **Language Filter (語言過濾):** 
    - 僅處理「繁體中文」或「簡體中文」內容。 
    - 若新聞內容主要為非中文或無法閱讀（如 "Redirecting...", "JavaScript required"），請回傳 `status: "error"`

2. **Target Entity Definition (目標實體定義):** 
    - **提取對象**：涉嫌犯罪、違法、被捕、被起訴、被搜索、被約談、被判刑的人物。
    - **嚴格排除**：執法人員（警察、檢察官、法官）、受害者、無辜路人、律師、單純受訪者。

3. **Exclude Aliases & Partial Names (排除代稱 - 絕對紅線規則):**
    - **嚴格禁止提取代稱**：忽略所有「X姓XX」、「X某」、「X嫌」、「X員」、「X男/女」、「X歲XX」、「少婦XX」等不完整稱呼。
    - **僅提取全名**：目標必須具備完整的姓氏與名字（如：陳小華、王大明）。
    - **Null Rule**：若文中**僅出現**「陳姓主嫌」、「林男」而未出現全名，請回傳空列表 `[]`。

4. **Handle Name with Romanization (處理附帶英文名):**
    - 必須提取前後方跟隨英文譯名的中文姓名。
    - 範例：「馬來西亞金融家**劉特佐**（Low Taek Jho）」 -> 提取：`劉特佐`。

5. **Contextual Logic (語境邏輯):**
    - **警匪邏輯**：在「警員A逮捕了B」的句型中，B是目標，A必須排除。
    - **範例**：「警員**王大明**逮捕了**陳小華**」 -> 提取：`陳小華` (排除王大明)。

6. **Enforcement Targets & Passive Subjects (被動語態/執法對象):**
    - 「警員**王大明**逮捕了**陳小華**」 -> 提取：`陳小華` (排除王大明)。
    - **Crucial Rule**: 即使沒有描述具體犯罪動作，只要是**執法行動的對象**，就必須提取。
    - **範例**：
    - 「**張大文**被聲押」 -> 提取：`張大文`
    - 「**陳小華**被通緝」 -> 提取：`陳小華`
    - 「**林天明**轉為被告」 -> 提取：`林天明`
    - 「警方前往**王朝貴**住處搜索」 -> 提取：`王朝貴`

# Negative Examples (錯誤示範 - 請勿犯此類錯誤):
- ❌ 錯誤提取: "陳男" (原因: 代稱，非全名)
- ❌ 錯誤提取: "林姓主嫌" (原因: 代稱)
- ❌ 錯誤提取: "王警官" (原因: 執法人員)
- ❌ 錯誤提取: "李某" (原因: 不完整姓名)
- ❌ 錯誤提取: "陳先生" (原因: 過於泛稱，除非確認是負面人物全名)

# Output Format (JSON Only):
Please return ONLY a JSON object with the following structure:
{
  "status": "success" | "error",
  "thought_process": "Brief explanation of logic",
  "extracted_entities": [
    {
      "name": "Full Name",
      "original_text_pattern": "Text in article",
      "reason": "Reason for extraction"
    }
  ]
}
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"# Input Text:\n{content_to_use}"}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        
        # Debug: Log raw response
        logging.info(f"LLM Response for '{item.get('title', 'Unknown')[:50]}...': {content[:200]}...")
        
        # Clean up markdown code blocks
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```", "")
        
        # Strip whitespace
        content = content.strip()
        
        parsed_data = json.loads(content)
        
        # Debug: Log parsed status
        logging.info(f"Parsed status: {parsed_data.get('status')}, entities count: {len(parsed_data.get('extracted_entities', []))}")
        
        if parsed_data.get("status") == "success":
            entities = parsed_data.get("extracted_entities", [])
            if entities:
                names = [e['name'] for e in entities if e.get('name')]
                reasons = [f"{e['name']}: {e['reason']}" for e in entities if e.get('name') and e.get('reason')]
                summary = "; ".join(reasons)
                
                if not summary:
                    summary = item.get('title', 'No summary available')

                if names:
                    results.append({
                        "names": names,
                        "summary": summary,
                        "source_name": item['source'],
                        "source_url": item['link'],
                        # Pass through metadata for DataFrame
                        "keyword": item.get('keyword', ''),
                        "pub_date": item.get('pub_date', '')
                    })
        else:
            logging.warning(f"LLM returned non-success status for '{item.get('title', 'Unknown')[:50]}': {parsed_data.get('status')}")
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response from {provider}: {e}")
        logging.error(f"Raw content was: {content[:500] if 'content' in dir() else 'N/A'}")
    except Exception as e:
        logging.error(f"Error calling {provider} API: {e}")
        
    debug_info = {
        "input_content": content_to_use,
        "raw_response": content if 'content' in locals() else "No response or error occurred",
        "scraping_method": scraping_method
    }
    
    return results, debug_info
