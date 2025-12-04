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

def analyze_news(news_items, api_key, provider="Perplexity"):
    """
    Phase 3: Deep Analysis.
    Analyzes a list of news items using specified LLM provider to extract criminal intelligence.
    
    Args:
        news_items (list): List of news item dictionaries.
        api_key (str): API Key for the selected provider.
        provider (str): LLM provider name.
        
    Returns:
        list: A list of dictionaries with enriched intelligence (names, summary).
    """
    if not news_items:
        return []
    
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
        logging.info(f"Using {provider} with model {model}")
    except Exception as e:
        logging.error(f"Failed to initialize {provider} client: {e}")
        raise
    
    results = []
    
    # Process items one by one to ensure strict adherence to the complex prompt
    for i, item in enumerate(news_items):
        # Prepare content
        content_to_use = item.get('full_text', '')
        if not content_to_use:
            content_to_use = item.get('summary', '')
            
        # Skip if content is too short
        if len(content_to_use) < 50:
            continue

        # Truncate content if necessary (Perplexity has high limits, but good practice)
        content_to_use = content_to_use[:10000] 
        
        system_prompt = (
            "# Role: Negative News Entity Extractor (負面新聞實體提取專家)\n\n"
            "# Objective:\n"
            "分析輸入的新聞文本，根據上下文邏輯，精準提取「負面新聞主角（犯罪嫌疑人、違法者）」的姓名或代稱。\n\n"
            "# Constraints & Guidelines:\n\n"
            "1.  **Language Filter (語言過濾):**\n"
            "    - 僅處理「繁體中文」或「簡體中文」內容。\n"
            "    - 若新聞內容主要為非中文（如英文、日文等），請直接回傳空結果，標註 \"non-chinese\"。\n\n"
            "2.  **Target Entity Definition (目標實體定義):**\n"
            "    - 僅提取新聞事件中「涉嫌犯罪」、「違法」、「被捕」、「被起訴」或「負面行為實施者」的人物。\n"
            "    - **嚴格排除**：執法人員（警察、檢察官）、受害者、無辜路人、律師、單純受訪者。\n\n"
            "3.  **Naming Convention Handling (特殊命名處理):**\n"
            "    - 必須識別並提取中文新聞特有的匿名或代稱格式，包括但不限於：\n"
            "      - 「X姓XX」（如：陳姓男子、林姓主嫌）\n"
            "      - 「X某」（如：張某）\n"
            "      - 「X男」、「X女」（如：李男、王女）\n"
            "      - 「X員」（如：陳員 - *需注意上下文，若是警員則排除，若是詐騙集團成員則提取*）\n"
            "      - 「X嫌」（如：劉嫌）\n"
            "      - 「X歲XX」（如：20歲林男）\n"
            "      - 「X姓」（如：黃姓）\n"
            "    - 若同一人在文中同時出現「全名」與「代稱」，優先提取「全名」。\n\n"
            "4.  **Contextual Logic (上下文邏輯 - 核心要求):**\n"
            "    - 必須閱讀整段文字來判斷角色關係。\n"
            "    - 範例邏輯：\n"
            "      - 句子：「警員**王大明**逮捕了涉嫌詐欺的**陳小華**。」 -> 提取：**陳小華** (排除王大明)。\n"
            "      - 句子：「**李男**因酒駕撞傷了路人**張女**。」 -> 提取：**李男** (排除張女)。\n\n"
            "# Output Format (JSON):\n"
            "請僅回傳符合以下 JSON Schema 的純 JSON 字串，不要包含 Markdown 代碼塊或其他解釋文字：\n\n"
            "{\n"
            "  \"status\": \"success\",  // 或 \"skipped_non_chinese\"\n"
            "  \"extracted_entities\": [\n"
            "    {\n"
            "      \"name\": \"提取到的名稱或代稱\",\n"
            "      \"original_text_pattern\": \"原文中出現的形式 (如：陳某)\",\n"
            "      \"type\": \"full_name\" | \"alias\", // 全名 或 代稱\n"
            "      \"reason\": \"簡短說明為何認定此人為負面人物 (例如：涉嫌詐欺、被警方逮捕)\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "# Input Text:\n"
            f"{content_to_use}"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": system_prompt} # Using user role for the whole prompt as it includes input
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            # logging.info(f"Received response from {provider} for item {i+1}")
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            
            parsed_data = json.loads(content)
            
            if parsed_data.get("status") == "success":
                entities = parsed_data.get("extracted_entities", [])
                if entities:
                    # Map to the format expected by app.py
                    names = [e['name'] for e in entities if e.get('name')]
                    
                    # Construct a summary from reasons
                    reasons = [f"{e['name']}: {e['reason']}" for e in entities if e.get('name') and e.get('reason')]
                    summary = "; ".join(reasons)
                    
                    if not summary:
                        summary = item.get('title', 'No summary available')

                    if names:
                        results.append({
                            "names": names,
                            "summary": summary,
                            "source_name": item['source'],
                            "source_url": item['link']
                        })
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from {provider}: {e}")
            logging.debug(f"Raw content: {content}")
        except Exception as e:
            logging.error(f"Error calling {provider} API for item {i+1}: {e}")
            # Continue to next item
            
    return results

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
    
    system_prompt = (
        "# Role: Negative News Entity Extractor (Full Name Only)\n\n"
        "# Objective:\n"
        "分析輸入的新聞文本（或根據標題自行搜尋相關報導），精準提取「負面新聞人物（犯罪嫌疑人、違法者、被調查對象）」的**完整真實姓名**。\n\n"
        "# Constraints & Guidelines:\n\n"
        "1.  **Language Filter (語言過濾):**\n"
        "    - 僅處理「繁體中文」或「簡體中文」內容。\n"
        "    - 若新聞內容主要為非中文或無法閱讀（如 \"Redirecting...\", \"JavaScript required\"），請回傳空結果並標註 status 為 error。\n\n"
        "2.  **Target Entity Definition (目標實體定義):**\n"
        "    - 提取對象：涉嫌犯罪、違法、被捕、被起訴、被搜索、被約談、被判刑的人物。\n"
        "    - **嚴格排除**：執法人員（警察、檢察官）、受害者、無辜路人、律師、單純受訪者。\n\n"
        "3.  **Exclude Aliases & Partial Names (排除代稱與不完整姓名 - 核心規則):**\n"
        "    - **嚴格禁止提取代稱**：請忽略所有「X姓男子」、「X某」、「X嫌」、「X男/女」等不完整稱呼。\n"
        "    - **僅提取全名**：目標必須具備完整的姓氏與名字（如：陳小華、王大明）。\n"
        "    - 若文中僅出現代稱（如「陳姓主嫌」）而未出現全名，請**不要提取**任何內容。\n\n"
        "4.  **Handle Name with Romanization (處理附帶英文名):**\n"
        "    - 必須提取後方跟隨英文譯名的中文姓名。\n"
        "      - 範例：「馬來西亞金融家**劉特佐**（Low Taek Jho）」 -> 提取：**劉特佐**。\n\n"
        "5.  **Contextual Logic & Action Verbs (行為動詞優先):**\n"
        "    - 判斷依據是**負面行為**，而非職稱。\n"
        "    - 無論職稱是「工程師」、「金融家」或「官員」，只要涉及：**挪用、詐騙、侵占、洗錢、收賄、內線交易、製毒、砍人**，即必須提取。\n"
        "    - **警匪邏輯**：「警員**王大明**逮捕了**陳小華**」 -> 提取：**陳小華**。\n\n"
        "6.  **Enforcement Targets & Passive Subjects (執法對象/被動語態):**\n"
        "    - **Crucial Rule**: 即使沒有描述正在犯罪，只要是**執法行動的對象**，就必須提取。\n"
        "    - **被動情境範例**：\n"
        "      - 「**張大文**被聲押」 -> 提取：**張大文**\n"
        "      - 「**張姓主嫌**被聲押」 -> **忽略** (代稱)\n"
        "      - 「警方前往**王男**住處搜索」 -> **忽略** (代稱)\n"
        "      - 「警方前往**王朝貴**住處搜索」 -> 提取：**王朝貴**\n"
        "      - 「**林xx**轉為被告」 -> **忽略** (不完整)\n"
        "      - 「**林天明**轉為被告」 -> 提取：**林天明**\n\n"
        "# Output Format (JSON Only):\n"
        "請僅回傳符合以下 JSON Schema 的純 JSON 字串，不要包含 Markdown 代碼塊：\n\n"
        "{\n"
        "  \"thought_process\": \"Step 1: Found News Sources - List the specific news article(s) found (Title/Source). Step 2: Entity Evaluation - Identify individuals mentioned and explicitly state why they are excluded (e.g., 'Wang is an official', 'No specific suspect named'). Step 3: Final Selection - Only include those who pass all filters.\",\n"
        "  \"status\": \"success\",\n"
        "  \"extracted_entities\": [\n"
        "    {\n"
        "      \"name\": \"提取到的完整姓名\",\n"
        "      \"original_text_pattern\": \"原文形式 (如: 被告謝聲德)\",\n"
        "      \"reason\": \"簡短說明 (例如: 涉嫌洗錢，被判刑 / 遭檢警約談)\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    
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
