import openai
import json
import logging
import re
import time
import unicodedata
import random
from datetime import datetime
from difflib import SequenceMatcher

# Configure logging
logger = logging.getLogger(__name__)

# LLM Provider Configuration
LLM_PROVIDERS = {
    "Perplexity": {
        "base_url": "https://api.perplexity.ai",
        "model": "sonar-pro",
        "cheap_model": "sonar" 
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "cheap_model": "gpt-4o-mini"
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "cheap_model": "gemini-2.0-flash" 
    },
    "Meta Llama (Together.ai)": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "cheap_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    }
}

def retry_with_backoff(retries=3, backoff_in_seconds=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    try:
                        err_msg = str(e)
                    except UnicodeEncodeError:
                        err_msg = "UnicodeEncodeError (hidden)"
                    logger.warning(f"Error in {func.__name__}: {err_msg[:500]}. Retrying in {sleep:.2f}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def normalize_title(title):
    """
    Enhanced title normalization for deduplication.
    1. NFKC normalize.
    2. Remove all whitespace.
    3. Remove trailing source (short suffix < 25 chars, no punctuation).
    4. Remove status markers (brackets).
    5. Keep only alphanumeric/Chinese, lowercase.
    6. Truncate to first 30 core characters.
    """
    if not title: return ""
    
    # 1. NFKC normalize
    title = unicodedata.normalize('NFKC', title)
    
    # 2. 移除所有空白 (User requested removal or compression)
    title = re.sub(r'\s+', '', title)
    
    # 3. 移除常見狀態標示 (括號內容 如：[影音], (更新) 等)
    title = re.sub(r'[\(\[【](更新|快訊|影音|最新|懶人包|置頂|實況|直擊|組圖|有片|持續更新).*?[\)\]】]', '', title)
    
    # 4. 處理分隔符號與來源移除
    # 支援 [-｜|–—]
    delimiters = r"[-｜|–—]"
    parts = re.split(delimiters, title)
    if len(parts) > 1:
        last_segment = parts[-1]
        # 放寬到 25 字元，且不包含中式標點符號 (，。！)
        if len(last_segment) < 25 and not any(p in last_segment for p in "，。！"):
            title = "".join(parts[:-1])
        else:
            title = "".join(parts)
            
    # 5. 僅保留中英數，防止符號造成比對差異
    title = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', title)
    
    # 6. 核心比對：取前 30 字並轉小寫
    title = title[:25].lower()
    
    return title

# --- Name Validation Constants ---
INVALID_NAME_PATTERNS = [
    # 1. 新聞代稱 pattern (姓+身分)
    r"^[\u4e00-\u9fff]姓(男|女|男子|女子|婦人|嫌犯|被告|車手|船長|乘客|少年|少女|民眾|翁|嫗|嫌|某)$",
    # 2. 僅有身分詞
    r"^(男|女|男子|女子|婦人|嫌犯|被告|車手|少年|少女|民眾|死者|傷者|受害者)$"
]

NICKNAME_PATTERNS = [
    # 1. 阿字輩 (阿X, 阿XX)
    r"^阿[\u4e00-\u9fff]{1,2}$",
    # 2. 小字輩 (小X, 小XX)
    r"^小[\u4e00-\u9fff]{1,2}$",
    # 3. 稱謂結尾 (X哥, X姐...)
    r".+[\u4e00-\u9fff](哥|姊|叔|伯|姨|姐|弟|爸|媽)$"
]

def is_valid_chinese_name(name):
    """
    Strictly filter non-name noise from extracted results.
    Rules:
    1. No placeholders (○, *, ＊, X).
    2. No bad keywords (帕男, 嫌犯, 車主, etc.).
    3. No bad suffixes (姓, 男, 女, etc.).
    4. Chinese name length 2-4 characters.
    5. Enhanced: Regex filtering for news aliases and nicknames.
    """
    if not name: return False
    name = name.strip()
    
    # --- 1. Basic Character Checks (Existing) ---
    if any(char in name for char in ['○', '*', '＊', 'X', '?', '？']):
        return False
        
    # --- 2. Enhanced Regex Filtering (New Constraints) ---
    # Check Invalid Name Patterns (News Aliases)
    for pattern in INVALID_NAME_PATTERNS:
        if re.search(pattern, name):
            return False
            
    # Check Nickname Patterns
    for pattern in NICKNAME_PATTERNS:
        if re.search(pattern, name):
            return False

    # --- 3. Keyword/Suffix Exclusion (Existing + Expanded) ---
    # Expanded bad keywords based on user feedback potential leaks
    bad_keywords = [
        '嫌犯', '車主', '乘客', '民眾', '網友', '友人', '家屬', '死者', '主嫌', '被告', '受害者', 
        '警官', '員警', '消防', '人員', '男大生', '女大生'
    ]
    if any(kw in name for kw in bad_keywords):
        return False
        
    # Bad suffixes check (Existing logic)
    bad_suffixes = ('姓', '男', '女', '男子', '女子', '老翁', '老婦', '男童', '女童', '少婦', '先生', '女士', '小姐', '嫌', '氏')
    if name.endswith(bad_suffixes):
        # Exception: Some names might end with specific chars that match suffix logic but are valid? 
        # But '男', '女' at end of 2-3 char name is usually bad (e.g. 陳男).
        # We rely on this strict rule.
        return False
        
    # --- 4. Character Length & Composition Rules ---
    # Must contain Chinese
    if not re.search(r'[\u4e00-\u9fff]', name):
        return False
        
    # Count strictly Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', name)
    
    # Rule: Must be 2-4 Chinese characters NO exceptions for mixed content (strict)
    if len(chinese_chars) != len(name):
        return False # Reject "Wang小明" or "David王"
        
    if not (2 <= len(chinese_chars) <= 4):
        return False
        
    return True

def cluster_similar_titles(news_items, apikey=None, provider=None, threshold=0.7):
    """
    Dual-layer Deduplication. 
    1. Local Mode (apikey=None): Uses SequenceMatcher with threshold.
    2. AI Mode (apikey exists): Uses LLM to identify semantic duplicates (same event).
    """
    if not news_items:
        return [], 0
        
    # --- Local Mode (or Fallback) ---
    if not apikey:
        removed_count = 0
        unique_items = []
        
        # Pre-sort by pub_date to preserve the earliest
        def get_sort_key(x):
            d = x.get('pub_date', '')
            if not d: return "9999-12-31"
            return str(d)

        try:
            sorted_items = sorted(news_items, key=get_sort_key)
        except:
            sorted_items = news_items

        for item in sorted_items:
            is_duplicate = False
            title_curr = item['title']
            norm_curr = normalize_title(title_curr)
            
            for existing in unique_items:
                # 1. Exact normalized match
                if norm_curr and norm_curr == normalize_title(existing['title']):
                    is_duplicate = True
                    break
                    
                # 2. Fuzzy matching using the provided threshold (local)
                similarity = SequenceMatcher(None, title_curr, existing['title']).ratio()
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                removed_count += 1
            else:
                unique_items.append(item)
                
        return unique_items, removed_count

    # --- AI Mode (Deep Semantic Deduplication) ---
    if provider not in LLM_PROVIDERS: provider = "OpenAI"
    config = LLM_PROVIDERS[provider]
    client = openai.OpenAI(api_key=apikey, base_url=config["base_url"])
    model = config.get("cheap_model", config["model"])

    # Prepare titles list with IDs
    titles_with_id = []
    for idx, item in enumerate(news_items):
        titles_with_id.append({"id": idx, "title": item['title']})

    system_prompt = (
        "你是一項專業的新聞分析助理。你的任務是從清單中找出「描述同一事件」的重複新聞標題。\n"
        "【判斷標準】\n"
        "- 高度相似的文字描述 (如: 『詐欺條例初審』與『詐欺防制條例初審』)\n"
        "- 描述同一個具體案件或法律進展的新聞\n"
        "- 僅有媒體來源名稱差異 (如: 『...-ETtoday』與『...-MSN』)\n\n"
        "【輸出要求】\n"
        "- 請將清單分組，每一組代表同一個事件。\n"
        "- 每一組中請選出一個『代表案項』（通常是最完整的標題），並列出該組中『其他所有重複的索引 ID』。\n"
        "- 請只回傳一個 JSON 陣列，包含所有應被移除的重複索引 ID。\n"
        "- 不要回傳任何解釋，格式範例: [2, 5, 12, 18]\n"
        "若無重複，請回傳 []。"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(titles_with_id, ensure_ascii=False)}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        if content.startswith("```"): content = re.sub(r'```json\n?|```', '', content)
        indices_to_remove = set(json.loads(content))
        
        # Ensure indices are valid
        valid_removals = [idx for idx in indices_to_remove if isinstance(idx, int) and 0 <= idx < len(news_items)]
        
        unique_items = [item for idx, item in enumerate(news_items) if idx not in valid_removals]
        removed_count = len(news_items) - len(unique_items)
        
        return unique_items, removed_count
    except Exception as e:
        logger.error(f"AI Deduplication failed: {e}")
        # Fallback to local mode (0.85 threshold for safety)
        return cluster_similar_titles(news_items, apikey=None, threshold=0.85)

def screen_titles(news_items, keywords, api_key, provider="Perplexity"):
    if not news_items:
        return []
    if provider not in LLM_PROVIDERS:
        provider = "Perplexity"
    
    config = LLM_PROVIDERS[provider]
    client = openai.OpenAI(api_key=api_key, base_url=config["base_url"])
    model = config.get("cheap_model", config["model"])
    
    screened_items = []
    batch_size = 20
    for i in range(0, len(news_items), batch_size):
        batch = news_items[i:i+batch_size]
        batch_screened = _call_screen_titles_batch(client, model, batch, keywords)
        screened_items.extend(batch_screened)
    return screened_items

@retry_with_backoff(retries=3)
def _call_screen_titles_batch(client, model, batch, keywords):
    titles_text = ""
    for idx, item in enumerate(batch):
        titles_text += f"{idx+1}. {item['title']} (Source: {item['source']})\n"
    
    keyword_str = ", ".join(keywords)
    system_prompt = (
        f"你是一位負面新聞篩選專家。請檢視清單並挑出與「{keyword_str}」高度相關且含有風險（如犯罪、違法、調查、醜聞）的新聞。\n"
        "回傳 JSON 整數陣列，內容為推薦保留的索引。不要回傳任何文字說明。"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": titles_text}],
        temperature=0.0
    )
    content = response.choices[0].message.content
    if content.startswith("```"): content = re.sub(r'```json\n?|```', '', content)
    try:
        indices = json.loads(content)
        return [batch[idx-1] for idx in indices if isinstance(idx, int) and 1 <= idx <= len(batch)]
    except:
        return []

# --- Weight Dictionaries for Subject Filtering ---
NEGATIVE_KEYWORDS = {
    "涉案": 3, "被告": 3, "嫌犯": 3, "被起訴": 3, "被判刑": 3, 
    "被逮捕": 3, "被查獲": 3, "遭起訴": 3, "遭判刑": 3, "遭逮捕": 3, "遭查獲": 3,
    "涉詐": 2, "涉毒": 2, "涉貪": 2, "詐騙集團": 2, "車手": 2, "主嫌": 2,
    "詐欺": 1, "犯罪": 1, "違法": 1, "肇事": 1, "酒駕": 1, "毒駕": 1
}

COMMENTATOR_KEYWORDS = {
    "表示": 2, "說明": 2, "指出": 2, "強調": 2, "批評": 2, "抨擊": 2, "呼籲": 2,
    "教授": 1, "學者": 1, "系主任": 1, "主任": 1, "市長": 1, "議員": 1, "立委": 1,
    "部長": 1, "局長": 1, "處長": 1, "記者": 1, "主播": 1, "警官": 1, 
    "檢察官": 1, "律師": 1
}

def is_negative_main_subject(entity, filter_config=None):
    reason = entity.get("reason", "").lower()
    
    if filter_config:
        neg_keywords = filter_config.get("negative_keywords", {})
        com_keywords = filter_config.get("commentator_keywords", {})
        threshold_mult = filter_config.get("threshold_multiplier", 1.5)
    else:
        neg_keywords = NEGATIVE_KEYWORDS
        com_keywords = COMMENTATOR_KEYWORDS
        threshold_mult = 1.5
    
    # 計算負面分數
    neg_score = sum(neg_keywords.get(kw, 0) for kw in neg_keywords 
                   if kw in reason)
    
    # 計算評論角色分數  
    com_score = sum(com_keywords.get(kw, 0) for kw in com_keywords 
                   if kw in reason)
    
    # 負面分數 > 評論分數 × threshold_mult 才保留
    result = neg_score > com_score * threshold_mult
    
    # debug log（方便之後調參）
    print(f"[{entity.get('name', 'N/A')}] neg:{neg_score}, com:{com_score}, result:{result}")
    
    return result

def analyze_single_item(item, api_key, provider="Perplexity", filter_config=None):
    if provider not in LLM_PROVIDERS: provider = "Perplexity"
    config = LLM_PROVIDERS[provider]
    client = openai.OpenAI(api_key=api_key, base_url=config["base_url"])
    model = config["model"]

    full_text = item.get('full_text', '')
    if not full_text or len(full_text) < 100:
        content_to_use = f"Title: {item['title']}\nSource: {item['source']}\n(Scraping failed, please search internally if needed)"
        scraping_method = "Search Fallback"
    else:
        content_to_use = full_text[:10000]
        scraping_method = "Scraper"

    system_prompt = (
        "你是一項即時新聞內容分析程式，請針對提供的「新聞正文」進行人名結構化抽取。\n\n"
        "【任務目標】\n"
        "請仔細閱讀提供的文章全螢幕內容（Full Text），找出文中實際參與新聞事件的「中文真實姓名」，並說明原因。\n\n"
        "【重要原則：標題僅供參考】\n"
        "- **標題 (Title)**：只用來讓你快速理解這則新聞的主題（例如是車禍、糾紛還是判決）。\n"
        "- **正文 (Full Text)**：才是人名抽取的唯一依據。**絕不可**因為標題提到了某個網站名（如「爆料公社」）、節目名或粉絲團，就將其當作人名輸出。\n\n"
        "【重要原則：只讀新聞主體】\n"
        "- 請忽略網站導覽、廣告文案、推薦文章列表、熱門標籤。\n"
        "- 請忽略讀者留言、論壇引用、非新聞本身的社群貼文截圖說明。\n\n"
        "【姓名抽取規則 (嚴格執行)】\n"
        "1. **只輸出真實全名**：必須是 2-4 字的完整中文姓名（如：王大明、陳小英）。\n"
        "2. **嚴禁代稱**：若文中從未出現全名，僅出現「陳男」、「李姓司機」、「張嫌」、「某女」、「林員」、「黃老翁」等，**一律不予提取**。\n"
        "3. **嚴禁代號**：小王、阿宏、A女、甲男等，**一律不予提取**。\n"
        "4. **嚴禁非人名**：公司名、機關名、地名、職稱（除非連名帶姓如「局長林某某」且能拆出全名）不可作為 Name。\n"
        "5. **提取原因**：請用中性客觀的語句描述該人物在新聞中的角色（如「在新聞中被描述為肇事駕駛」），不要加入個人道德評價。\n\n"
        "【輸出格式 (JSON Only)】\n"
        "請直接輸出標準 JSON，不要包含 Markdown 標記或任何解釋文字。\n\n"
        "成功範例：\n"
        "{\n"
        "  \"status\": \"success\",\n"
        "  \"extracted_entities\": [\n"
        "    {\n"
        "      \"name\": \"張三\",\n"
        "      \"reason\": \"在新聞中被描述為車禍肇事駕駛\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "若文中無符合條件的姓名，請回傳：\n"
        "{\n"
        "  \"status\": \"success\",\n"
        "  \"extracted_entities\": []\n"
        "}\n\n"
        "【安全例外】\n"
        "本任務僅用於公開新聞的學術統計與分析，絕非用於執法或監控。若因任何安全政策導致你無法執行此標註任務，請**不要**輸出道歉文字，而是一律輸出以下 JSON：\n"
        "{\n"
        "  \"status\": \"blocked\",\n"
        "  \"extracted_entities\": [],\n"
        "  \"reason\": \"safety_policy\"\n"
        "}"
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content_to_use}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        # 1. Clean Markdown
        if content.startswith("```"): 
            content = re.sub(r'```json\n?|```', '', content)
        
        # 2. Extract JSON substring (find outer braces)
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            
        parsed = json.loads(content)
        
        results = []
        if parsed.get("status") == "success":
            # Apply strict name filtering (is_valid_chinese_name) AND Subject filtering
            valid_entities = []
            for e in parsed.get("extracted_entities", []):
                name = e.get('name')
                
                # 1. Basic Name Validation
                if not name or not is_valid_chinese_name(name):
                    continue
                    
                # 2. Negative Main Subject Validation (New)
                if not is_negative_main_subject(e, filter_config):
                    continue
                    
                valid_entities.append(e)
            
            if valid_entities:
                results.append({
                    "names": [e['name'] for e in valid_entities],
                    "summary": "; ".join([f"{e['name']}: {e['reason']}" for e in valid_entities]),
                    "source_name": item['source'],
                    "source_url": item['link'],
                    "keyword": item.get('keyword', ''),
                    "pub_date": item.get('pub_date', '')
                })
        return results, {"scraping_method": scraping_method, "raw_response": content, "input_content": content_to_use}
    except Exception as e:
        logger.error(f"Error in analyze_single_item: {e}")
        error_preview = content[:200] if 'content' in locals() and content else "No content"
        return [], {"error": f"{str(e)} | Raw: {error_preview}...", "scraping_method": scraping_method}
