import openai
import json
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def analyze_news(news_items, api_key):
    """
    Analyzes a list of news items using Perplexity API to extract criminal intelligence.
    
    Args:
        news_items (list): List of news item dictionaries.
        api_key (str): Perplexity API Key.
        
    Returns:
        list: A list of dictionaries with enriched intelligence (names, summary).
    """
    if not news_items:
        return []
        
    client = openai.OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    
    results = []
    
    # Process in batches
    batch_size = 5
    for i in range(0, len(news_items), batch_size):
        batch = news_items[i:i+batch_size]
        
        # Prepare context for the LLM
        context_text = ""
        for idx, item in enumerate(batch):
            context_text += f"News Item {idx+1}:\n"
            context_text += f"Title: {item['title']}\n"
            context_text += f"Source: {item['source']}\n"
            context_text += f"URL: {item['link']}\n"
            context_text += f"Date: {item['pub_date']}\n"
            context_text += f"Content: {item.get('summary', '')} {item.get('full_text', '')[:500]}\n\n"
            
        system_prompt = (
            "You are a professional criminal intelligence analyst. "
            "Your task is to analyze the provided news articles, group them by specific events, and extract entities.\n\n"
            "### Steps:\n"
            "1. **Filter**: Ignore articles that are:\n"
            "   - Not in Traditional Chinese (content is English or other languages).\n"
            "   - Not news (e.g., forum discussions, ads, short invalid text).\n"
            "   - If an article is ignored, do not include it in the output.\n\n"
            "2. **Event Integration**:\n"
            "   - Group articles that describe the **same specific criminal event** or case.\n"
            "   - Combine information from all articles in the group.\n\n"
            "3. **Entity Extraction (Strict)**:\n"
            "   - Extract names of people involved in the event (suspects, perpetrators, victims if named).\n"
            "   - **Criteria**:\n"
            "     - MUST be a full name (e.g., '陳大文', 'Zhang San').\n"
            "     - **STRICTLY EXCLUDE**: Single surnames ('陳先生'), fuzzy references ('某甲', '嫌犯'), titles ('檢察官').\n"
            "     - **STRICTLY EXCLUDE**: Journalistic placeholders like 'Surname + Gender' (e.g., '陳男', '林女', '張姓男子', '林姓公務員').\n"
            "     - If no valid names are found, return an empty list `[]`.\n\n"
            "4. **Summarization**:\n"
            "   - Write a fluent Traditional Chinese summary for the event group.\n"
            "   - Limit: **Max 100 words**.\n"
            "   - Focus on the key facts: Who, What, When, Where, Why.\n\n"
            "### Output Format:\n"
            "Return a JSON array where each object represents a unique event:\n"
            "[\n"
            "  {\n"
            "    \"names\": [\"Name1\", \"Name2\"],\n"
            "    \"summary\": \"Summary text...\",\n"
            "    \"source_name\": \"Source Name (or Multiple)\",\n"
            "    \"source_url\": \"URL of the most representative article\"\n"
            "  }\n"
            "]\n"
            "IMPORTANT: Only include items where at least one COMPLETE person name is identified.\n"
            "Output ONLY valid JSON."
        )
        
        try:
            response = client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_text}
                ],
            )
            
            content = response.choices[0].message.content
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
                
            parsed_data = json.loads(content)
            
            # If the API returns a single object instead of a list, wrap it
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
            
            # Filter and validate results
            for item in parsed_data:
                names = item.get('names', [])
                
                # Handle if names is a string (comma-separated) instead of array
                if isinstance(names, str):
                    names = [n.strip() for n in names.split(',')]
                
                # Filter for complete names only
                complete_names = [n for n in names if is_complete_name(n)]
                
                if complete_names:
                    item['names'] = complete_names  # Store as array
                    results.append(item)
            
        except Exception as e:
            logging.error(f"Error calling Perplexity API: {e}")
            
    return results
