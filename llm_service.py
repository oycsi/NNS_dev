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
            # Use full_text if available, otherwise summary
            content_to_use = item.get('full_text', '')
            if not content_to_use:
                content_to_use = item.get('summary', '')
            
            # Truncate content to avoid token limits (though Perplexity has high limits)
            context_text += f"Content: {content_to_use[:2000]}\n\n"
            
        system_prompt = (
            "You are an Adverse Media Analyst. Analyze the provided text. "
            "Strictly verify two conditions:\n"
            "1. Is the keyword/subject involved in a negative event?\n"
            "2. Is the event genuinely negative/risky (scandal, crime, regulation, fraud, corruption)?\n\n"
            "If NO to either, output 'IRRELEVANT' for that item (do not include it in the output list).\n"
            "If YES, provide a concise summary and extract entities.\n\n"
            "### Output Format:\n"
            "Return a JSON array where each object represents a unique event:\n"
            "[\n"
            "  {\n"
            "    \"names\": [\"Name1\", \"Name2\"],\n"
            "    \"summary\": \"Concise summary of the adverse event...\",\n"
            "    \"source_name\": \"Source Name\",\n"
            "    \"source_url\": \"URL\"\n"
            "  }\n"
            "]\n"
            "IMPORTANT:\n"
            "- Only include items where at least one COMPLETE person name is identified.\n"
            "- STRICTLY EXCLUDE: Journalistic placeholders like 'Surname + Gender' (e.g., '陳男', '林女').\n"
            "- Output ONLY valid JSON."
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_text}
                ],
            )
            
            content = response.choices[0].message.content
            logging.info(f"Received response from {provider}")
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            
            # Log the raw content for debugging
            logging.debug(f"API Response: {content[:500]}")
            
            if "IRRELEVANT" in content and len(content) < 20:
                continue

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
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from {provider}: {e}")
            logging.error(f"Raw content: {content}")
        except Exception as e:
            logging.error(f"Error calling {provider} API: {e}")
            logging.error(f"Full error details: {str(e)}")
            # Don't raise, just log and continue to next batch
            
    return results
