from mistralai import Mistral
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import logging
import requests
import yaml
from logging.handlers import RotatingFileHandler

# Load environment variables from .env file
load_dotenv()

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configure logging for the application with enhanced features"""
    try:
        config = load_config()
        log_dir = Path(config['paths']['logs_dir'])
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / config['paths'].get('log_file', 'ai_category.log')
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(config['logging']['format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config['logging']['level']))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info('Logging system initialized successfully')
        
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")
        raise

def load_articles():
    """Load articles from remote URL"""
    config = load_config()
    url = config['articles']['source_url']
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except Exception as e:
        logging.error(f"Error loading articles from URL: {str(e)}")
        raise

def setup_directories():
    """Create necessary directories at startup"""
    config = load_config()
    # Create logs directory
    log_dir = Path(config['paths']['logs_dir'])
    log_dir.mkdir(exist_ok=True)
    
    # Create data directory
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(exist_ok=True)
    
    logging.info('Created necessary directories')

def determine_category(client, title, optimized_title, description, retry_count=0):
    """Determine the most suitable category using Mistral AI with retry logic"""
    config = load_config()
    logging.info(f'Determining category for: "{title}"')
    
    prompt = config['ai']['prompt_template'].format(
        title=title,
        optimized_title=optimized_title,
        description=description
    )

    try:
        response = client.chat.complete(
            model=config['api']['mistral_model'],
            messages=[{"role": "user", "content": prompt}]
        )
        category = response.choices[0].message.content.strip()
        logging.info(f'Category determined: "{category}"')
        return category
    except Exception as e:
        if "rate limit" in str(e).lower() and retry_count < config['optimization']['max_retries']:
            # Calculate delay with exponential backoff
            if config['optimization'].get('exponential_backoff', False):
                delay = min(
                    config['optimization']['retry_delay'] * (2 ** retry_count),
                    config['optimization'].get('max_backoff', 30)
                )
            else:
                delay = config['optimization']['retry_delay']
            
            logging.warning(f'Rate limit hit, retrying in {delay}s (attempt {retry_count + 1}/{config["optimization"]["max_retries"]})')
            time.sleep(delay)
            return determine_category(client, title, optimized_title, description, retry_count + 1)
        logging.error(f'Error determining category: {str(e)}')
        raise e

def save_categorized_articles(articles, append=False):
    """Save categorized articles to data/categorized_articles.json"""
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    output_path = data_dir / 'categorized_articles.json'
    
    if append and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            articles = existing_data.get('articles', []) + articles
    
    output_data = {
        "categorization_timestamp": datetime.now().isoformat(),
        "articles": articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f'Saved categorized articles to: {output_path}')

def load_existing_categories():
    """Load existing categorized articles to avoid re-categorization"""
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    output_path = data_dir / 'categorized_articles.json'
    
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {article['link']: article for article in data.get('articles', [])}
    return {}

def main():
    config = load_config()
    setup_directories()
    setup_logging()
    
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logging.error("MISTRAL_API_KEY not found in .env file")
            raise ValueError("MISTRAL_API_KEY not found in .env file")
        
        client = Mistral(api_key=api_key)
        articles = load_articles()
        
        print("\n=== Article Categorization Tool ===\n")
        
        # Load existing categorized articles
        existing_categories = load_existing_categories()
        
        for i, article in enumerate(articles[:config['articles']['limit']], 1):
            original_title = article.get('title', '')
            optimized_title = article.get('optimized_title', original_title)
            description = article.get('description', '')
            link = article.get('link', '')
            
            # Skip if article already categorized
            if link in existing_categories:
                logging.info(f'Skipping already categorized article: {original_title}')
                continue
                
            print(f"\nArticle {i}:")
            print(f"Title: {original_title}")
            
            try:
                if i > 1:
                    delay = config['optimization']['request_delay']
                    logging.info(f'Waiting {delay}s before next request')
                    time.sleep(delay)
                    
                category = determine_category(client, original_title, optimized_title, description)
                print(f"Category: {category}")
                print("-" * 50)
                
                categorized_article = {
                    "original_title": original_title,
                    "optimized_title": optimized_title,
                    "category": category,
                    "description": description,
                    "link": article.get('link', ''),
                    "published": article.get('published', ''),
                    "categorized_at": datetime.now().isoformat()
                }
                save_categorized_articles([categorized_article], append=True)
                
            except Exception as e:
                print(f"Error categorizing article: {str(e)}")
                logging.error(f'Error processing article {i}: {str(e)}')
                continue
        
        logging.info('Article categorization process completed')
        
    except Exception as e:
        logging.error(f'Fatal error in main process: {str(e)}')
        raise

if __name__ == "__main__":
    main()
