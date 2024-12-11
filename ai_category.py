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

# Load environment variables from .env file
load_dotenv()

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configure logging for the application"""
    config = load_config()
    log_dir = Path(config['paths']['logs_dir'])
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / config['paths'].get('log_file', 'title_optimizer.log')
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info('Starting title optimization process')

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

def load_existing_optimizations():
    """Load existing optimized titles if they exist"""
    config = load_config()
    output_path = Path(config['paths']['data_dir']) / config['paths']['optimized_titles_file']
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Create a dictionary of original_title -> optimized_article for easy lookup
            return {
                article['original_title']: article 
                for article in data.get('articles', [])
            }
    return {}

def save_optimized_titles(articles, append=False):
    """Save optimized titles to data/optimized_titles.json"""
    config = load_config()
    data_dir = Path(config['paths']['data_dir'])
    output_path = data_dir / config['paths']['optimized_titles_file']
    
    # If appending and file exists, load existing data
    if append and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            articles = existing_data.get('articles', []) + articles
    
    output_data = {
        "optimization_timestamp": datetime.now().isoformat(),
        "articles": articles
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f'Saved optimized titles to: {output_path}')

def optimize_title(client, title, description, retry_count=0):
    """Generate an optimized title using Mistral AI with retry logic"""
    config = load_config()
    logging.info(f'Optimizing title: "{title}"')
    
    prompt = config['ai']['prompt_template'].format(
        title=title,
        description=description
    )

    try:
        response = client.chat.complete(
            model=config['api']['mistral_model'],
            messages=[{"role": "user", "content": prompt}]
        )
        optimized = response.choices[0].message.content.strip('"')
        logging.info(f'Successfully optimized title to: "{optimized}"')
        return optimized
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
            return optimize_title(client, title, description, retry_count + 1)
        logging.error(f'Error optimizing title: {str(e)}')
        raise e

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
        
        for i, article in enumerate(articles[:config['articles']['limit']], 1):
            original_title = article.get('title', '')
            optimized_title = article.get('optimized_title', original_title)
            description = article.get('description', '')
            
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
