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

def main():
    config = load_config()
    setup_directories()
    setup_logging()
    
    try:
        # Get API key from .env file
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logging.error("MISTRAL_API_KEY not found in .env file")
            raise ValueError("MISTRAL_API_KEY not found in .env file")
        
        # Initialize Mistral client with API key
        client = Mistral(api_key=api_key)
        
        # Load articles and existing optimizations
        articles = load_articles()
        existing_optimizations = load_existing_optimizations()
        
        print("\n=== Title Optimization Tool ===\n")
        
        # Process each article
        for i, article in enumerate(articles[:config['articles']['limit']], 1):
            original_title = article.get('title', '')
            description = article.get('description', '')
            
            print(f"\nArticle {i}:")
            print(f"Original: {original_title}")
            
            # Check if we already have an optimization for this title
            if original_title in existing_optimizations:
                existing_article = existing_optimizations[original_title]
                print(f"Optimized (existing): {existing_article['optimized_title']}")
                print("-" * 50)
                continue
                
            try:
                # Increase initial delay between requests
                if i > 1:
                    delay = config['optimization']['request_delay']
                    logging.info(f'Waiting {delay}s before next request')
                    time.sleep(delay)
                    
                optimized_title = optimize_title(client, original_title, description)
                print(f"Optimized (new): {optimized_title}")
                print("-" * 50)
                
                # Create and save single optimized article immediately
                optimized_article = {
                    "original_title": original_title,
                    "optimized_title": optimized_title,
                    "description": description,
                    "link": article.get('link', ''),
                    "published": article.get('published', ''),
                    "optimized_at": datetime.now().isoformat()
                }
                save_optimized_titles([optimized_article], append=True)
                
            except Exception as e:
                print(f"Error optimizing title: {str(e)}")
                logging.error(f'Error processing article {i}: {str(e)}')
                continue
        
        logging.info('Title optimization process completed')
        
    except Exception as e:
        logging.error(f'Fatal error in main process: {str(e)}')
        raise

if __name__ == "__main__":
    main()
