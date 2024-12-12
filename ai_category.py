import os
import requests
from mistralai import Mistral
import json
import time
import sys
import yaml

def analyze_and_categorize_data():
    # Load config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract categories from config
    categories = config['ai']['category_criteria']['primary_categories']
    categories_str = ", ".join(categories)
    
    # Initialize Mistral client
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    client = Mistral(api_key=api_key)

    # Fetch data from GitHub URL instead of local file
    github_url = "https://raw.githubusercontent.com/ahmedahmedovv/rss-ai-title/refs/heads/main/data/optimized_titles.json"
    response = requests.get(github_url)
    data = response.json()
    
    # Extract the articles array from the data
    data = data.get('articles', [])

    # Verify data is a list
    if not isinstance(data, list):
        print(f"Error: Expected a list but got {type(data)}")
        return

    print("Data type:", type(data))
    print("First entry:", data[:100] if isinstance(data, str) else data[0])

    # Load existing categorized data if available
    output_path = os.path.join('data', 'categorized_articles.json')
    existing_categorized = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            # Create lookup dictionary using original_title as key
            existing_categorized = {item['original_title']: item for item in existing_data}

    categorized_data = list(existing_categorized.values())  # Convert existing data to list

    def save_progress():
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(categorized_data, f, indent=4, ensure_ascii=False)
        print(f"💾 Progress saved to {output_path}")

    try:
        # Add rate limiting handling
        retry_count = 0
        max_retries = 3
        retry_delay = 5  # seconds

        for entry in data:
            # Skip if already categorized
            if entry['original_title'] in existing_categorized:
                print(f"⏩ Skipping already categorized: {entry['original_title']}")
                continue

            while retry_count < max_retries:
                try:
                    # Combine text for analysis
                    combined_text = f"""
                    Original Title: {entry['original_title']}
                    Optimized Title: {entry['optimized_title']}
                    Description: {entry.get('description', 'No description available')}
                    """

                    # Get category from Mistral AI
                    response = client.chat.complete(
                        model="mistral-small-latest",
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a content categorizer. Analyze the given content and assign ONE category from the following list: {categories_str}. Return ONLY the category name, nothing else."
                            },
                            {
                                "role": "user",
                                "content": combined_text
                            }
                        ]
                    )
                    
                    category = response.choices[0].message.content.strip()

                    # Add category to entry
                    categorized_entry = entry.copy()
                    categorized_entry['category'] = category
                    categorized_data.append(categorized_entry)
                    
                    print(f"✅ Categorized: {entry['original_title']} -> {category}")
                    
                    # Save progress after each successful categorization
                    save_progress()
                    
                    retry_count = 0
                    break
                except Exception as e:
                    if "429" in str(e):
                        retry_count += 1
                        print(f"Rate limit hit, waiting {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"❌ Error analyzing entry: {str(e)}")
                        break
            
            if retry_count >= max_retries:
                print("Maximum retries reached, skipping entry")
                retry_count = 0
                continue
                
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user.")
        save_progress()
        sys.exit(0)

if __name__ == "__main__":
    print("🔍 Analyzing and categorizing content...")
    analyze_and_categorize_data()