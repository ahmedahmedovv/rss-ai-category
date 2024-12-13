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
    categories = config['ai']['category_criteria']['categories']
    categories_str = ", ".join(categories)
    
    # Initialize Mistral client
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    client = Mistral(api_key=api_key)

    # Fetch data from GitHub URL instead of local file
    github_url = "https://raw.githubusercontent.com/ahmedahmedovv/rss-ai-title/refs/heads/main/data/optimized_titles.json"
    print(f"Debug: Attempting to fetch data from GitHub URL: {github_url}")
    response = requests.get(github_url)
    print(f"Debug: Response status code: {response.status_code}")
    
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response content: {response.text[:500]}...")  # Print first 500 chars
        return
    
    # Extract the articles array from the data
    data = data.get('articles', [])
    
    # Add more debug information
    print(f"Debug: Number of articles to process: {len(data)}")
    print(f"Debug: API Key present: {'Yes' if api_key else 'No'}")
    print(f"Debug: First few characters of API key: {api_key[:4]}..." if api_key else "No API key")

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

        # Add timeout handling and improved rate limiting
        BATCH_SIZE = 10  # Process in smaller batches
        BATCH_TIMEOUT = 240  # 4 minutes per batch
        start_time = time.time()

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            batch_start = time.time()
            
            # Check overall execution time
            if time.time() - start_time > BATCH_TIMEOUT:
                print("⚠️ Approaching timeout limit, saving progress and exiting")
                save_progress()
                return

            for entry in batch:
                # Skip if already categorized
                if entry['original_title'] in existing_categorized:
                    print(f"⏩ Skipping already categorized: {entry['original_title']}")
                    continue

                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Combine text for analysis
                        combined_text = f"""
                        Original Title: {entry['original_title']}
                        Optimized Title: {entry['optimized_title']}
                        Description: {entry.get('description', 'No description available')}
                        """

                        # Modify the API call to get both category and summary
                        response = client.chat.complete(
                            model="mistral-small-latest",
                            messages=[
                                {
                                    "role": "system",
                                    "content": f"""You have two tasks:
                                    1. Categorize the content into ONE of these categories: {categories_str}
                                    2. Create a brief 2-3 sentence summary of the content.
                                    
                                    Format your response exactly like this:
                                    CATEGORY: [category name]
                                    SUMMARY: [2-3 sentence summary]"""
                                },
                                {
                                    "role": "user",
                                    "content": combined_text
                                }
                            ]
                        )
                        
                        # Parse the response to extract category and summary
                        response_text = response.choices[0].message.content.strip()
                        category = ""
                        summary = ""

                        # Extract category and summary from response
                        for line in response_text.split('\n'):
                            if line.startswith('CATEGORY:'):
                                category = line.replace('CATEGORY:', '').strip()
                            elif line.startswith('SUMMARY:'):
                                summary = line.replace('SUMMARY:', '').strip()

                        # Add both category and summary to entry
                        categorized_entry = entry.copy()
                        categorized_entry['category'] = category
                        categorized_entry['summary'] = summary
                        categorized_data.append(categorized_entry)
                        
                        print(f"✅ Processed: {entry['original_title']}")
                        print(f"Category: {category}")
                        print(f"Summary: {summary[:100]}...")  # Print first 100 chars of summary
                        
                        # Add exponential backoff for rate limiting
                        retry_delay = 5 * (2 ** retry_count)  # Exponential backoff
                        time.sleep(1)  # Basic rate limiting between successful requests
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

            # Save progress after each batch
            save_progress()
            print(f"✅ Completed batch {i//BATCH_SIZE + 1}, time taken: {time.time() - batch_start:.2f}s")

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user.")
        save_progress()
        sys.exit(0)

if __name__ == "__main__":
    print("🔍 Analyzing and categorizing content...")
    analyze_and_categorize_data()