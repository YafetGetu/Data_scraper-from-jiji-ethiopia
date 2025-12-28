import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import time

def scrape_jiji_listings():
    """
    Scrape product listings from Jiji website
    """
    # Website URL 
    url = 'https://jiji.com.et/'
    
    # Set headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Send HTTP request with headers
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Find all titles and prices
        titles = soup.find_all('div', class_='b-advert-title-inner qa-advert-title b-advert-title-inner--div')
        prices = soup.find_all('div', class_='qa-advert-price')
        
        scraped_data = []
        
        # Print and collect data
        print(f"\n Scraping Jiji Listings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        for title, price in zip(titles, prices):
            title_text = title.text.strip()
            price_text = price.text.strip()
            
            # Create data dictionary
            item_data = {
                'title': title_text,
                'price': price_text,
                'scraped_at': datetime.now().isoformat(),
                'source': 'Jiji Ethiopia'
            }
            
            scraped_data.append(item_data)
            print(f" {title_text} -  {price_text}")
        
        print(f"\n Successfully scraped {len(scraped_data)} items")
        
        # Save to files
        save_data(scraped_data)
        
        return scraped_data
        
    except requests.exceptions.RequestException as e:
        print(f" Error fetching the page: {e}")
        return []
    except Exception as e:
        print(f" An error occurred: {e}")
        return []

def save_data(data):
    """
    Save scraped data to multiple formats
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to JSON
    with open(f'jiji_data_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save to CSV
    with open(f'jiji_data_{timestamp}.csv', 'w', newline='', encoding='utf-8') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    # Append to daily log
    with open('daily_scraping_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} - Scraped {len(data)} items\n")
    
    print(f" Data saved to: jiji_data_{timestamp}.json and jiji_data_{timestamp}.csv")

def main():
    """
    Main execution function
    """
    print(" Starting Jiji Web Scraper")
    print("=" * 40)
    
    # Scrape data
    data = scrape_jiji_listings()
    
    # Summary
    if data:
        total_items = len(data)
        print(f"\n Summary: Scraped {total_items} items")
        print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate average price (if prices are numeric)
        try:
            prices = [float(item['price'].replace('Birr', '').replace(',', '').strip()) 
                     for item in data if 'Birr' in item['price']]
            if prices:
                avg_price = sum(prices) / len(prices)
                print(f"💰 Average Price: Birr {avg_price:,.2f}")
        except:
            pass
    
    print("\n Scraping completed successfully!")

if __name__ == "__main__":
    main()