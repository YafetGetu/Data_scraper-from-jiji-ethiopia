import requests
from bs4 import BeautifulSoup
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import re
from collections import defaultdict
import statistics
from sklearn.linear_model import LinearRegression
import warnings
from urllib.parse import quote_plus, urljoin
import random

warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Scraper settings
REQUEST_DELAY = 2
MAX_RETRIES = 3
TIMEOUT = 10
MAX_PAGES = 5  # Maximum pages to scrape per search
ITEMS_PER_PAGE = 40  # Jiji shows ~40 items per page

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# PRICE PARSING
def parse_price(price_text):
    """Convert price text to numeric value"""
    if not price_text:
        return None
    
    try:
        # Remove currency symbols, commas, and spaces
        cleaned = re.sub(r'[^\d.]', '', price_text)
        if cleaned:
            return float(cleaned)
    except:
        pass
    return None

def extract_currency(price_text):
    """Extract currency from price text"""
    if not price_text:
        return 'ETB'
    
    price_text = price_text.upper()
    if 'USD' in price_text or '$' in price_text:
        return 'USD'
    elif 'ETB' in price_text or 'BIRR' in price_text:
        return 'ETB'
    else:
        # Default to ETB for Jiji Ethiopia
        return 'ETB'

def parse_date(date_str):
    """Parse date string to datetime"""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return datetime.now()

# CATEGORY CLASSIFICATION
def categorize_item(title, search_query=None):
    """Categorize items based on keywords"""
    title_lower = title.lower()
    
    # Enhanced categories for Jiji Ethiopia
    categories = {
        'Mobile Phones': ['iphone', 'samsung', 'galaxy', 'huawei', 'xiaomi', 'redmi', 'nokia', 
                         'tecno', 'infinix', 'oppo', 'vivo', 'oneplus', 'smartphone', 'mobile', 'phone'],
        'Electronics': ['laptop', 'tablet', 'computer', 'macbook', 'dell', 'hp', 'lenovo',
                       'tv', 'television', 'camera', 'headphone', 'earphone', 'speaker',
                       'playstation', 'xbox', 'nintendo', 'console'],
        'Cars': ['car', 'toyota', 'bmw', 'mercedes', 'honda', 'hyundai', 'volkswagen',
                'ford', 'nissan', 'mazda', 'suzuki', 'kia', 'peugeot', 'yaris', 'corolla',
                'prius', 'camry', 'rav4', 'land cruiser', 'prado'],
        'Motorcycles': ['motorcycle', 'bike', 'yamaha', 'kawasaki', 'scooter', 'motorbike'],
        'Real Estate': ['house', 'apartment', 'condominium', 'villa', 'land', 'property',
                       'rent', 'for rent', 'sale', 'bedroom', 'bathroom'],
        'Fashion': ['shoe', 'clothes', 'dress', 'shirt', 'jacket', 'watch', 'bag',
                   'perfume', 'jewelry', 'ring', 'necklace'],
        'Home Appliances': ['fridge', 'refrigerator', 'oven', 'washing machine', 'microwave',
                           'blender', 'mixer', 'iron', 'vacuum', 'ac', 'air conditioner'],
        'Furniture': ['bed', 'sofa', 'table', 'chair', 'wardrobe', 'cabinet', 'shelf'],
        'Jobs': ['job', 'vacancy', 'employment', 'hire', 'recruit', 'position', 'wanted'],
        'Services': ['service', 'repair', 'maintenance', 'delivery', 'training', 'course'],
        'Animals': ['dog', 'cat', 'pet', 'puppy', 'kitten', 'cattle', 'cow', 'goat', 'sheep'],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category
    
    return 'Other'

# WEB SCRAPER FOR JIJI SEARCH
def fetch_page(url, retry_count=0):
    """Fetch page with retry logic and random delays"""
    if retry_count >= MAX_RETRIES:
        print(f"  Max retries reached for {url}")
        return None
    
    try:
        # Add random delay to avoid rate limiting
        delay = REQUEST_DELAY + random.uniform(0.5, 2.0)
        time.sleep(delay)
        
        print(f"  Fetching: {url}")
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        
        # Check if we got a valid HTML response
        if 'text/html' in response.headers.get('Content-Type', ''):
            return response.text
        else:
            print(f"  Non-HTML response received")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  Request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {e}")
        time.sleep(REQUEST_DELAY * 2)  # Longer delay on failure
        return fetch_page(url, retry_count + 1)
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None

def extract_item_details(item_html, search_query, page_num, item_num):
    """Extract detailed information from a single item listing"""
    soup = BeautifulSoup(item_html, 'lxml')
    
    # Extract title
    title_elem = soup.find('div', class_=re.compile(r'.*title.*', re.I))
    if not title_elem:
        title_elem = soup.find('a', class_=re.compile(r'.*title.*', re.I))
    if not title_elem:
        title_elem = soup.find('h3')
    if not title_elem:
        title_elem = soup.find('div', class_=re.compile(r'name', re.I))
    
    title = title_elem.get_text(strip=True) if title_elem else "No Title"
    
    # Extract price
    price_elem = soup.find('div', class_=re.compile(r'.*price.*', re.I))
    if not price_elem:
        price_elem = soup.find('div', class_=re.compile(r'amount', re.I))
    if not price_elem:
        price_elem = soup.find('span', class_=re.compile(r'.*price.*', re.I))
    
    price_text = price_elem.get_text(strip=True) if price_elem else "Price not listed"
    price_value = parse_price(price_text)
    currency = extract_currency(price_text)
    
    # Extract location
    location_elem = soup.find('div', class_=re.compile(r'.*location.*', re.I))
    if not location_elem:
        location_elem = soup.find('span', class_=re.compile(r'.*location.*', re.I))
    if not location_elem:
        location_elem = soup.find('div', class_=re.compile(r'region', re.I))
    
    location = location_elem.get_text(strip=True) if location_elem else "Location not specified"
    
    # Extract link
    link_elem = soup.find('a', href=True)
    item_url = urljoin("https://jiji.com.et", link_elem['href']) if link_elem else ""
    
    # Extract date posted
    date_elem = soup.find('div', class_=re.compile(r'.*date.*', re.I))
    if not date_elem:
        date_elem = soup.find('div', class_=re.compile(r'.*time.*', re.I))
    if not date_elem:
        date_elem = soup.find('span', class_=re.compile(r'.*date.*', re.I))
    
    date_posted = date_elem.get_text(strip=True) if date_elem else "Recently"
    
    # Extract image URL
    img_elem = soup.find('img', src=True)
    image_url = img_elem['src'] if img_elem else ""
    
    # Determine condition (new/used)
    title_lower = title.lower()
    if any(word in title_lower for word in ['new', 'brand new', 'fresh', 'sealed']):
        condition = "New"
    elif any(word in title_lower for word in ['used', 'second', 'secondhand', 'pre-owned']):
        condition = "Used"
    else:
        condition = "Unknown"
    
    # Extract seller type if possible
    seller_elem = soup.find('div', class_=re.compile(r'.*seller.*', re.I))
    seller_type = seller_elem.get_text(strip=True) if seller_elem else "Unknown"
    
    return {
        "id": f"{search_query}_{page_num}_{item_num}",
        "title": title,
        "price_text": price_text,
        "price_value": price_value,
        "currency": currency,
        "location": location,
        "date_posted": date_posted,
        "condition": condition,
        "category": categorize_item(title, search_query),
        "seller_type": seller_type,
        "url": item_url,
        "image_url": image_url,
        "search_query": search_query,
        "page_number": page_num,
        "item_number": item_num,
        "scraped_at": datetime.now().isoformat(),
        "source": "Jiji Ethiopia"
    }

def scrape_jiji_search_page(search_query, page=1):
    """Scrape a single page of Jiji search results"""
    encoded_query = quote_plus(search_query)
    url = f"https://jiji.com.et/search?query={encoded_query}&page={page}"
    
    print(f"\n📄 Page {page}: {url}")
    
    html_content = fetch_page(url)
    if not html_content:
        print(f"  Failed to fetch page {page}")
        return []
    
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Try different selectors for item containers
    item_selectors = [
        'div.b-list-advert__item',
        'div.b-list-advert-base',
        'div.b-advert-list__item',
        'div[data-id*="advert"]',
        'div.advert-list-item',
        'div.search-list-item'
    ]
    
    items = []
    for selector in item_selectors:
        items = soup.select(selector)
        if items:
            print(f"  Found {len(items)} items with selector: {selector}")
            break
    
    if not items:
        # Fallback: look for any div that looks like a listing
        all_divs = soup.find_all('div')
        items = [div for div in all_divs if 'advert' in str(div.get('class', ''))]
        if items:
            print(f"  Found {len(items)} items using fallback method")
    
    if not items:
        print(f"  No items found on page {page}")
        # Save HTML for debugging
        debug_file = os.path.join(DATA_DIR, f"debug_page_{page}.html")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Saved HTML for debugging: {debug_file}")
        return []
    
    scraped_items = []
    for i, item_html in enumerate(items[:ITEMS_PER_PAGE], 1):
        try:
            item_data = extract_item_details(str(item_html), search_query, page, i)
            scraped_items.append(item_data)
            
            # Show progress for first few items
            if i <= 3:
                price_display = f"{item_data['price_value']:,.2f} {item_data['currency']}" if item_data['price_value'] else "Price N/A"
                print(f"    {i}. {item_data['title'][:50]}... - {price_display}")
                
        except Exception as e:
            print(f"    Error processing item {i}: {e}")
            continue
    
    if len(items) > 3:
        print(f"    ... and {len(items)-3} more items")
    
    return scraped_items

def scrape_jiji_search(search_query, max_pages=MAX_PAGES):
    """Scrape multiple pages from Jiji search results"""
    print(f"\n{'='*60}")
    print(f" SEARCHING JIJI FOR: '{search_query}'")
    print(f"{'='*60}")
    
    all_items = []
    
    for page in range(1, max_pages + 1):
        page_items = scrape_jiji_search_page(search_query, page)
        
        if not page_items:
            print(f"  No items found on page {page}, stopping.")
            break
        
        all_items.extend(page_items)
        
        # Check if we should continue to next page
        if len(page_items) < ITEMS_PER_PAGE:
            print(f"  Fewer items than expected on page {page}, might be last page.")
            break
        
        # Add delay between pages
        time.sleep(REQUEST_DELAY)
    
    print(f"\n Scraping complete: Found {len(all_items)} items for '{search_query}'")
    return all_items

# DATA ANALYSIS FUNCTIONS
def calculate_search_statistics(items, search_query):
    """Calculate comprehensive statistics for search results"""
    if not items:
        return {
            "search_query": search_query,
            "total_items": 0,
            "message": "No items found"
        }
    
    # Filter items with valid prices
    priced_items = [item for item in items if item['price_value'] is not None]
    prices = [item['price_value'] for item in priced_items]
    
    stats = {
        "search_query": search_query,
        "total_items": len(items),
        "items_with_price": len(priced_items),
        "scraped_at": datetime.now().isoformat()
    }
    
    if prices:
        # Basic statistics
        stats.update({
            "average_price": round(statistics.mean(prices), 2),
            "median_price": round(statistics.median(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "price_range": max(prices) - min(prices),
        })
        
        # Advanced statistics
        if len(prices) > 1:
            stats["std_dev"] = round(statistics.stdev(prices), 2)
            stats["variance"] = round(statistics.variance(prices), 2)
        
        # Quartiles
        if len(prices) >= 4:
            q1 = np.percentile(prices, 25)
            q3 = np.percentile(prices, 75)
            stats.update({
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(q3 - q1, 2)
            })
        
        # Most common price ranges
        price_bins = defaultdict(int)
        for price in prices:
            if price < 1000:
                bin_key = f"Under 1,000"
            elif price < 5000:
                bin_key = f"1,000 - 5,000"
            elif price < 10000:
                bin_key = f"5,000 - 10,000"
            elif price < 50000:
                bin_key = f"10,000 - 50,000"
            elif price < 100000:
                bin_key = f"50,000 - 100,000"
            elif price < 500000:
                bin_key = f"100,000 - 500,000"
            elif price < 1000000:
                bin_key = f"500,000 - 1,000,000"
            else:
                bin_key = f"Over 1,000,000"
            
            price_bins[bin_key] += 1
        
        stats["price_distribution"] = dict(sorted(price_bins.items(), key=lambda x: x[1], reverse=True))
    
    # Condition analysis
    conditions = defaultdict(int)
    for item in items:
        conditions[item.get('condition', 'Unknown')] += 1
    stats["condition_distribution"] = dict(conditions)
    
    # Location analysis
    locations = defaultdict(int)
    for item in items:
        locations[item.get('location', 'Unknown')] += 1
    # Get top 5 locations
    top_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]
    stats["top_locations"] = dict(top_locations)
    
    # Category analysis
    categories = defaultdict(int)
    category_prices = defaultdict(list)
    for item in items:
        cat = item.get('category', 'Other')
        categories[cat] += 1
        if item['price_value']:
            category_prices[cat].append(item['price_value'])
    
    category_stats = {}
    for cat, price_list in category_prices.items():
        if price_list:
            category_stats[cat] = {
                "count": categories[cat],
                "avg_price": round(statistics.mean(price_list), 2),
                "min_price": min(price_list),
                "max_price": max(price_list)
            }
    
    stats["category_analysis"] = category_stats
    
    return stats

def analyze_price_trends(items):
    """Analyze price trends based on condition and other factors"""
    if not items:
        return {}
    
    analysis = {}
    
    # New vs Used price comparison
    new_items = [item for item in items if item.get('condition') == 'New' and item['price_value']]
    used_items = [item for item in items if item.get('condition') == 'Used' and item['price_value']]
    
    if new_items and used_items:
        new_prices = [item['price_value'] for item in new_items]
        used_prices = [item['price_value'] for item in used_items]
        
        analysis['new_vs_used'] = {
            'new_count': len(new_items),
            'new_avg_price': round(statistics.mean(new_prices), 2),
            'used_count': len(used_items),
            'used_avg_price': round(statistics.mean(used_prices), 2),
            'price_difference_percent': round(
                ((statistics.mean(new_prices) - statistics.mean(used_prices)) / 
                 statistics.mean(used_prices)) * 100, 2
            ) if statistics.mean(used_prices) > 0 else 0
        }
    
    # Price by location (top locations)
    location_prices = defaultdict(list)
    for item in items:
        if item['price_value'] and item.get('location'):
            location_prices[item['location']].append(item['price_value'])
    
    top_location_stats = {}
    for location, prices in sorted(location_prices.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        if prices:
            top_location_stats[location] = {
                'count': len(prices),
                'avg_price': round(statistics.mean(prices), 2),
                'min_price': min(prices),
                'max_price': max(prices)
            }
    
    if top_location_stats:
        analysis['price_by_location'] = top_location_stats
    
    return analysis

def generate_recommendations(stats):
    """Generate buying/selling recommendations based on statistics"""
    if not stats or stats.get('total_items', 0) == 0:
        return {"message": "Insufficient data for recommendations"}
    
    recommendations = {
        "search_query": stats.get("search_query", ""),
        "generated_at": datetime.now().isoformat()
    }
    
    if 'average_price' in stats:
        avg_price = stats['average_price']
        median_price = stats.get('median_price', avg_price)
        std_dev = stats.get('std_dev', 0)
        
        # Buying recommendations
        recommendations['buying'] = {
            "good_price_range": f"{avg_price - std_dev:,.2f} - {avg_price + std_dev:,.2f}",
            "bargain_price": f"Below {avg_price - std_dev:,.2f}",
            "high_price": f"Above {avg_price + std_dev:,.2f}",
            "fair_price": f"Around {median_price:,.2f}",
            "recommendation": "Look for items within 1 standard deviation of average"
        }
        
        # Selling recommendations
        recommendations['selling'] = {
            "competitive_price": f"{median_price:,.2f} - {avg_price + std_dev:,.2f}",
            "premium_possible": f"If condition is excellent and location is prime",
            "quick_sale_price": f"Around {avg_price - (std_dev * 0.5):,.2f}",
            "recommendation": "Price competitively based on condition and location"
        }
    
    # Condition-based recommendations
    if 'condition_distribution' in stats:
        conditions = stats['condition_distribution']
        total = sum(conditions.values())
        
        if total > 0:
            condition_rec = {}
            for condition, count in conditions.items():
                percentage = (count / total) * 100
                if condition == 'New':
                    if percentage < 20:
                        condition_rec[condition] = "Rare, can command premium price"
                    elif percentage > 80:
                        condition_rec[condition] = "Common, price competitively"
                elif condition == 'Used':
                    if percentage > 60:
                        condition_rec[condition] = "Market is competitive, highlight unique features"
            
            if condition_rec:
                recommendations['condition_insights'] = condition_rec
    
    return recommendations

# DATA SAVING AND EXPORT
def save_search_results(items, search_query, stats=None, recommendations=None):
    """Save search results and analysis to files"""
    if not items:
        print(f"\n No data to save for '{search_query}'")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = re.sub(r'[^\w\s-]', '', search_query).replace(' ', '_').lower()[:50]
    
    # Create query-specific directory
    query_dir = os.path.join(DATA_DIR, query_slug)
    os.makedirs(query_dir, exist_ok=True)
    
    # Save raw data as JSON
    json_file = os.path.join(query_dir, f"{query_slug}_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    # Save as CSV
    csv_file = os.path.join(query_dir, f"{query_slug}_{timestamp}.csv")
    if items:
        df = pd.DataFrame(items)
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Save statistics
    if stats:
        stats_file = os.path.join(query_dir, f"stats_{query_slug}_{timestamp}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Save recommendations
    if recommendations:
        rec_file = os.path.join(query_dir, f"recommendations_{query_slug}_{timestamp}.json")
        with open(rec_file, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)
    
    # Generate and save report
    report_file = generate_comprehensive_report(items, stats, recommendations, query_slug, timestamp)
    
    print(f"\n Data saved to: {query_dir}")
    print(f"    Raw data (JSON): {json_file}")
    print(f"    CSV export: {csv_file}")
    if stats:
        print(f"    Statistics: {stats_file}")
    if recommendations:
        print(f"    Recommendations: {rec_file}")
    print(f"    Report: {report_file}")
    
    return {
        "directory": query_dir,
        "json_file": json_file,
        "csv_file": csv_file,
        "report_file": report_file
    }

def generate_comprehensive_report(items, stats, recommendations, query_slug, timestamp):
    """Generate a comprehensive text report"""
    report_file = os.path.join(DATA_DIR, query_slug, f"report_{query_slug}_{timestamp}.txt")
    
    report = f"""
{'='*80}
JIJI ETHIOPIA SEARCH ANALYSIS REPORT
{'='*80}

SEARCH DETAILS:
{'-'*40}
Search Query:      {stats.get('search_query', 'Unknown') if stats else 'Unknown'}
Search Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Items Found: {len(items)}
Items with Price:  {stats.get('items_with_price', 0) if stats else 0}

{'='*80}
PRICE ANALYSIS
{'='*80}
"""
    
    if stats and 'average_price' in stats:
        report += f"""
 PRICE STATISTICS:
{'-'*40}
Average Price:     {stats['average_price']:,.2f} {items[0]['currency'] if items else 'ETB'}
Median Price:      {stats.get('median_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}
Minimum Price:     {stats.get('min_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}
Maximum Price:     {stats.get('max_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}
Price Range:       {stats.get('price_range', 0):,.2f} {items[0]['currency'] if items else 'ETB'}
Standard Deviation:{stats.get('std_dev', 0):,.2f} {items[0]['currency'] if items else 'ETB'}

"""
        
        if 'q1' in stats:
            report += f"""
 PRICE QUARTILES:
{'-'*40}
25th Percentile (Q1): {stats['q1']:,.2f} {items[0]['currency'] if items else 'ETB'}
75th Percentile (Q3): {stats['q3']:,.2f} {items[0]['currency'] if items else 'ETB'}
Interquartile Range:  {stats['iqr']:,.2f} {items[0]['currency'] if items else 'ETB'}
"""
    
    # Price distribution
    if stats and 'price_distribution' in stats:
        report += f"""
 PRICE DISTRIBUTION:
{'-'*40}
"""
        for price_range, count in stats['price_distribution'].items():
            percentage = (count / stats['items_with_price']) * 100
            report += f"{price_range}: {count} items ({percentage:.1f}%)\n"
    
    # Condition distribution
    if stats and 'condition_distribution' in stats:
        report += f"""
 CONDITION DISTRIBUTION:
{'-'*40}
"""
        total = sum(stats['condition_distribution'].values())
        for condition, count in stats['condition_distribution'].items():
            percentage = (count / total) * 100
            report += f"{condition}: {count} items ({percentage:.1f}%)\n"
    
    # Top locations
    if stats and 'top_locations' in stats:
        report += f"""
📍 TOP LOCATIONS:
{'-'*40}
"""
        for location, count in stats['top_locations'].items():
            percentage = (count / stats['total_items']) * 100
            report += f"{location}: {count} items ({percentage:.1f}%)\n"
    
    # Category analysis
    if stats and 'category_analysis' in stats:
        report += f"""
🏷️ CATEGORY ANALYSIS:
{'-'*40}
"""
        for category, cat_stats in stats['category_analysis'].items():
            report += f"{category}:\n"
            report += f"  Count: {cat_stats['count']} items\n"
            report += f"  Avg Price: {cat_stats['avg_price']:,.2f} {items[0]['currency'] if items else 'ETB'}\n"
            report += f"  Price Range: {cat_stats['min_price']:,.2f} - {cat_stats['max_price']:,.2f}\n"
    
    # Recommendations
    if recommendations:
        report += f"""
{'='*80}
RECOMMENDATIONS
{'='*80}
"""
        if 'buying' in recommendations:
            report += f"""
🛒 BUYING RECOMMENDATIONS:
{'-'*40}
Good Price Range:    {recommendations['buying']['good_price_range']}
Bargain Price:       {recommendations['buying']['bargain_price']}
High Price Alert:    {recommendations['buying']['high_price']}
Fair Market Price:   {recommendations['buying']['fair_price']}
Advice:              {recommendations['buying']['recommendation']}
"""
        
        if 'selling' in recommendations:
            report += f"""
 SELLING RECOMMENDATIONS:
{'-'*40}
Competitive Price:   {recommendations['selling']['competitive_price']}
Premium Possible:    {recommendations['selling']['premium_possible']}
Quick Sale Price:    {recommendations['selling']['quick_sale_price']}
Advice:              {recommendations['selling']['recommendation']}
"""
    
    # Sample listings
    report += f"""
{'='*80}
SAMPLE LISTINGS (First 10)
{'='*80}
"""
    for i, item in enumerate(items[:10], 1):
        price_display = f"{item['price_value']:,.2f} {item['currency']}" if item['price_value'] else "Price N/A"
        report += f"""
{i}. {item['title']}
   Price: {price_display}
   Location: {item.get('location', 'N/A')}
   Condition: {item.get('condition', 'N/A')}
   Posted: {item.get('date_posted', 'N/A')}
"""
    
    report += f"""
{'='*80}
END OF REPORT
Generated by Jiji Ethiopia Scraper
{'='*80}
"""
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report_file

# INTERACTIVE DISPLAY
def display_search_summary(items, stats, search_query):
    """Display a summary of search results in the console"""
    if not items:
        print(f"\n❌ No items found for '{search_query}'")
        return
    
    print(f"\n{'='*80}")
    print(f" SEARCH RESULTS SUMMARY: '{search_query}'")
    print(f"{'='*80}")
    
    print(f"\n OVERVIEW:")
    print(f"   Total Items Found: {len(items)}")
    print(f"   Items with Price: {stats.get('items_with_price', 0)}")
    
    if 'average_price' in stats:
        print(f"\n PRICE ANALYSIS:")
        print(f"   Average Price: {stats['average_price']:,.2f} {items[0]['currency'] if items else 'ETB'}")
        print(f"   Median Price: {stats.get('median_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}")
        print(f"   Price Range: {stats.get('min_price', 0):,.2f} - {stats.get('max_price', 0):,.2f}")
        
        if 'q1' in stats:
            print(f"   25th Percentile: {stats['q1']:,.2f}")
            print(f"   75th Percentile: {stats['q3']:,.2f}")
    
    if 'condition_distribution' in stats:
        print(f"\n CONDITION DISTRIBUTION:")
        for condition, count in stats['condition_distribution'].items():
            percentage = (count / sum(stats['condition_distribution'].values())) * 100
            print(f"   {condition}: {count} items ({percentage:.1f}%)")
    
    if 'top_locations' in stats:
        print(f"\n TOP LOCATIONS:")
        for location, count in list(stats['top_locations'].items())[:3]:
            print(f"   {location}: {count} items")
    
    # Display sample items
    print(f"\n SAMPLE LISTINGS:")
    for i, item in enumerate(items[:5], 1):
        price_display = f"{item['price_value']:,.2f} {item['currency']}" if item['price_value'] else "Price N/A"
        print(f"   {i}. {item['title'][:60]}...")
        print(f"      {price_display} | {item.get('location', 'N/A')} | {item.get('condition', 'N/A')}")

# MAIN EXECUTION
def main():
    """Main execution function"""
    print(f"""
{'='*80}
 JIJI ETHIOPIA SMART SEARCH SCRAPER
{'='*80}
Scrape and analyze prices from Jiji Ethiopia search results
""")
    
    while True:
        print(f"\n{'='*80}")
        print("MAIN MENU")
        print(f"{'='*80}")
        print("1.  Search for items (e.g., 'iphone 12', 'toyota yaris')")
        print("2.  View recent searches")
        print("3.  Export data")
        print("4.  Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Get search query
            search_query = input("\nEnter search query (e.g., 'iphone 12', 'toyota yaris'): ").strip()
            
            if not search_query:
                print(" Please enter a search query")
                continue
            
            # Get number of pages to scrape
            try:
                pages_input = input(f"Pages to scrape (1-{MAX_PAGES}, default=3): ").strip()
                pages = int(pages_input) if pages_input else 3
                pages = max(1, min(pages, MAX_PAGES))
            except:
                pages = 3
            
            print(f"\n Searching for '{search_query}' ({pages} pages)...")
            
            # Scrape data
            items = scrape_jiji_search(search_query, pages)
            
            if not items:
                print(f"\n No items found for '{search_query}'")
                continue
            
            # Calculate statistics
            stats = calculate_search_statistics(items, search_query)
            
            # Analyze price trends
            trends = analyze_price_trends(items)
            if trends:
                stats.update(trends)
            
            # Generate recommendations
            recommendations = generate_recommendations(stats)
            
            # Display summary
            display_search_summary(items, stats, search_query)
            
            # Save data
            saved_files = save_search_results(items, search_query, stats, recommendations)
            
            # Ask what to do next
            while True:
                print(f"\n{'='*80}")
                print(f"OPTIONS FOR '{search_query}'")
                print(f"{'='*80}")
                print("1.  View detailed report")
                print("2.  View price distribution")
                print("3.  View location analysis")
                print("4.  New search")
                print("5.  Back to main menu")
                
                sub_choice = input("\nSelect option (1-5): ").strip()
                
                if sub_choice == "1":
                    # View report file
                    if saved_files and os.path.exists(saved_files['report_file']):
                        with open(saved_files['report_file'], 'r', encoding='utf-8') as f:
                            print(f"\n{'-'*80}")
                            print("REPORT CONTENT:")
                            print(f"{'-'*80}")
                            print(f.read()[:2000] + "...\n(Full report saved to file)")
                    else:
                        print("Report file not found")
                
                elif sub_choice == "2":
                    # Price distribution
                    if stats and 'price_distribution' in stats:
                        print(f"\n PRICE DISTRIBUTION FOR '{search_query}':")
                        print(f"{'-'*40}")
                        for price_range, count in stats['price_distribution'].items():
                            percentage = (count / stats['items_with_price']) * 100
                            bar = "█" * int(percentage / 5)  # Each █ represents 5%
                            print(f"{price_range:20} {count:3} items | {bar} ({percentage:.1f}%)")
                
                elif sub_choice == "3":
                    # Location analysis
                    if stats and 'top_locations' in stats:
                        print(f"\n📍 TOP LOCATIONS FOR '{search_query}':")
                        print(f"{'-'*40}")
                        for location, count in stats['top_locations'].items():
                            percentage = (count / stats['total_items']) * 100
                            print(f"{location:30} {count:3} items ({percentage:.1f}%)")
                
                elif sub_choice == "4":
                    break  # Break inner loop for new search
                
                elif sub_choice == "5":
                    return  # Return to main menu
                
                else:
                    print("Invalid option")
        
        elif choice == "2":
            # View recent searches
            print(f"\n RECENT SEARCHES IN DATA DIRECTORY:")
            print(f"{'='*80}")
            
            if os.path.exists(DATA_DIR):
                subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
                
                if subdirs:
                    for subdir in subdirs[-10:]:  # Show last 10 searches
                        subdir_path = os.path.join(DATA_DIR, subdir)
                        files = os.listdir(subdir_path) if os.path.exists(subdir_path) else []
                        json_files = [f for f in files if f.endswith('.json') and 'stats' not in f]
                        
                        if json_files:
                            latest_file = max([os.path.join(subdir_path, f) for f in json_files], 
                                            key=os.path.getctime)
                            try:
                                with open(latest_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    if data:
                                        item_count = len(data)
                                        search_query = data[0].get('search_query', subdir.replace('_', ' '))
                                        print(f"🔍 {search_query}: {item_count} items")
                            except:
                                print(f" {subdir.replace('_', ' ')}")
                else:
                    print("No search data found")
            else:
                print("Data directory not found")
        
        elif choice == "3":
            # Export data
            print(f"\n AVAILABLE DATA FOR EXPORT:")
            print(f"{'='*80}")
            
            if os.path.exists(DATA_DIR):
                subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
                
                if subdirs:
                    for i, subdir in enumerate(subdirs, 1):
                        print(f"{i}. {subdir.replace('_', ' ')}")
                    
                    try:
                        export_choice = int(input(f"\nSelect search to export (1-{len(subdirs)}): "))
                        if 1 <= export_choice <= len(subdirs):
                            selected_dir = subdirs[export_choice - 1]
                            dir_path = os.path.join(DATA_DIR, selected_dir)
                            
                            # Find all CSV files in the directory
                            csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                            
                            if csv_files:
                                latest_csv = max([os.path.join(dir_path, f) for f in csv_files], 
                                               key=os.path.getctime)
                                
                                # Create export directory
                                export_dir = os.path.join(BASE_DIR, "exports")
                                os.makedirs(export_dir, exist_ok=True)
                                
                                # Export file name
                                export_name = f"jiji_export_{selected_dir}_{datetime.now().strftime('%Y%m%d')}.csv"
                                export_path = os.path.join(export_dir, export_name)
                                
                                # Copy the file
                                import shutil
                                shutil.copy2(latest_csv, export_path)
                                
                                print(f"\n Data exported to: {export_path}")
                                
                                # Show preview
                                try:
                                    df = pd.read_csv(export_path)
                                    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                                    print(f"\n   First 5 rows:")
                                    print(df.head().to_string())
                                except Exception as e:
                                    print(f"   Could not preview data: {e}")
                            else:
                                print("No CSV files found in the directory")
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a valid number")
                else:
                    print("No search data available for export")
            else:
                print("Data directory not found")
        
        elif choice == "4":
            print(f"\n Thank you for using Jiji Ethiopia Scraper!")
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()