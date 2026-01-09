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
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Scraper settings
REQUEST_DELAY = 2
MAX_RETRIES = 3
TIMEOUT = 15
MAX_PAGES = 10  # Increased maximum pages
ITEMS_PER_PAGE = 40

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

# NEW FEATURE: Price tracking database
class PriceDatabase:
    """Database for tracking price history"""
    
    def __init__(self, db_path: str = os.path.join(DATA_DIR, "price_history.db")):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for price tracking"""
        try:
            import sqlite3
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT NOT NULL,
                    title TEXT NOT NULL,
                    price REAL,
                    currency TEXT,
                    condition TEXT,
                    location TEXT,
                    date_posted TEXT,
                    url TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(search_query, title, price, scraped_at)
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT NOT NULL,
                    pages_scraped INTEGER,
                    total_items INTEGER,
                    average_price REAL,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
        except ImportError:
            print("Note: SQLite not available. Using JSON fallback for price tracking.")
            self.use_sqlite = False
        except Exception as e:
            print(f"Database initialization error: {e}")
            self.use_sqlite = False
    
    def save_price_data(self, items: List[Dict], search_query: str):
        """Save price data to database"""
        if not hasattr(self, 'use_sqlite') or not self.use_sqlite:
            return self._save_to_json(items, search_query)
        
        try:
            for item in items:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO price_history 
                    (search_query, title, price, currency, condition, location, date_posted, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    search_query,
                    item.get('title', ''),
                    item.get('price_value'),
                    item.get('currency', 'ETB'),
                    item.get('condition', 'Unknown'),
                    item.get('location', 'Unknown'),
                    item.get('date_posted', ''),
                    item.get('url', '')
                ))
            
            # Save search session info
            prices = [item['price_value'] for item in items if item.get('price_value')]
            avg_price = statistics.mean(prices) if prices else 0
            
            self.cursor.execute('''
                INSERT INTO search_sessions 
                (search_query, pages_scraped, total_items, average_price)
                VALUES (?, ?, ?, ?)
            ''', (
                search_query,
                max([item.get('page_number', 0) for item in items], default=0),
                len(items),
                avg_price
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            return self._save_to_json(items, search_query)
    
    def _save_to_json(self, items: List[Dict], search_query: str):
        """Fallback: Save price data to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d")
            history_file = os.path.join(DATA_DIR, f"price_history_{search_query}_{timestamp}.json")
            
            history_data = {
                "search_query": search_query,
                "scraped_at": datetime.now().isoformat(),
                "items": items
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    def get_price_trend(self, search_query: str, days: int = 30):
        """Get price trend for a search query over time"""
        if not hasattr(self, 'use_sqlite') or not self.use_sqlite:
            return self._get_trend_from_json(search_query, days)
        
        try:
            self.cursor.execute('''
                SELECT DATE(scraped_at) as date, AVG(price) as avg_price, COUNT(*) as count
                FROM price_history
                WHERE search_query = ? AND scraped_at >= DATE('now', '-? days')
                GROUP BY DATE(scraped_at)
                ORDER BY date
            ''', (search_query, days))
            
            results = self.cursor.fetchall()
            return [
                {"date": row[0], "average_price": row[1], "count": row[2]}
                for row in results
            ]
        except Exception as e:
            print(f"Error getting price trend: {e}")
            return []
    
    def _get_trend_from_json(self, search_query: str, days: int = 30):
        """Fallback: Get price trend from JSON files"""
        trend_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        pattern = re.compile(f"price_history_{re.escape(search_query)}_.*\\.json$")
        
        for filename in os.listdir(DATA_DIR):
            if pattern.match(filename):
                file_path = os.path.join(DATA_DIR, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    scraped_date = datetime.fromisoformat(data.get('scraped_at', '').replace('Z', '+00:00'))
                    if scraped_date >= cutoff_date:
                        prices = [item.get('price_value') for item in data.get('items', []) 
                                 if item.get('price_value')]
                        if prices:
                            trend_data.append({
                                "date": scraped_date.strftime("%Y-%m-%d"),
                                "average_price": statistics.mean(prices),
                                "count": len(prices)
                            })
                except Exception as e:
                    continue
        
        return sorted(trend_data, key=lambda x: x["date"])

# Initialize price database
price_db = PriceDatabase()

# PRICE PARSING
def parse_price(price_text: str) -> Optional[float]:
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

def extract_currency(price_text: str) -> str:
    """Extract currency from price text"""
    if not price_text:
        return 'ETB'
    
    price_text = price_text.upper()
    if 'USD' in price_text or '$' in price_text:
        return 'USD'
    elif 'ETB' in price_text or 'BIRR' in price_text:
        return 'ETB'
    else:
        return 'ETB'

# NEW FEATURE: Advanced price normalization
def normalize_price(price_value: float, currency: str) -> Dict[str, Any]:
    """Normalize price to ETB and provide conversion rates"""
    conversion_rates = {
        'USD': 57.5,  # Approximate USD to ETB rate
        'ETB': 1.0,
    }
    
    rate = conversion_rates.get(currency.upper(), 1.0)
    etb_price = price_value * rate
    
    return {
        "original_price": price_value,
        "original_currency": currency,
        "etb_price": etb_price,
        "conversion_rate": rate,
        "usd_price": etb_price / conversion_rates['USD'] if currency != 'USD' else price_value
    }

# CATEGORY CLASSIFICATION
def categorize_item(title: str, search_query: Optional[str] = None) -> str:
    """Categorize items based on keywords"""
    title_lower = title.lower()
    
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

# NEW FEATURE: Smart search query enhancement
def enhance_search_query(query: str) -> str:
    """Enhance search query with common variations"""
    query_lower = query.lower()
    
    # Add common variations for better results
    enhancements = {
        'iphone': ['iphone', 'iphones'],
        'samsung': ['samsung', 'galaxy'],
        'laptop': ['laptop', 'notebook'],
        'car': ['car', 'cars', 'vehicle'],
        'house': ['house', 'home', 'apartment'],
    }
    
    for key, variations in enhancements.items():
        if key in query_lower:
            # Don't modify if already specific
            if not any(variation in query_lower for variation in variations[1:]):
                return query
    
    return query

# WEB SCRAPER FUNCTIONS
def fetch_page(url: str, retry_count: int = 0) -> Optional[str]:
    """Fetch page with retry logic and random delays"""
    if retry_count >= MAX_RETRIES:
        print(f"  Maximum retries reached for {url}")
        return None
    
    try:
        delay = REQUEST_DELAY + random.uniform(0.5, 2.0)
        time.sleep(delay)
        
        print(f"  Fetching page: {url}")
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        
        if 'text/html' in response.headers.get('Content-Type', ''):
            return response.text
        else:
            print(f"  Received non-HTML response")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  Request failed (attempt {retry_count + 1}/{MAX_RETRIES}): {str(e)[:100]}")
        time.sleep(REQUEST_DELAY * 2)
        return fetch_page(url, retry_count + 1)
    except Exception as e:
        print(f"  Unexpected error: {str(e)[:100]}")
        return None

# NEW FEATURE: Improved item extraction with more fields
def extract_item_details(item_html: str, search_query: str, page_num: int, item_num: int) -> Dict[str, Any]:
    """Extract detailed information from a single item listing"""
    soup = BeautifulSoup(item_html, 'lxml')
    
    # Extract title with multiple selector options
    title_selectors = [
        ('div', re.compile(r'.*title.*', re.I)),
        ('a', re.compile(r'.*title.*', re.I)),
        ('h3', None),
        ('div', re.compile(r'name', re.I)),
        ('span', re.compile(r'.*title.*', re.I)),
    ]
    
    title = "No Title"
    for tag, cls in title_selectors:
        elem = soup.find(tag, class_=cls) if cls else soup.find(tag)
        if elem:
            title = elem.get_text(strip=True)
            break
    
    # Extract price
    price_selectors = [
        ('div', re.compile(r'.*price.*', re.I)),
        ('div', re.compile(r'amount', re.I)),
        ('span', re.compile(r'.*price.*', re.I)),
        ('div', re.compile(r'cost', re.I)),
    ]
    
    price_text = "Price not listed"
    for tag, cls in price_selectors:
        elem = soup.find(tag, class_=cls)
        if elem:
            price_text = elem.get_text(strip=True)
            break
    
    price_value = parse_price(price_text)
    currency = extract_currency(price_text)
    
    # Extract location
    location_selectors = [
        ('div', re.compile(r'.*location.*', re.I)),
        ('span', re.compile(r'.*location.*', re.I)),
        ('div', re.compile(r'region', re.I)),
        ('div', re.compile(r'area', re.I)),
    ]
    
    location = "Location not specified"
    for tag, cls in location_selectors:
        elem = soup.find(tag, class_=cls)
        if elem:
            location = elem.get_text(strip=True)
            break
    
    # Extract link
    link_elem = soup.find('a', href=True)
    item_url = urljoin("https://jiji.com.et", link_elem['href']) if link_elem else ""
    
    # Extract date posted
    date_selectors = [
        ('div', re.compile(r'.*date.*', re.I)),
        ('div', re.compile(r'.*time.*', re.I)),
        ('span', re.compile(r'.*date.*', re.I)),
        ('div', re.compile(r'posted', re.I)),
    ]
    
    date_posted = "Recently"
    for tag, cls in date_selectors:
        elem = soup.find(tag, class_=cls)
        if elem:
            date_posted = elem.get_text(strip=True)
            break
    
    # Extract image URL
    img_elem = soup.find('img', src=True)
    image_url = img_elem['src'] if img_elem else ""
    
    # Determine condition
    title_lower = title.lower()
    if any(word in title_lower for word in ['new', 'brand new', 'fresh', 'sealed']):
        condition = "New"
    elif any(word in title_lower for word in ['used', 'second', 'secondhand', 'pre-owned', 'old']):
        condition = "Used"
    elif any(word in title_lower for word in ['refurbished', 'refurbish']):
        condition = "Refurbished"
    else:
        condition = "Unknown"
    
    # Extract seller information if available
    seller_selectors = [
        ('div', re.compile(r'.*seller.*', re.I)),
        ('div', re.compile(r'.*vendor.*', re.I)),
        ('span', re.compile(r'.*seller.*', re.I)),
    ]
    
    seller_type = "Unknown"
    for tag, cls in seller_selectors:
        elem = soup.find(tag, class_=cls)
        if elem:
            seller_type = elem.get_text(strip=True)
            break
    
    # NEW: Extract description if available
    desc_elem = soup.find('div', class_=re.compile(r'.*description.*', re.I))
    description = desc_elem.get_text(strip=True)[:200] if desc_elem else ""
    
    # NEW: Calculate price score (for sorting)
    price_score = 0
    if price_value:
        # Lower prices get higher scores (better deals)
        price_score = 1000 / (price_value + 1) if price_value > 0 else 0
    
    item_data = {
        "id": f"{search_query}_{page_num}_{item_num}",
        "title": title,
        "price_text": price_text,
        "price_value": price_value,
        "currency": currency,
        "normalized_price": normalize_price(price_value, currency) if price_value else None,
        "location": location,
        "date_posted": date_posted,
        "condition": condition,
        "category": categorize_item(title, search_query),
        "seller_type": seller_type,
        "description": description,
        "url": item_url,
        "image_url": image_url,
        "search_query": search_query,
        "page_number": page_num,
        "item_number": item_num,
        "price_score": round(price_score, 2),
        "scraped_at": datetime.now().isoformat(),
        "source": "Jiji Ethiopia"
    }
    
    return item_data

def scrape_jiji_search_page(search_query: str, page: int = 1) -> List[Dict[str, Any]]:
    """Scrape a single page of Jiji search results"""
    encoded_query = quote_plus(search_query)
    url = f"https://jiji.com.et/search?query={encoded_query}&page={page}"
    
    print(f"\nPage {page}: {url}")
    
    html_content = fetch_page(url)
    if not html_content:
        print(f"  Failed to fetch page {page}")
        return []
    
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Try multiple selectors for item containers
    item_selectors = [
        'div.b-list-advert__item',
        'div.b-list-advert-base',
        'div.b-advert-list__item',
        'div[data-id*="advert"]',
        'div.advert-list-item',
        'div.search-list-item',
        'div.listing-item',
        'div.product-item',
    ]
    
    items = []
    for selector in item_selectors:
        found_items = soup.select(selector)
        if found_items:
            print(f"  Found {len(found_items)} items with selector: {selector}")
            items = found_items
            break
    
    if not items:
        # Fallback: look for divs with advert-related classes
        all_divs = soup.find_all('div', class_=True)
        items = [div for div in all_divs if any(keyword in div.get('class', '') 
                for keyword in ['advert', 'listing', 'product', 'item'])]
        if items:
            print(f"  Found {len(items)} items using fallback method")
    
    if not items:
        print(f"  No items found on page {page}")
        return []
    
    scraped_items = []
    for i, item_html in enumerate(items[:ITEMS_PER_PAGE], 1):
        try:
            item_data = extract_item_details(str(item_html), search_query, page, i)
            scraped_items.append(item_data)
            
            if i <= 3:
                price_display = f"{item_data['price_value']:,.2f} {item_data['currency']}" if item_data['price_value'] else "Price N/A"
                print(f"    {i}. {item_data['title'][:50]}... - {price_display}")
                
        except Exception as e:
            print(f"    Error processing item {i}: {str(e)[:50]}")
            continue
    
    if len(items) > 3:
        print(f"    ... and {len(items)-3} more items")
    
    return scraped_items

def scrape_jiji_search(search_query: str, max_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
    """Scrape multiple pages from Jiji search results"""
    print(f"\n" + "="*60)
    print(f"SEARCHING JIJI FOR: '{search_query}'")
    print("="*60)
    
    enhanced_query = enhance_search_query(search_query)
    if enhanced_query != search_query:
        print(f"Using enhanced query: '{enhanced_query}'")
    
    all_items = []
    
    for page in range(1, max_pages + 1):
        page_items = scrape_jiji_search_page(enhanced_query, page)
        
        if not page_items:
            print(f"  No items found on page {page}, stopping.")
            break
        
        all_items.extend(page_items)
        
        if len(page_items) < ITEMS_PER_PAGE * 0.5:  # If less than 50% of expected items
            print(f"  Few items found on page {page}, might be last page.")
            break
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\nScraping complete: Found {len(all_items)} items for '{enhanced_query}'")
    return all_items

# NEW FEATURE: Advanced data analysis functions
class PriceAnalyzer:
    """Advanced price analysis and recommendations"""
    
    @staticmethod
    def detect_outliers(prices: List[float]) -> Dict[str, Any]:
        """Detect outliers in price data using IQR method"""
        if len(prices) < 4:
            return {"outliers": [], "threshold_low": 0, "threshold_high": 0}
        
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = [p for p in prices if p < lower_bound or p > upper_bound]
        
        return {
            "outliers": outliers,
            "threshold_low": lower_bound,
            "threshold_high": upper_bound,
            "outlier_count": len(outliers),
            "percentage": (len(outliers) / len(prices)) * 100
        }
    
    @staticmethod
    def calculate_price_elasticity(items: List[Dict]) -> float:
        """Calculate approximate price elasticity based on condition distribution"""
        new_items = [i for i in items if i.get('condition') == 'New']
        used_items = [i for i in items if i.get('condition') == 'Used']
        
        if not new_items or not used_items:
            return 0.0
        
        avg_new = statistics.mean([i['price_value'] for i in new_items if i.get('price_value')])
        avg_used = statistics.mean([i['price_value'] for i in used_items if i.get('price_value')])
        
        price_diff = (avg_new - avg_used) / avg_used
        quantity_diff = (len(new_items) - len(used_items)) / len(used_items)
        
        if quantity_diff != 0:
            elasticity = price_diff / quantity_diff
            return round(elasticity, 3)
        
        return 0.0
    
    @staticmethod
    def generate_market_report(items: List[Dict], search_query: str) -> Dict[str, Any]:
        """Generate comprehensive market report"""
        if not items:
            return {"error": "No items found"}
        
        priced_items = [i for i in items if i.get('price_value')]
        prices = [i['price_value'] for i in priced_items]
        
        if not prices:
            return {"error": "No valid prices found"}
        
        # Basic statistics
        stats = {
            "mean": statistics.mean(prices),
            "median": statistics.median(prices),
            "mode": statistics.mode(prices) if len(prices) > 1 else prices[0],
            "std_dev": statistics.stdev(prices) if len(prices) > 1 else 0,
            "variance": statistics.variance(prices) if len(prices) > 1 else 0,
            "range": max(prices) - min(prices),
            "cv": (statistics.stdev(prices) / statistics.mean(prices)) * 100 if statistics.mean(prices) > 0 else 0,
        }
        
        # Market segmentation
        conditions = defaultdict(list)
        locations = defaultdict(list)
        
        for item in priced_items:
            conditions[item.get('condition', 'Unknown')].append(item['price_value'])
            locations[item.get('location', 'Unknown')].append(item['price_value'])
        
        condition_stats = {}
        for condition, price_list in conditions.items():
            if price_list:
                condition_stats[condition] = {
                    "count": len(price_list),
                    "avg_price": statistics.mean(price_list),
                    "min_price": min(price_list),
                    "max_price": max(price_list)
                }
        
        # Top 5 locations by item count
        top_locations = sorted(
            [(loc, len(prices)) for loc, prices in locations.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_location_stats = {}
        for location, _ in top_locations:
            if location in locations:
                price_list = locations[location]
                top_location_stats[location] = {
                    "count": len(price_list),
                    "avg_price": statistics.mean(price_list),
                    "min_price": min(price_list),
                    "max_price": max(price_list)
                }
        
        # Price distribution analysis
        price_bins = [0, 1000, 5000, 10000, 50000, 100000, 500000, float('inf')]
        bin_labels = [
            "Under 1,000",
            "1,000 - 5,000", 
            "5,000 - 10,000",
            "10,000 - 50,000",
            "50,000 - 100,000",
            "100,000 - 500,000",
            "Over 500,000"
        ]
        
        distribution = defaultdict(int)
        for price in prices:
            for i in range(len(price_bins)-1):
                if price_bins[i] <= price < price_bins[i+1]:
                    distribution[bin_labels[i]] += 1
                    break
        
        # Market competitiveness score (0-100)
        price_range_ratio = stats['range'] / stats['mean'] if stats['mean'] > 0 else 0
        cv_score = min(100, max(0, 100 - stats['cv']))
        item_count_score = min(100, len(priced_items) / 10)
        
        competitiveness = (cv_score * 0.4 + item_count_score * 0.3 + (100 - price_range_ratio * 100) * 0.3)
        
        return {
            "search_query": search_query,
            "total_items": len(items),
            "priced_items": len(priced_items),
            "basic_statistics": {k: round(v, 2) for k, v in stats.items()},
            "condition_analysis": {k: {sk: round(sv, 2) for sk, sv in v.items()} 
                                  for k, v in condition_stats.items()},
            "location_analysis": {k: {sk: round(sv, 2) for sk, sv in v.items()} 
                                 for k, v in top_location_stats.items()},
            "price_distribution": dict(distribution),
            "market_competitiveness": round(competitiveness, 1),
            "analysis_date": datetime.now().isoformat()
        }

# DATA PROCESSING FUNCTIONS
def calculate_search_statistics(items: List[Dict], search_query: str) -> Dict[str, Any]:
    """Calculate comprehensive statistics for search results"""
    if not items:
        return {
            "search_query": search_query,
            "total_items": 0,
            "message": "No items found"
        }
    
    # Use PriceAnalyzer for advanced statistics
    analyzer = PriceAnalyzer()
    market_report = analyzer.generate_market_report(items, search_query)
    
    if "error" in market_report:
        # Fallback to basic statistics
        priced_items = [item for item in items if item['price_value'] is not None]
        prices = [item['price_value'] for item in priced_items]
        
        if not prices:
            return {
                "search_query": search_query,
                "total_items": len(items),
                "items_with_price": 0,
                "scraped_at": datetime.now().isoformat()
            }
        
        stats = {
            "search_query": search_query,
            "total_items": len(items),
            "items_with_price": len(priced_items),
            "average_price": round(statistics.mean(prices), 2),
            "median_price": round(statistics.median(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "price_range": max(prices) - min(prices),
            "scraped_at": datetime.now().isoformat()
        }
        
        if len(prices) > 1:
            stats["std_dev"] = round(statistics.stdev(prices), 2)
        
        return stats
    
    return market_report

def analyze_price_trends(items: List[Dict]) -> Dict[str, Any]:
    """Analyze price trends based on various factors"""
    if not items:
        return {}
    
    analysis = {}
    analyzer = PriceAnalyzer()
    
    # Price outliers analysis
    prices = [item['price_value'] for item in items if item.get('price_value')]
    if prices:
        outliers = analyzer.detect_outliers(prices)
        analysis['price_outliers'] = outliers
    
    # Condition-based analysis
    new_items = [item for item in items if item.get('condition') == 'New' and item.get('price_value')]
    used_items = [item for item in items if item.get('condition') == 'Used' and item.get('price_value')]
    
    if new_items and used_items:
        new_prices = [item['price_value'] for item in new_items]
        used_prices = [item['price_value'] for item in used_items]
        
        analysis['condition_comparison'] = {
            'new_count': len(new_items),
            'new_avg_price': round(statistics.mean(new_prices), 2),
            'used_count': len(used_items),
            'used_avg_price': round(statistics.mean(used_prices), 2),
            'price_difference': round(statistics.mean(new_prices) - statistics.mean(used_prices), 2),
            'price_difference_percent': round(
                ((statistics.mean(new_prices) - statistics.mean(used_prices)) / 
                 statistics.mean(used_prices)) * 100, 2
            ) if statistics.mean(used_prices) > 0 else 0
        }
    
    # Location-based analysis
    location_data = defaultdict(list)
    for item in items:
        if item.get('price_value') and item.get('location'):
            location_data[item['location']].append(item['price_value'])
    
    if location_data:
        location_stats = {}
        for location, prices in location_data.items():
            if prices:
                location_stats[location] = {
                    'count': len(prices),
                    'avg_price': round(statistics.mean(prices), 2),
                    'min_price': min(prices),
                    'max_price': max(prices),
                    'price_range': max(prices) - min(prices)
                }
        
        # Sort by count and take top 5
        sorted_locations = sorted(location_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        analysis['location_analysis'] = dict(sorted_locations)
    
    return analysis

def generate_recommendations(stats: Dict[str, Any]) -> Dict[str, Any]:
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
        recommendations['buying_recommendations'] = {
            "optimal_price_range": f"{avg_price - std_dev:,.0f} - {avg_price + std_dev:,.0f} ETB",
            "good_deal_threshold": f"Below {avg_price - std_dev:,.0f} ETB",
            "premium_price_threshold": f"Above {avg_price + std_dev:,.0f} ETB",
            "fair_market_value": f"Approximately {median_price:,.0f} ETB",
            "advice": [
                "Consider items within one standard deviation of the average price",
                "Verify condition and seller reputation for premium-priced items",
                "Look for items with detailed descriptions and clear photos"
            ]
        }
        
        # Selling recommendations
        recommendations['selling_recommendations'] = {
            "competitive_pricing": f"{median_price:,.0f} - {avg_price + std_dev:,.0f} ETB",
            "quick_sale_pricing": f"{avg_price - (std_dev * 0.5):,.0f} ETB",
            "premium_pricing_conditions": "Excellent condition with original accessories and warranty",
            "strategic_pricing_tips": [
                "Price slightly below median for faster sales",
                "Include high-quality photos from multiple angles",
                "Provide detailed specifications and condition report",
                "Mention original purchase date and warranty if available"
            ]
        }
    
    # Market condition insights
    if 'condition_distribution' in stats:
        conditions = stats['condition_distribution']
        total = sum(conditions.values())
        
        if total > 0:
            market_insights = []
            
            new_percentage = (conditions.get('New', 0) / total) * 100
            used_percentage = (conditions.get('Used', 0) / total) * 100
            
            if new_percentage < 15:
                market_insights.append("Limited new items available - sellers may command premium prices")
            elif new_percentage > 60:
                market_insights.append("Abundant new items - competitive pricing recommended")
            
            if used_percentage > 70:
                market_insights.append("Market dominated by used items - emphasize condition and provenance")
            
            if market_insights:
                recommendations['market_insights'] = market_insights
    
    # Location-based recommendations
    if 'top_locations' in stats and len(stats['top_locations']) > 0:
        top_location = list(stats['top_locations'].keys())[0]
        location_percentage = (list(stats['top_locations'].values())[0] / stats['total_items']) * 100
        
        if location_percentage > 40:
            recommendations['location_analysis'] = {
                "primary_market": top_location,
                "market_share": f"{location_percentage:.1f}%",
                "recommendation": f"Focus search efforts in {top_location} for best selection"
            }
    
    return recommendations

# NEW FEATURE: Data visualization
def create_price_visualizations(items: List[Dict], search_query: str, save_dir: str):
    """Create visualizations of price data"""
    if not items or len(items) < 5:
        return None
    
    priced_items = [item for item in items if item.get('price_value')]
    if len(priced_items) < 5:
        return None
    
    prices = [item['price_value'] for item in priced_items]
    
    try:
        # Create visualizations directory
        viz_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Price distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(prices, bins=20, edgecolor='black', alpha=0.7)
        plt.title(f'Price Distribution: {search_query}')
        plt.xlabel('Price (ETB)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        hist_file = os.path.join(viz_dir, f'histogram_{search_query}_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(hist_file, dpi=150)
        plt.close()
        
        # 2. Box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot(prices, vert=True, patch_artist=True)
        plt.title(f'Price Range: {search_query}')
        plt.ylabel('Price (ETB)')
        plt.grid(True, alpha=0.3)
        box_file = os.path.join(viz_dir, f'boxplot_{search_query}_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(box_file, dpi=150)
        plt.close()
        
        # 3. Price by condition (if available)
        conditions_data = {}
        for item in priced_items:
            condition = item.get('condition', 'Unknown')
            if condition not in conditions_data:
                conditions_data[condition] = []
            conditions_data[condition].append(item['price_value'])
        
        if len(conditions_data) > 1:
            plt.figure(figsize=(10, 6))
            condition_labels = list(conditions_data.keys())
            condition_prices = [conditions_data[label] for label in condition_labels]
            
            plt.boxplot(condition_prices, labels=condition_labels)
            plt.title(f'Price by Condition: {search_query}')
            plt.ylabel('Price (ETB)')
            plt.grid(True, alpha=0.3)
            condition_file = os.path.join(viz_dir, f'condition_comparison_{search_query}_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(condition_file, dpi=150)
            plt.close()
        
        return {
            "histogram": hist_file,
            "boxplot": box_file,
            "condition_comparison": condition_file if 'condition_file' in locals() else None
        }
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None

# DATA SAVING AND EXPORT
def save_search_results(items: List[Dict], search_query: str, 
                       stats: Optional[Dict] = None, 
                       recommendations: Optional[Dict] = None) -> Optional[Dict]:
    """Save search results and analysis to files"""
    if not items:
        print(f"No data to save for '{search_query}'")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = re.sub(r'[^\w\s-]', '', search_query).replace(' ', '_').lower()[:50]
    
    # Create query-specific directory
    query_dir = os.path.join(DATA_DIR, query_slug)
    os.makedirs(query_dir, exist_ok=True)
    
    # Save raw data
    json_file = os.path.join(query_dir, f"{query_slug}_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    # Save as CSV with enhanced formatting
    csv_file = os.path.join(query_dir, f"{query_slug}_{timestamp}.csv")
    if items:
        df = pd.DataFrame(items)
        
        # Reorder columns for better readability
        preferred_order = ['title', 'price_value', 'currency', 'condition', 
                          'location', 'date_posted', 'seller_type', 'category',
                          'description', 'url', 'page_number', 'scraped_at']
        
        # Keep only columns that exist
        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in preferred_order]
        df = df[existing_cols + other_cols]
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Save statistics
    if stats:
        stats_file = os.path.join(query_dir, f"statistics_{query_slug}_{timestamp}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Save recommendations
    if recommendations:
        rec_file = os.path.join(query_dir, f"recommendations_{query_slug}_{timestamp}.json")
        with open(rec_file, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)
    
    # Generate comprehensive report
    report_file = generate_comprehensive_report(items, stats, recommendations, query_slug, timestamp)
    
    # Create visualizations
    viz_files = create_price_visualizations(items, search_query, query_dir)
    
    # Save to price tracking database
    price_db.save_price_data(items, search_query)
    
    print(f"\nData saved to directory: {query_dir}")
    print(f"  Raw data (JSON): {json_file}")
    print(f"  CSV export: {csv_file}")
    if stats:
        print(f"  Statistics: {stats_file}")
    if recommendations:
        print(f"  Recommendations: {rec_file}")
    print(f"  Report: {report_file}")
    if viz_files:
        print(f"  Visualizations created in: {os.path.join(query_dir, 'visualizations')}")
    
    return {
        "directory": query_dir,
        "json_file": json_file,
        "csv_file": csv_file,
        "report_file": report_file,
        "visualizations": viz_files
    }

def generate_comprehensive_report(items: List[Dict], stats: Optional[Dict], 
                                recommendations: Optional[Dict], query_slug: str, 
                                timestamp: str) -> str:
    """Generate a comprehensive text report"""
    report_file = os.path.join(DATA_DIR, query_slug, f"report_{query_slug}_{timestamp}.txt")
    
    report_lines = [
        "=" * 80,
        "JIJI ETHIOPIA SEARCH ANALYSIS REPORT",
        "=" * 80,
        "",
        "SEARCH DETAILS:",
        "-" * 40,
        f"Search Query:      {stats.get('search_query', 'Unknown') if stats else 'Unknown'}",
        f"Report Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Items Found: {len(items)}",
        f"Items with Price:  {stats.get('items_with_price', 0) if stats else 0}",
        "",
        "=" * 80,
        "PRICE ANALYSIS",
        "=" * 80,
    ]
    
    if stats and 'average_price' in stats:
        report_lines.extend([
            "",
            "BASIC STATISTICS:",
            "-" * 40,
            f"Average Price:     {stats['average_price']:,.2f} {items[0]['currency'] if items else 'ETB'}",
            f"Median Price:      {stats.get('median_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}",
            f"Minimum Price:     {stats.get('min_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}",
            f"Maximum Price:     {stats.get('max_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}",
            f"Price Range:       {stats.get('price_range', 0):,.2f} {items[0]['currency'] if items else 'ETB'}",
        ])
        
        if 'std_dev' in stats:
            report_lines.append(f"Standard Deviation: {stats['std_dev']:,.2f} {items[0]['currency'] if items else 'ETB'}")
        
        if 'q1' in stats:
            report_lines.extend([
                "",
                "PRICE QUARTILES:",
                "-" * 40,
                f"25th Percentile (Q1): {stats['q1']:,.2f} {items[0]['currency'] if items else 'ETB'}",
                f"75th Percentile (Q3): {stats['q3']:,.2f} {items[0]['currency'] if items else 'ETB'}",
                f"Interquartile Range:  {stats['iqr']:,.2f} {items[0]['currency'] if items else 'ETB'}",
            ])
    
    # Price distribution
    if stats and 'price_distribution' in stats:
        report_lines.extend([
            "",
            "PRICE DISTRIBUTION:",
            "-" * 40,
        ])
        for price_range, count in stats['price_distribution'].items():
            percentage = (count / stats['items_with_price']) * 100
            report_lines.append(f"{price_range:25} {count:3} items ({percentage:.1f}%)")
    
    # Condition distribution
    if stats and 'condition_distribution' in stats:
        report_lines.extend([
            "",
            "CONDITION DISTRIBUTION:",
            "-" * 40,
        ])
        total = sum(stats['condition_distribution'].values())
        for condition, count in stats['condition_distribution'].items():
            percentage = (count / total) * 100
            report_lines.append(f"{condition:15} {count:3} items ({percentage:.1f}%)")
    
    # Top locations
    if stats and 'top_locations' in stats:
        report_lines.extend([
            "",
            "TOP LOCATIONS:",
            "-" * 40,
        ])
        for location, count in stats['top_locations'].items():
            percentage = (count / stats['total_items']) * 100
            report_lines.append(f"{location:30} {count:3} items ({percentage:.1f}%)")
    
    # Recommendations
    if recommendations:
        report_lines.extend([
            "",
            "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
        ])
        
        if 'buying_recommendations' in recommendations:
            buying = recommendations['buying_recommendations']
            report_lines.extend([
                "",
                "BUYING RECOMMENDATIONS:",
                "-" * 40,
                f"Optimal Price Range:    {buying.get('optimal_price_range', 'N/A')}",
                f"Good Deal Threshold:    {buying.get('good_deal_threshold', 'N/A')}",
                f"Premium Price Alert:    {buying.get('premium_price_threshold', 'N/A')}",
                f"Fair Market Value:      {buying.get('fair_market_value', 'N/A')}",
                "",
                "Advice:",
            ])
            for advice in buying.get('advice', []):
                report_lines.append(f"  * {advice}")
        
        if 'selling_recommendations' in recommendations:
            selling = recommendations['selling_recommendations']
            report_lines.extend([
                "",
                "SELLING RECOMMENDATIONS:",
                "-" * 40,
                f"Competitive Price Range: {selling.get('competitive_pricing', 'N/A')}",
                f"Quick Sale Price:        {selling.get('quick_sale_pricing', 'N/A')}",
                f"Premium Pricing:         {selling.get('premium_pricing_conditions', 'N/A')}",
                "",
                "Strategic Tips:",
            ])
            for tip in selling.get('strategic_pricing_tips', []):
                report_lines.append(f"  * {tip}")
    
    # Sample listings
    report_lines.extend([
        "",
        "=" * 80,
        "SAMPLE LISTINGS (First 10)",
        "=" * 80,
    ])
    
    for i, item in enumerate(items[:10], 1):
        price_display = f"{item['price_value']:,.2f} {item['currency']}" if item['price_value'] else "Price N/A"
        report_lines.extend([
            "",
            f"{i}. {item['title'][:80]}{'...' if len(item['title']) > 80 else ''}",
            f"   Price: {price_display}",
            f"   Location: {item.get('location', 'N/A')}",
            f"   Condition: {item.get('condition', 'N/A')}",
            f"   Posted: {item.get('date_posted', 'N/A')}",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "Generated by Jiji Ethiopia Price Analysis System",
        "=" * 80,
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return report_file

# NEW FEATURE: Interactive data exploration
def explore_search_results(items: List[Dict], search_query: str):
    """Interactive exploration of search results"""
    if not items:
        print("No items to explore")
        return
    
    while True:
        print(f"\n" + "=" * 80)
        print(f"DATA EXPLORATION: '{search_query}'")
        print("=" * 80)
        print("\nOptions:")
        print("1. View items sorted by price (low to high)")
        print("2. View items sorted by price (high to low)")
        print("3. Filter by condition")
        print("4. Filter by location")
        print("5. Find best value items")
        print("6. Return to main menu")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            # Sort by price low to high
            priced_items = [item for item in items if item.get('price_value')]
            sorted_items = sorted(priced_items, key=lambda x: x['price_value'])
            
            print(f"\nItems sorted by price (low to high):")
            print("-" * 80)
            for i, item in enumerate(sorted_items[:20], 1):
                price = item['price_value']
                print(f"{i:3}. {price:,.0f} ETB - {item['title'][:60]}...")
                print(f"     Condition: {item.get('condition', 'N/A')}, Location: {item.get('location', 'N/A')}")
        
        elif choice == "2":
            # Sort by price high to low
            priced_items = [item for item in items if item.get('price_value')]
            sorted_items = sorted(priced_items, key=lambda x: x['price_value'], reverse=True)
            
            print(f"\nItems sorted by price (high to low):")
            print("-" * 80)
            for i, item in enumerate(sorted_items[:20], 1):
                price = item['price_value']
                print(f"{i:3}. {price:,.0f} ETB - {item['title'][:60]}...")
                print(f"     Condition: {item.get('condition', 'N/A')}, Location: {item.get('location', 'N/A')}")
        
        elif choice == "3":
            # Filter by condition
            conditions = set(item.get('condition', 'Unknown') for item in items)
            print(f"\nAvailable conditions: {', '.join(conditions)}")
            
            condition_filter = input("Enter condition to filter by: ").strip()
            if condition_filter:
                filtered_items = [item for item in items 
                                if item.get('condition', '').lower() == condition_filter.lower()]
                
                print(f"\nFound {len(filtered_items)} items with condition '{condition_filter}':")
                print("-" * 80)
                for i, item in enumerate(filtered_items[:20], 1):
                    price = item.get('price_value', 'N/A')
                    price_str = f"{price:,.0f} ETB" if isinstance(price, (int, float)) else price
                    print(f"{i:3}. {price_str} - {item['title'][:60]}...")
        
        elif choice == "4":
            # Filter by location
            locations = set(item.get('location', 'Unknown') for item in items)
            print(f"\nTop locations (first 10):")
            for i, loc in enumerate(list(locations)[:10], 1):
                count = sum(1 for item in items if item.get('location') == loc)
                print(f"{i:2}. {loc}: {count} items")
            
            location_filter = input("\nEnter location to filter by (or press Enter to skip): ").strip()
            if location_filter:
                filtered_items = [item for item in items 
                                if location_filter.lower() in item.get('location', '').lower()]
                
                print(f"\nFound {len(filtered_items)} items in '{location_filter}':")
                print("-" * 80)
                for i, item in enumerate(filtered_items[:20], 1):
                    price = item.get('price_value', 'N/A')
                    price_str = f"{price:,.0f} ETB" if isinstance(price, (int, float)) else price
                    print(f"{i:3}. {price_str} - {item['title'][:60]}...")
        
        elif choice == "5":
            # Find best value items (high price score)
            priced_items = [item for item in items if item.get('price_value') and item.get('price_score')]
            if priced_items:
                # Sort by price score (higher is better)
                sorted_items = sorted(priced_items, key=lambda x: x.get('price_score', 0), reverse=True)
                
                print(f"\nBest value items (considering price and other factors):")
                print("-" * 80)
                for i, item in enumerate(sorted_items[:15], 1):
                    price = item['price_value']
                    score = item.get('price_score', 0)
                    print(f"{i:3}. Score: {score:.1f} - {price:,.0f} ETB - {item['title'][:50]}...")
                    print(f"     Condition: {item.get('condition', 'N/A')}, Location: {item.get('location', 'N/A')}")
            else:
                print("No items with price scores available")
        
        elif choice == "6":
            break
        
        else:
            print("Invalid option. Please try again.")

# MAIN EXECUTION
def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("JIJI ETHIOPIA SMART PRICE ANALYSIS SYSTEM")
    print("=" * 80)
    print("Advanced web scraper with price tracking, analysis, and recommendations")
    
    while True:
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("1. Search for items")
        print("2. Explore previous search results")
        print("3. View price trends")
        print("4. Export data")
        print("5. System information")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            # New search
            search_query = input("\nEnter search query (e.g., 'iphone 12', 'toyota yaris'): ").strip()
            
            if not search_query:
                print("Please enter a search query")
                continue
            
            try:
                pages_input = input(f"Pages to scrape (1-{MAX_PAGES}, default=3): ").strip()
                pages = int(pages_input) if pages_input else 3
                pages = max(1, min(pages, MAX_PAGES))
            except ValueError:
                pages = 3
            
            print(f"\nStarting search for '{search_query}' ({pages} pages)...")
            
            # Scrape data
            items = scrape_jiji_search(search_query, pages)
            
            if not items:
                print(f"No items found for '{search_query}'")
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
            
            # Interactive options
            while True:
                print(f"\n" + "=" * 80)
                print(f"OPTIONS FOR '{search_query}'")
                print("=" * 80)
                print("1. Explore data interactively")
                print("2. View detailed report")
                print("3. View price distribution")
                print("4. View location analysis")
                print("5. New search")
                print("6. Return to main menu")
                
                sub_choice = input("\nSelect option (1-6): ").strip()
                
                if sub_choice == "1":
                    explore_search_results(items, search_query)
                
                elif sub_choice == "2":
                    if saved_files and os.path.exists(saved_files['report_file']):
                        with open(saved_files['report_file'], 'r', encoding='utf-8') as f:
                            print(f"\n" + "-" * 80)
                            print("REPORT PREVIEW (first 2000 characters):")
                            print("-" * 80)
                            print(f.read()[:2000] + "...\n(Full report saved to file)")
                    else:
                        print("Report file not found")
                
                elif sub_choice == "3":
                    if stats and 'price_distribution' in stats:
                        print(f"\nPRICE DISTRIBUTION FOR '{search_query}':")
                        print("-" * 40)
                        for price_range, count in stats['price_distribution'].items():
                            percentage = (count / stats['items_with_price']) * 100
                            bar_length = int(percentage / 2)  # Each character represents 2%
                            bar = "#" * bar_length
                            print(f"{price_range:25} {count:3} items | {bar} ({percentage:.1f}%)")
                
                elif sub_choice == "4":
                    if stats and 'top_locations' in stats:
                        print(f"\nTOP LOCATIONS FOR '{search_query}':")
                        print("-" * 40)
                        for location, count in stats['top_locations'].items():
                            percentage = (count / stats['total_items']) * 100
                            print(f"{location:30} {count:3} items ({percentage:.1f}%)")
                
                elif sub_choice == "5":
                    break  # Break inner loop for new search
                
                elif sub_choice == "6":
                    return  # Return to main menu
                
                else:
                    print("Invalid option")
        
        elif choice == "2":
            # Explore previous searches
            print(f"\n" + "=" * 80)
            print("PREVIOUS SEARCH RESULTS")
            print("=" * 80)
            
            if os.path.exists(DATA_DIR):
                subdirs = [d for d in os.listdir(DATA_DIR) 
                          if os.path.isdir(os.path.join(DATA_DIR, d)) and d != 'analysis']
                
                if subdirs:
                    print("\nAvailable searches:")
                    for i, subdir in enumerate(subdirs[-15:], 1):  # Show last 15 searches
                        search_name = subdir.replace('_', ' ').title()
                        search_path = os.path.join(DATA_DIR, subdir)
                        
                        # Count JSON files
                        json_files = [f for f in os.listdir(search_path) 
                                     if f.endswith('.json') and 'stats' not in f and 'recommendations' not in f]
                        
                        if json_files:
                            latest_json = max([os.path.join(search_path, f) for f in json_files], 
                                            key=os.path.getctime)
                            try:
                                with open(latest_json, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    item_count = len(data) if isinstance(data, list) else 0
                                    print(f"{i:3}. {search_name}: {item_count} items")
                            except:
                                print(f"{i:3}. {search_name}")
                    
                    try:
                        search_choice = input(f"\nSelect search to explore (1-{len(subdirs)} or 0 to go back): ").strip()
                        if search_choice == '0':
                            continue
                        
                        search_idx = int(search_choice) - 1
                        if 0 <= search_idx < len(subdirs):
                            selected_dir = subdirs[search_idx]
                            dir_path = os.path.join(DATA_DIR, selected_dir)
                            
                            # Find latest JSON data file
                            json_files = [f for f in os.listdir(dir_path) 
                                         if f.endswith('.json') and 'stats' not in f and 'recommendations' not in f]
                            
                            if json_files:
                                latest_json = max([os.path.join(dir_path, f) for f in json_files], 
                                                key=os.path.getctime)
                                
                                with open(latest_json, 'r', encoding='utf-8') as f:
                                    items = json.load(f)
                                    search_query = items[0].get('search_query', selected_dir.replace('_', ' ')) if items else selected_dir.replace('_', ' ')
                                
                                explore_search_results(items, search_query)
                            else:
                                print("No data files found for this search")
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a valid number")
                else:
                    print("No previous search data found")
            else:
                print("Data directory not found")
        
        elif choice == "3":
            # View price trends
            search_query = input("\nEnter search query to view price trends: ").strip()
            if search_query:
                trends = price_db.get_price_trend(search_query, 30)
                
                if trends:
                    print(f"\nPRICE TRENDS FOR '{search_query}' (Last 30 days):")
                    print("-" * 80)
                    print(f"{'Date':12} {'Avg Price':15} {'Items':10}")
                    print("-" * 80)
                    
                    for trend in trends:
                        date = trend['date']
                        avg_price = trend['average_price']
                        count = trend['count']
                        print(f"{date:12} {avg_price:15,.2f} ETB {count:10}")
                    
                    # Calculate overall trend
                    if len(trends) >= 2:
                        first_avg = trends[0]['average_price']
                        last_avg = trends[-1]['average_price']
                        trend_percent = ((last_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
                        
                        print(f"\nOverall Trend: {'UP' if trend_percent > 0 else 'DOWN'} by {abs(trend_percent):.1f}%")
                else:
                    print(f"No price trend data available for '{search_query}'")
        
        elif choice == "4":
            # Export data
            print(f"\n" + "=" * 80)
            print("DATA EXPORT")
            print("=" * 80)
            
            export_types = ["CSV", "JSON", "Excel", "All Formats"]
            print("\nExport formats:")
            for i, export_type in enumerate(export_types, 1):
                print(f"{i}. {export_type}")
            
            try:
                format_choice = int(input("\nSelect export format (1-4): "))
                if 1 <= format_choice <= 4:
                    selected_format = export_types[format_choice - 1]
                    
                    # Get available searches
                    if os.path.exists(DATA_DIR):
                        subdirs = [d for d in os.listdir(DATA_DIR) 
                                  if os.path.isdir(os.path.join(DATA_DIR, d))]
                        
                        if subdirs:
                            print(f"\nAvailable searches to export:")
                            for i, subdir in enumerate(subdirs, 1):
                                print(f"{i}. {subdir.replace('_', ' ')}")
                            
                            search_choice = int(input(f"\nSelect search to export (1-{len(subdirs)}): "))
                            if 1 <= search_choice <= len(subdirs):
                                selected_dir = subdirs[search_choice - 1]
                                dir_path = os.path.join(DATA_DIR, selected_dir)
                                
                                # Find latest data file
                                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                                json_files = [f for f in os.listdir(dir_path) 
                                            if f.endswith('.json') and 'stats' not in f and 'recommendations' not in f]
                                
                                if csv_files and json_files:
                                    latest_csv = max([os.path.join(dir_path, f) for f in csv_files], 
                                                   key=os.path.getctime)
                                    latest_json = max([os.path.join(dir_path, f) for f in json_files], 
                                                    key=os.path.getctime)
                                    
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    if selected_format in ["CSV", "All Formats"]:
                                        export_csv = os.path.join(EXPORT_DIR, f"export_{selected_dir}_{timestamp}.csv")
                                        import shutil
                                        shutil.copy2(latest_csv, export_csv)
                                        print(f"CSV exported to: {export_csv}")
                                    
                                    if selected_format in ["JSON", "All Formats"]:
                                        export_json = os.path.join(EXPORT_DIR, f"export_{selected_dir}_{timestamp}.json")
                                        shutil.copy2(latest_json, export_json)
                                        print(f"JSON exported to: {export_json}")
                                    
                                    if selected_format in ["Excel", "All Formats"]:
                                        try:
                                            export_excel = os.path.join(EXPORT_DIR, f"export_{selected_dir}_{timestamp}.xlsx")
                                            df = pd.read_csv(latest_csv)
                                            df.to_excel(export_excel, index=False)
                                            print(f"Excel exported to: {export_excel}")
                                        except Exception as e:
                                            print(f"Excel export failed: {e}")
                                    
                                    print(f"\nExport completed successfully")
                                else:
                                    print("No data files found for export")
                            else:
                                print("Invalid selection")
                        else:
                            print("No search data available for export")
                    else:
                        print("Data directory not found")
                else:
                    print("Invalid format selection")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == "5":
            # System information
            print(f"\n" + "=" * 80)
            print("SYSTEM INFORMATION")
            print("=" * 80)
            print(f"Data Directory: {DATA_DIR}")
            print(f"Analysis Directory: {ANALYSIS_DIR}")
            print(f"Export Directory: {EXPORT_DIR}")
            print(f"Maximum Pages per Search: {MAX_PAGES}")
            print(f"Request Delay: {REQUEST_DELAY} seconds")
            
            # Count total searches
            if os.path.exists(DATA_DIR):
                subdirs = [d for d in os.listdir(DATA_DIR) 
                          if os.path.isdir(os.path.join(DATA_DIR, d))]
                print(f"Total Searches Stored: {len(subdirs)}")
            
            # Show disk usage
            try:
                import shutil
                total, used, free = shutil.disk_usage(BASE_DIR)
                print(f"Disk Usage: {used // (2**30)}GB used, {free // (2**30)}GB free")
            except:
                pass
        
        elif choice == "6":
            print(f"\nThank you for using Jiji Ethiopia Price Analysis System")
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")

def display_search_summary(items: List[Dict], stats: Dict, search_query: str):
    """Display a summary of search results"""
    if not items:
        print(f"No items found for '{search_query}'")
        return
    
    print(f"\n" + "=" * 80)
    print(f"SEARCH RESULTS SUMMARY: '{search_query}'")
    print("=" * 80)
    
    print(f"\nOVERVIEW:")
    print(f"  Total Items Found: {len(items)}")
    print(f"  Items with Price: {stats.get('items_with_price', 0)}")
    
    if 'average_price' in stats:
        print(f"\nPRICE ANALYSIS:")
        print(f"  Average Price: {stats['average_price']:,.2f} {items[0]['currency'] if items else 'ETB'}")
        print(f"  Median Price: {stats.get('median_price', 0):,.2f} {items[0]['currency'] if items else 'ETB'}")
        print(f"  Price Range: {stats.get('min_price', 0):,.2f} - {stats.get('max_price', 0):,.2f}")
        
        if 'std_dev' in stats:
            print(f"  Standard Deviation: {stats['std_dev']:,.2f}")
        
        if 'q1' in stats:
            print(f"  25th Percentile: {stats['q1']:,.2f}")
            print(f"  75th Percentile: {stats['q3']:,.2f}")
    
    if 'condition_distribution' in stats:
        print(f"\nCONDITION DISTRIBUTION:")
        for condition, count in stats['condition_distribution'].items():
            percentage = (count / sum(stats['condition_distribution'].values())) * 100
            print(f"  {condition}: {count} items ({percentage:.1f}%)")
    
    if 'top_locations' in stats:
        print(f"\nTOP LOCATIONS:")
        for location, count in list(stats['top_locations'].items())[:3]:
            print(f"  {location}: {count} items")
    
    # Display sample items
    print(f"\nSAMPLE LISTINGS:")
    for i, item in enumerate(items[:5], 1):
        price_display = f"{item['price_value']:,.2f} {item['currency']}" if item['price_value'] else "Price N/A"
        print(f"  {i}. {item['title'][:60]}...")
        print(f"     {price_display} | {item.get('location', 'N/A')} | {item.get('condition', 'N/A')}")

if __name__ == "__main__":
    main()