import requests
from bs4 import BeautifulSoup
import csv
import json
import pandas as pd # type: ignore
import numpy as np
from datetime import datetime, timedelta
import os
import time
import re
from collections import defaultdict
import statistics
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

#CONFIGURATION


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
PAGES_TO_SCRAPE = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# PRICE PARSING 
def parse_price(price_text):
    """Convert price text to numeric value"""
    try:
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.]', '', price_text)
        if cleaned:
            return float(cleaned)
    except:
        pass
    return None

def parse_date(date_str):
    """Parse date string to datetime"""
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return datetime.now()

# CATEGORY CLASSIFICATION 

def categorize_item(title):
    """Categorize items based on keywords"""
    title_lower = title.lower()
    
    categories = {
        'Electronics': ['phone', 'iphone', 'samsung', 'laptop', 'tablet', 'computer', 
                       'tv', 'television', 'camera', 'headphone', 'earphone'],
        'Vehicles': ['car', 'vehicle', 'bmw', 'toyota', 'mercedes', 'motor', 'bike'],
        'Real Estate': ['house', 'apartment', 'land', 'property', 'rent', 'for rent'],
        'Fashion': ['shoe', 'clothes', 'dress', 'shirt', 'jacket', 'watch'],
        'Home Appliances': ['fridge', 'refrigerator', 'oven', 'washing machine', 'microwave'],
        'Furniture': ['bed', 'sofa', 'table', 'chair', 'wardrobe'],
        'Jobs': ['job', 'vacancy', 'employment', 'hire', 'recruit'],
        'Services': ['service', 'repair', 'maintenance', 'delivery']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in title_lower:
                return category
    
    return 'Other'

# SIMILARITY DETECTION 

def find_similar_items(target_title, items_list, threshold=0.3):
    """Find similar items using keyword matching"""
    target_words = set(target_title.lower().split())
    similar_items = []
    
    for item in items_list:
        item_words = set(item['title'].lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(target_words.intersection(item_words))
        union = len(target_words.union(item_words))
        
        if union > 0:
            similarity = intersection / union
            if similarity >= threshold:
                similar_items.append({
                    'item': item,
                    'similarity_score': round(similarity, 2)
                })
    
    # Sort by similarity score
    similar_items.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar_items

def advanced_similarity_search(target_item, all_items):
    """Advanced similarity search with multiple criteria"""
    target_title = target_item['title'].lower()
    target_category = categorize_item(target_item['title'])
    
    similar = []
    
    for item in all_items:
        if item['id'] == target_item.get('id'):
            continue
            
        # Multiple similarity checks
        score = 0
        
        # Title similarity
        title_sim = len(set(target_title.split()) & set(item['title'].lower().split())) / \
                   max(len(set(target_title.split())), 1)
        score += title_sim * 0.6
        
        # Category match
        item_category = categorize_item(item['title'])
        if target_category == item_category:
            score += 0.3
        
        # Price range similarity
        if target_item.get('price_birr') and item.get('price_birr'):
            price_diff = abs(target_item['price_birr'] - item['price_birr'])
            price_sim = 1 / (1 + price_diff / 1000)  # Normalize
            score += price_sim * 0.1
        
        if score >= 0.4:  # Threshold
            similar.append({
                'item': item,
                'similarity_score': round(score, 3),
                'match_type': 'title' if title_sim > 0.3 else 'category'
            })
    
    return sorted(similar, key=lambda x: x['similarity_score'], reverse=True)

# STATISTICAL ANALYSIS 

def calculate_item_statistics(items):
    """Calculate statistics for a list of items"""
    if not items:
        return {}
    
    prices = [item['price_birr'] for item in items if item['price_birr'] is not None]
    
    if not prices:
        return {"count": len(items), "valid_prices": 0}
    
    stats = {
        'count': len(items),
        'valid_prices': len(prices),
        'average': round(statistics.mean(prices), 2),
        'median': round(statistics.median(prices), 2),
        'min': min(prices),
        'max': max(prices),
        'range': max(prices) - min(prices),
        'std_dev': round(statistics.stdev(prices), 2) if len(prices) > 1 else 0,
        'price_per_item': round(statistics.mean(prices), 2)
    }
    
    # Calculate quartiles
    if len(prices) >= 4:
        q1 = np.percentile(prices, 25)
        q3 = np.percentile(prices, 75)
        stats['q1'] = round(q1, 2)
        stats['q3'] = round(q3, 2)
        stats['iqr'] = round(q3 - q1, 2)
    
    return stats

def analyze_by_category(all_items):
    """Analyze items by category"""
    categorized = defaultdict(list)
    
    for item in all_items:
        category = categorize_item(item['title'])
        categorized[category].append(item)
    
    category_stats = {}
    for category, items in categorized.items():
        category_stats[category] = {
            'count': len(items),
            'stats': calculate_item_statistics(items)
        }
    
    return category_stats

# PRICE PREDICTION 

def predict_future_prices(historical_data, days_ahead=7):
    """Predict future prices using linear regression"""
    if len(historical_data) < 3:
        return {"error": "Insufficient data for prediction"}
    
    # Prepare data
    dates = []
    prices = []
    
    for item in historical_data:
        if item.get('price_birr') and item.get('scraped_at'):
            try:
                date = parse_date(item['scraped_at'])
                dates.append(date.timestamp())  # Convert to numeric
                prices.append(item['price_birr'])
            except:
                continue
    
    if len(dates) < 3:
        return {"error": "Not enough valid price points"}
    
    # Reshape for sklearn
    X = np.array(dates).reshape(-1, 1)
    y = np.array(prices)
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future dates
    last_date = datetime.fromtimestamp(max(dates))
    predictions = []
    
    for i in range(1, days_ahead + 1):
        future_date = last_date + timedelta(days=i)
        future_timestamp = future_date.timestamp()
        predicted_price = model.predict([[future_timestamp]])[0]
        
        predictions.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'predicted_price': round(predicted_price, 2),
            'confidence': round(model.score(X, y), 3) if len(X) > 1 else 0
        })
    
    # Calculate trend
    trend = "increasing" if model.coef_[0] > 0 else "decreasing"
    trend_strength = abs(model.coef_[0])
    
    return {
        'predictions': predictions,
        'trend': trend,
        'trend_strength': round(trend_strength, 4),
        'model_score': round(model.score(X, y), 3) if len(X) > 1 else 0,
        'data_points': len(dates)
    }

def seasonal_price_prediction(items):
    """Simple seasonal/trend analysis"""
    if len(items) < 5:
        return {"error": "Need at least 5 items for trend analysis"}
    
    # Group by month if we have historical data
    monthly_avg = defaultdict(list)
    
    for item in items:
        if item.get('price_birr') and item.get('scraped_at'):
            try:
                date = parse_date(item['scraped_at'])
                month_key = date.strftime('%Y-%m')
                monthly_avg[month_key].append(item['price_birr'])
            except:
                continue
    
    if not monthly_avg:
        return {"error": "No valid price data with dates"}
    
    # Calculate monthly averages
    monthly_stats = {}
    for month, prices in monthly_avg.items():
        monthly_stats[month] = {
            'avg_price': round(statistics.mean(prices), 2),
            'count': len(prices),
            'min': min(prices),
            'max': max(prices)
        }
    
    # Simple trend calculation
    months = sorted(monthly_stats.keys())
    if len(months) >= 2:
        first_avg = monthly_stats[months[0]]['avg_price']
        last_avg = monthly_stats[months[-1]]['avg_price']
        trend_percentage = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        # Predict next month
        predicted_next = last_avg * (1 + trend_percentage/100/len(months))
        
        return {
            'monthly_stats': monthly_stats,
            'trend': 'up' if trend_percentage > 0 else 'down',
            'trend_percentage': round(trend_percentage, 2),
            'predicted_next_month': round(predicted_next, 2),
            'confidence': 'high' if len(months) >= 3 else 'low'
        }
    
    return {"error": "Insufficient monthly data"}

# WEB SCRAPER 

def fetch_page(url):
    """Fetch page with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Fetching page (attempt {attempt})...")
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            time.sleep(REQUEST_DELAY)
    return None

def scrape_jiji_listings():
    """Scrape multiple pages from Jiji"""
    all_data = []
    
    print("\n" + "="*60)
    print("STARTING JIJI WEB SCRAPER WITH ANALYSIS")
    print("="*60)
    
    for page in range(1, PAGES_TO_SCRAPE + 1):
        url = f"https://jiji.com.et/?page={page}"
        print(f"\n[Page {page}] Scraping: {url}")
        
        html = fetch_page(url)
        if not html:
            print(f"  Failed to fetch page {page}")
            continue
        
        soup = BeautifulSoup(html, "lxml")
        
        # Find items
        titles = soup.find_all(
            "div",
            class_="b-advert-title-inner qa-advert-title b-advert-title-inner--div"
        )
        prices = soup.find_all("div", class_="qa-advert-price")
        
        print(f"  Found {len(titles)} items")
        
        for i, (t, p) in enumerate(zip(titles, prices), 1):
            title_text = t.text.strip()
            price_text = p.text.strip()
            price_value = parse_price(price_text)
            
            item = {
                "id": len(all_data) + 1,
                "title": title_text,
                "price_text": price_text,
                "price_birr": price_value,
                "category": categorize_item(title_text),
                "scraped_at": datetime.now().isoformat(),
                "source": "Jiji Ethiopia",
                "page": page,
                "item_number": i
            }
            
            all_data.append(item)
            
            # Show first few items
            if i <= 3:
                price_display = f"{price_value:,.2f} ETB" if price_value else price_text
                print(f"    {i}. {title_text[:50]}... - {price_display}")
        
        if len(titles) > 3:
            print(f"    ... and {len(titles)-3} more items")
        
        time.sleep(REQUEST_DELAY)
    
    return all_data

# DATA SAVING 

def save_data_with_analysis(data):
    """Save data with comprehensive analysis"""
    if not data:
        print("\n[ERROR] No data to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    json_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.json")
    csv_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.csv")
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    if data:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    # Create samples
    create_sample_files(data)
    
    # Perform analysis
    analysis_results = perform_comprehensive_analysis(data)
    
    # Save analysis
    analysis_file = os.path.join(ANALYSIS_DIR, f"analysis_{timestamp}.json")
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate report
    generate_analysis_report(analysis_results, timestamp)
    
    print(f"\n[SUCCESS] Data saved:")
    print(f"  Raw data: {json_file}")
    print(f"  Analysis: {analysis_file}")
    
    return analysis_results

def perform_comprehensive_analysis(data):
    """Perform all analysis on scraped data"""
    print("\n" + "="*60)
    print("PERFORMING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # 1. Overall statistics
    overall_stats = calculate_item_statistics(data)
    print("\n1. OVERALL STATISTICS:")
    for key, value in overall_stats.items():
        print(f"   {key}: {value}")
    
    # 2. Category analysis
    category_analysis = analyze_by_category(data)
    print("\n2. CATEGORY ANALYSIS:")
    for category, info in category_analysis.items():
        if info['count'] > 0:
            print(f"   {category}: {info['count']} items, "
                  f"Avg: {info['stats'].get('average', 'N/A'):,.2f} ETB")
    
    # 3. Price prediction
    print("\n3. PRICE PREDICTION:")
    prediction = predict_future_prices(data)
    if 'error' not in prediction:
        print(f"   Trend: {prediction['trend']} (strength: {prediction['trend_strength']})")
        print(f"   Model Score: {prediction['model_score']}")
        for pred in prediction.get('predictions', [])[:3]:
            print(f"   {pred['date']}: ~{pred['predicted_price']:,.2f} ETB")
    else:
        print(f"   {prediction['error']}")
    
    # 4. Find similar items example
    if data:
        sample_item = data[0]
        similar_items = advanced_similarity_search(sample_item, data[:20])
        print(f"\n4. SIMILARITY SEARCH EXAMPLE:")
        print(f"   Sample: '{sample_item['title'][:50]}...'")
        print(f"   Found {len(similar_items)} similar items")
        if similar_items:
            best_match = similar_items[0]
            print(f"   Best match: '{best_match['item']['title'][:50]}...'")
            print(f"   Similarity: {best_match['similarity_score']}")
            print(f"   Price: {best_match['item'].get('price_birr', 'N/A'):,.2f} ETB")
    
    # 5. Seasonal analysis
    seasonal = seasonal_price_prediction(data)
    if 'error' not in seasonal:
        print(f"\n5. SEASONAL ANALYSIS:")
        print(f"   Trend: {seasonal['trend']} ({seasonal['trend_percentage']}%)")
        print(f"   Next month prediction: {seasonal['predicted_next_month']:,.2f} ETB")
    
    return {
        'overall_stats': overall_stats,
        'category_analysis': category_analysis,
        'price_prediction': prediction,
        'seasonal_analysis': seasonal,
        'timestamp': datetime.now().isoformat(),
        'total_items': len(data)
    }

def generate_analysis_report(analysis, timestamp):
    """Generate readable analysis report"""
    report_file = os.path.join(ANALYSIS_DIR, f"report_{timestamp}.txt")
    
    report = f"""
JIJI ETHIOPIA MARKET ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

OVERVIEW:
- Total Items Analyzed: {analysis['total_items']}
- Analysis Timestamp: {analysis['timestamp']}

OVERALL STATISTICS:
{'-'*40}
"""
    
    # Add overall stats
    for key, value in analysis['overall_stats'].items():
        if isinstance(value, (int, float)):
            if 'price' in key.lower() or 'average' in key.lower() or key in ['min', 'max', 'median']:
                report += f"{key.replace('_', ' ').title()}: {value:,.2f} ETB\n"
            else:
                report += f"{key.replace('_', ' ').title()}: {value}\n"
    
    # Add category analysis
    report += f"\nCATEGORY DISTRIBUTION:\n{'-'*40}\n"
    for category, info in analysis['category_analysis'].items():
        if info['count'] > 0:
            avg = info['stats'].get('average', 0)
            report += f"{category}: {info['count']} items | Avg: {avg:,.2f} ETB\n"
    
    # Add price predictions
    if 'error' not in analysis['price_prediction']:
        report += f"\nPRICE PREDICTIONS (Next 7 days):\n{'-'*40}\n"
        report += f"Overall Trend: {analysis['price_prediction']['trend']}\n"
        report += f"Trend Strength: {analysis['price_prediction']['trend_strength']}\n"
        report += f"Model Confidence: {analysis['price_prediction']['model_score']}\n\n"
        
        for pred in analysis['price_prediction'].get('predictions', []):
            report += f"{pred['date']}: {pred['predicted_price']:,.2f} ETB\n"
    
    # Add seasonal analysis
    if 'error' not in analysis['seasonal_analysis']:
        report += f"\nSEASONAL TREND ANALYSIS:\n{'-'*40}\n"
        report += f"Trend Direction: {analysis['seasonal_analysis']['trend']}\n"
        report += f"Trend Percentage: {analysis['seasonal_analysis']['trend_percentage']}%\n"
        report += f"Next Month Prediction: {analysis['seasonal_analysis']['predicted_next_month']:,.2f} ETB\n"
    
    report += f"\n{'='*60}\nEND OF REPORT\n"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n[REPORT] Analysis report saved: {report_file}")
    return report_file

def create_sample_files(data):
    """Create sample files for GitHub"""
    if not data:
        return
    
    sample = data[:3]
    
    with open(os.path.join(DATA_DIR, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(DATA_DIR, "sample_data.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(sample)

# INTERACTIVE FEATURES 

def interactive_search(data):
    """Interactive search for similar items"""
    if not data:
        print("\n[ERROR] No data available for search")
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE SIMILARITY SEARCH")
    print("="*60)
    
    # Show some sample items
    print("\nSample items in database:")
    for i, item in enumerate(data[:10], 1):
        print(f"{i}. {item['title'][:60]}... - {item.get('price_birr', 'N/A'):,.2f} ETB")
    
    # Get search query
    query = input("\nEnter search term or item number (1-10): ").strip()
    
    if query.isdigit() and 1 <= int(query) <= min(10, len(data)):
        target_item = data[int(query)-1]
    else:
        # Find best match for search term
        best_match = None
        best_score = 0
        
        for item in data[:50]:  # Limit search to first 50 items
            similarity = len(set(query.lower().split()) & set(item['title'].lower().split()))
            if similarity > best_score:
                best_score = similarity
                best_match = item
        
        target_item = best_match or data[0]
    
    print(f"\nSearching for items similar to:")
    print(f"  '{target_item['title']}'")
    print(f"  Price: {target_item.get('price_birr', 'N/A'):,.2f} ETB")
    print(f"  Category: {target_item.get('category', 'Unknown')}")
    
    # Find similar items
    similar_items = advanced_similarity_search(target_item, data)
    
    if similar_items:
        print(f"\nFound {len(similar_items)} similar items:")
        print("-" * 60)
        
        for i, sim in enumerate(similar_items[:10], 1):
            item = sim['item']
            print(f"{i}. Similarity: {sim['similarity_score']}")
            print(f"   Title: {item['title'][:60]}...")
            print(f"   Price: {item.get('price_birr', 'N/A'):,.2f} ETB")
            print(f"   Category: {item.get('category', 'Unknown')}")
            print()
        
        # Calculate average of similar items
        similar_prices = [sim['item'].get('price_birr') for sim in similar_items 
                         if sim['item'].get('price_birr') is not None]
        
        if similar_prices:
            avg_price = statistics.mean(similar_prices)
            print(f"Average price of similar items: {avg_price:,.2f} ETB")
            print(f"Target item price: {target_item.get('price_birr', 'N/A'):,.2f} ETB")
            
            if target_item.get('price_birr'):
                diff = target_item['price_birr'] - avg_price
                diff_percent = (diff / avg_price * 100) if avg_price > 0 else 0
                
                if diff > 0:
                    print(f"Target is {diff_percent:.1f}% MORE expensive than average")
                else:
                    print(f"Target is {abs(diff_percent):.1f}% LESS expensive than average")
    else:
        print("\nNo similar items found.")

def price_analysis_dashboard(data):
    """Display price analysis dashboard"""
    if not data:
        print("\n[ERROR] No data available for analysis")
        return
    
    print("\n" + "="*60)
    print("PRICE ANALYSIS DASHBOARD")
    print("="*60)
    
    # Overall statistics
    stats = calculate_item_statistics(data)
    
    print(f"\n📊 OVERALL MARKET ANALYSIS")
    print(f"   Total Items: {stats.get('count', 0)}")
    print(f"   Average Price: {stats.get('average', 0):,.2f} ETB")
    print(f"   Price Range: {stats.get('min', 0):,.2f} - {stats.get('max', 0):,.2f} ETB")
    print(f"   Standard Deviation: {stats.get('std_dev', 0):,.2f} ETB")
    
    # Category breakdown
    categories = defaultdict(list)
    for item in data:
        categories[item.get('category', 'Unknown')].append(item)
    
    print(f"\n📈 CATEGORY BREAKDOWN")
    for category, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        cat_stats = calculate_item_statistics(items)
        print(f"   {category}: {len(items)} items | "
              f"Avg: {cat_stats.get('average', 0):,.2f} ETB")
    
    # Price distribution
    prices = [item.get('price_birr') for item in data if item.get('price_birr')]
    if prices:
        print(f"\n📉 PRICE DISTRIBUTION")
        bins = [0, 1000, 5000, 10000, 50000, 100000, float('inf')]
        labels = ['<1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '>100K']
        
        for i in range(len(bins)-1):
            count = sum(1 for p in prices if bins[i] <= p < bins[i+1])
            if count > 0:
                percentage = (count / len(prices)) * 100
                print(f"   {labels[i]}: {count} items ({percentage:.1f}%)")
    
    # Prediction
    print(f"\n🔮 PRICE PREDICTION")
    prediction = predict_future_prices(data[:50])  # Use recent items
    
    if 'error' not in prediction:
        trend_icon = "📈" if prediction['trend'] == 'increasing' else "📉"
        print(f"   Trend: {trend_icon} {prediction['trend'].upper()}")
        print(f"   Confidence: {prediction.get('model_score', 0):.1%}")
        
        if prediction.get('predictions'):
            next_pred = prediction['predictions'][0]
            print(f"   Tomorrow's prediction: {next_pred['predicted_price']:,.2f} ETB")

# MAIN EXECUTION 

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("JIJI ETHIOPIA WEB SCRAPER WITH ADVANCED ANALYTICS")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Pages to scrape: {PAGES_TO_SCRAPE}")
    print(f"  Request delay: {REQUEST_DELAY}s")
    print(f"  Data directory: {DATA_DIR}")
    print("="*60)
    
    # Scrape data
    data = scrape_jiji_listings()
    
    if not data:
        print("\n[ERROR] No data was scraped. Exiting.")
        return
    
    print(f"\n[SUCCESS] Scraped {len(data)} items")
    
    # Save data with analysis
    analysis = save_data_with_analysis(data)
    
    # Interactive features
    print("\n" + "="*60)
    print("INTERACTIVE FEATURES")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Search for similar items")
        print("2. View price analysis dashboard")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            interactive_search(data)
        elif choice == "2":
            price_analysis_dashboard(data)
        elif choice == "3":
            print("\nExiting. Goodbye!")
            break
        else:
            print("\nInvalid option. Please try again.")

if __name__ == "__main__":
    main()