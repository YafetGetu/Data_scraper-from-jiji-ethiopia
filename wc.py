import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os
import time

# Base directory (where wc.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Scraper settings
REQUEST_DELAY = 2      # seconds
MAX_RETRIES = 3
TIMEOUT = 10
PAGES_TO_SCRAPE = 3    # number of pages

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def fetch_page(url):
    """Fetch page with retry logic"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            time.sleep(REQUEST_DELAY)
    return None

def parse_price(price_text):
    """Convert '45,000 Birr' -> 45000"""
    try:
        return float(price_text.replace("Birr", "").replace(",", "").strip())
    except:
        return None

def scrape_jiji_listings():
    all_data = []

    print("\nStarting scraping...")
    print("-" * 50)

    for page in range(1, PAGES_TO_SCRAPE + 1):
        url = f"https://jiji.com.et/?page={page}"
        print(f"Scraping page {page}")

        html = fetch_page(url)
        if not html:
            continue

        soup = BeautifulSoup(html, "lxml")

        titles = soup.find_all(
            "div",
            class_="b-advert-title-inner qa-advert-title b-advert-title-inner--div"
        )
        prices = soup.find_all("div", class_="qa-advert-price")

        for t, p in zip(titles, prices):
            price_text = p.text.strip()

            item = {
                "title": t.text.strip(),
                "price_text": price_text,
                "price_birr": parse_price(price_text),
                "scraped_at": datetime.now().isoformat(),
                "source": "Jiji Ethiopia",
                "page": page
            }

            all_data.append(item)
            print(f"{item['title']} - {item['price_text']}")

        time.sleep(REQUEST_DELAY)

    save_data(all_data)
    return all_data

def save_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.json")
    csv_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.csv")
    log_file = os.path.join(DATA_DIR, "daily_scraping_log.txt")

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if data:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - Scraped {len(data)} items\n")

    create_sample_files(data)

    print("\nData saved in /data folder")

def create_sample_files(data):
    if not data:
        return

    sample = data[:3]

    with open(os.path.join(DATA_DIR, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "sample_data.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(sample)

def main():
    print("Jiji Web Scraper")
    print(f"Output folder: {DATA_DIR}")
    print(f"Pages to scrape: {PAGES_TO_SCRAPE}")
    scrape_jiji_listings()
    print("Scraping completed")

if __name__ == "__main__":
    main()
