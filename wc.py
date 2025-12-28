import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os

# Base directory (where wc.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory inside project
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_jiji_listings():
    url = "https://jiji.com.et/"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        titles = soup.find_all(
            "div",
            class_="b-advert-title-inner qa-advert-title b-advert-title-inner--div"
        )
        prices = soup.find_all("div", class_="qa-advert-price")

        data = []

        print("\nScraping Jiji listings")
        print("-" * 50)

        for t, p in zip(titles, prices):
            item = {
                "title": t.text.strip(),
                "price": p.text.strip(),
                "scraped_at": datetime.now().isoformat(),
                "source": "Jiji Ethiopia"
            }
            data.append(item)
            print(f"{item['title']} - {item['price']}")

        save_data(data)
        return data

    except Exception as e:
        print("Error:", e)
        return []

def save_data(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.json")
    csv_file = os.path.join(DATA_DIR, f"jiji_data_{timestamp}.csv")
    log_file = os.path.join(DATA_DIR, "daily_scraping_log.txt")

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Save CSV
    if data:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    # Append log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - Scraped {len(data)} items\n")

    create_sample_files(data)

    print("\nData saved in /data folder")

def create_sample_files(data):
    if not data:
        return

    sample = data[:3]

    sample_json = os.path.join(DATA_DIR, "sample_data.json")
    sample_csv = os.path.join(DATA_DIR, "sample_data.csv")

    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(sample)

def main():
    print("Starting Jiji Web Scraper")
    print(f"Output folder: {DATA_DIR}")
    scrape_jiji_listings()
    print("Scraping completed")

if __name__ == "__main__":
    main()
