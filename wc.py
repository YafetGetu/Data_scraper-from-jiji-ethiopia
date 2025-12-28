import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime
import os

# Directory where wc.py is located (portable, GitHub-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

    json_file = os.path.join(BASE_DIR, f"jiji_data_{timestamp}.json")
    csv_file = os.path.join(BASE_DIR, f"jiji_data_{timestamp}.csv")
    log_file = os.path.join(BASE_DIR, "daily_scraping_log.txt")

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

def create_sample_files(data):
    if not data:
        return

    sample = data[:3]

    with open(os.path.join(BASE_DIR, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    with open(os.path.join(BASE_DIR, "sample_data.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(sample)

def main():
    print("Starting Jiji Web Scraper")
    scrape_jiji_listings()
    print("Done")

if __name__ == "__main__":
    main()
