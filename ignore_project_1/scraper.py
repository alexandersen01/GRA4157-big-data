import requests
import json
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import queue
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GUARDIAN_KEY")
BASE_URL = "https://content.guardianapis.com/search"


class RateLimiter:
    def __init__(self, max_calls_per_second=12):
        self.max_calls = max_calls_per_second
        self.calls = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls older than 1 second
            while self.calls and self.calls[0] <= now - 1.0:
                self.calls.popleft()

            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = 1.0 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up old calls again after sleeping
                    now = time.time()
                    while self.calls and self.calls[0] <= now - 1.0:
                        self.calls.popleft()

            # Record this call
            self.calls.append(now)


rate_limiter = RateLimiter(12)  # 12 requests per second
all_articles = []
articles_lock = threading.Lock()


def fetch_page(topic, page_num, start_date, end_date, total_pages_found):
    """Fetch a single page of results"""
    rate_limiter.wait_if_needed()

    params = {
        "api-key": API_KEY,
        "q": topic,
        "from-date": start_date,
        "to-date": end_date,
        "page-size": 50,
        "page": page_num,
        "show-fields": "headline,byline,firstPublicationDate,body",
        "order-by": "newest",
    }

    try:
        print(f"Fetching page {page_num}...")
        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()

        if data["response"]["status"] == "ok":
            articles = data["response"]["results"]

            # Update total pages on first successful request
            if total_pages_found[0] == 0:
                total_pages_found[0] = data["response"]["pages"]
                print(f"Total pages to fetch: {total_pages_found[0]}")

            # Thread-safe addition to results
            with articles_lock:
                all_articles.extend(articles)

            return {
                "page": page_num,
                "articles": len(articles),
                "success": True,
                "total_pages": data["response"]["pages"],
            }
        else:
            print(f"Error on page {page_num}: {data}")
            return {"page": page_num, "success": False, "error": data}

    except Exception as e:
        print(f"Exception on page {page_num}: {str(e)}")
        return {"page": page_num, "success": False, "error": str(e)}


def get_all_articles_threaded(topic, start_date, end_date, max_workers=12):
    """
    Fetch all articles using multi-threading with rate limiting
    """
    total_pages_found = [0]  # Use list for mutable reference

    # First, get the first page to determine total pages
    print("Getting first page to determine total pages...")
    first_result = fetch_page(topic, 1, start_date, end_date, total_pages_found)

    if not first_result["success"]:
        print("Failed to get first page. Aborting.")
        return []

    total_pages = total_pages_found[0]
    print(f"Found {total_pages} total pages to process")

    if total_pages <= 1:
        return all_articles

    # Create page numbers for remaining pages (we already got page 1)
    remaining_pages = list(range(2, total_pages + 1))

    # Process remaining pages with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page requests
        future_to_page = {
            executor.submit(
                fetch_page, topic, page, start_date, end_date, total_pages_found
            ): page
            for page in remaining_pages
        }

        # Process completed requests
        completed = 0
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    print(
                        f"✓ Page {result['page']} complete ({result['articles']} articles) - Progress: {completed + 1}/{total_pages}"
                    )
                else:
                    print(f"✗ Page {page_num} failed")
            except Exception as e:
                print(f"✗ Page {page_num} generated exception: {e}")

    return all_articles


def get_data_by_year_ranges(topic, years_back=20, max_workers=12):
    """
    Break down the request by year ranges to avoid overwhelming single requests
    """
    end_date = datetime.now()

    all_results = []

    # Process year by year for better control
    for year_offset in range(years_back):
        year_end = end_date - timedelta(days=year_offset * 365)
        year_start = end_date - timedelta(days=(year_offset + 1) * 365)

        start_date_str = year_start.strftime("%Y-%m-%d")
        end_date_str = year_end.strftime("%Y-%m-%d")

        print(
            f"\n=== Processing {year_start.year} ({start_date_str} to {end_date_str}) ==="
        )

        # Clear the global articles list for this year
        global all_articles
        all_articles = []

        year_articles = get_all_articles_threaded(
            topic, start_date_str, end_date_str, max_workers=max_workers
        )

        print(f"Year {year_start.year}: {len(year_articles)} articles collected")
        all_results.extend(year_articles)

        # Optional: save yearly results incrementally
        if year_articles:
            import pandas as pd

            df_year = pd.DataFrame(
                [
                    {
                        "id": article["id"],
                        "headline": article["fields"].get("headline", ""),
                        "date": article["webPublicationDate"],
                        "url": article["webUrl"],
                        "section": article["sectionName"],
                    }
                    for article in year_articles
                ]
            )

            df_year.to_csv(f"guardian_immigration_{year_start.year}.csv", index=False)

    return all_results


# Usage
if __name__ == "__main__":
    print("Starting multi-threaded Guardian API collection...")
    print("Rate limited to 12 requests per second")
    TOPIC = "immigration"

    start_time = time.time()

    # Option 1: Get all 20 years in one go (may take a while)
    # all_articles = get_all_articles_threaded(
    #     start_date=(datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d'),
    #     end_date=datetime.now().strftime('%Y-%m-%d'),
    #     max_workers=12
    # )

    # Option 2: Process year by year (recommended for large datasets)
    all_articles = get_data_by_year_ranges(TOPIC, years_back=20, max_workers=6)

    end_time = time.time()

    print(f"\n=== COMPLETE ===")
    print(f"Total articles collected: {len(all_articles)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(
        f"Average rate: {len(all_articles) / (end_time - start_time):.2f} articles/second"
    )

    # Save final combined results
    if all_articles:
        import pandas as pd

        df_final = pd.DataFrame(
            [
                {
                    "id": article["id"],
                    "headline": article["fields"].get("headline", ""),
                    "date": article["webPublicationDate"],
                    "url": article["webUrl"],
                    "section": article["sectionName"],
                }
                for article in all_articles
            ]
        )

        df_final.to_csv("guardian_immigration_all_years.csv", index=False)
        print("Saved to guardian_immigration_all_years.csv")
