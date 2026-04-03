import json
import sys, os
import subprocess
from dotenv import load_dotenv

import send_mail
from source.scrape import BookCollector



load_dotenv()        # Load .env file
FILE_PATH = "source/books_output.json"
TEMP_NEW_FILE = "source/new_books.json"

API_URL = "https://booksdashboard-production.up.railway.app/api/all_filtered_results/?min_price=0&max_price=2000&group_by=isbn"


def run_scraper():
    collector = BookCollector(API_URL, FILE_PATH)
    new_books = collector.run()

    # Save ONLY new records for YOLO
    with open(TEMP_NEW_FILE, "w", encoding="utf-8") as f:
        json.dump(new_books, f, indent=4)

    return len(new_books)


def run_yolo():
    print("\n🔥 Starting YOLO on NEW data...\n")
    subprocess.run([sys.executable, "yolo_test.py", TEMP_NEW_FILE])


if __name__ == "__main__":
    print("🚀 PIPELINE STARTED\n")

    new_count = run_scraper()

    # if new_count == 0:
    #     print("✅ No new data — YOLO skipped")
    # else:
    run_yolo()

    # Send mail after YOLO
    send_mail.send_photocopy_email(
        sender_email= os.getenv("SENDER_EMAIL"),
        sender_password = os.getenv("SENDER_PASSWORD"),
        receiver_email = os.getenv("RECEIVER_EMAIL")
    )

    print("\n✅ PIPELINE FINISHED")