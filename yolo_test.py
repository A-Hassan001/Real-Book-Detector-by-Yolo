import os
import json
import time
import requests
from ultralytics import YOLO

# ----------------------- CONFIG -----------------------
PROCESSED_FILE = "source/processed_urls.txt"
INPUT_JSON = "source/books_output.json"
OUTPUT_DIR = "output_results"

MODEL_PATH = r"C:\office work\Books Identifier\runs\classify\train9\weights\best.pt"

# Output files for each class (must match model names)
PATHS = {
    "Photocopy_Pdf": os.path.join(OUTPUT_DIR, "photocopy_pdf.txt"),
    "Real_Book": os.path.join(OUTPUT_DIR, "real_book.txt"),
    "Unknown": os.path.join(OUTPUT_DIR, "Unknown.txt")
}

# -------------------- HELPERS ------------------------

def load_processed_urls():
    """Load already processed book URLs from txt"""
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed_url(book_url):
    """Append a processed URL to the txt file"""
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(book_url + "\n")

def load_books():
    """Load books JSON"""
    if not os.path.exists(INPUT_JSON):
        print(f"❌ JSON file not found: {INPUT_JSON}")
        return []
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def download_image(url, idx):
    """Download image from URL to a temp file"""
    img_name = f"temp_{idx}.webp"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Image {url} not found!")
            return None
        with open(img_name, "wb") as f:
            f.write(response.content)
        return img_name
    except Exception as e:
        print(f"Skipping download failed: {e}")
        return None

def cleanup_image(img_name):
    """Remove temp image file"""
    if os.path.exists(img_name):
        try:
            time.sleep(0.2)
            os.remove(img_name)
        except PermissionError:
            print(f"⚠️ Could not delete {img_name}, skipping...")

def predict_image(model, img_name):
    """Run YOLO prediction and return class name and confidence"""
    results = model(img_name, verbose=False)
    result = results[0]
    raw_pred = model.names[result.probs.top1]
    pred_class = raw_pred.strip().title()
    conf = result.probs.top1conf.item()
    if pred_class == "Photocopy/Pdf":
        pred_class = "Photocopy_Pdf"  # match output file key
    return pred_class, conf

# -------------------- MAIN ---------------------------

def main():
    # Load processed URLs
    processed_urls = load_processed_urls()
    print(f"Already processed URLs: {len(processed_urls)}")

    # Load books
    books = load_books()
    if not books:
        print("❌ No books to process")
        return

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open output files
    files = {cls: open(path, "a", encoding="utf-8") for cls, path in PATHS.items()}

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    print(f"Model Classes: {model.names}")

    try:
        for idx, book in enumerate(books[11650:]):
            book_url = book.get("book_url", "")
            if not book_url or book_url in processed_urls:
                continue

            print(f"\n[{idx + 1}/{len(books)}] Processing: {book_url}")

            img_url = book.get("image_url", "")
            img_name = download_image(img_url, idx)
            if not img_name:
                continue

            # Predict
            pred_class, conf = predict_image(model, img_name)
            print(f"\nResult: {pred_class} ({conf:.2%})")

            # Save result
            if pred_class in files:
                files[pred_class].write(f"{book_url}\n")
                files[pred_class].flush()
            else:
                print(f"!! Warning: Predicted '{pred_class}', but no output file exists.")

            # Mark as processed
            processed_urls.add(book_url)
            append_processed_url(book_url)

            # Cleanup
            cleanup_image(img_name)

    finally:
        # Close all output files safely
        for f in files.values():
            f.close()

    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder for results.")

# -------------------- RUN ---------------------------

if __name__ == "__main__":
    main()





# import os
# import json
# import time
#
# import requests
# from ultralytics import YOLO
#
#
# # Load trained model
# # model = YOLO(r'C:\office work\runs\classify\train4\weights\best.pt')
# model = YOLO(r'C:\office work\Books Identifier\runs\classify\train8\weights\best.pt')
#
# # INPUT_JSON = r"C:\office work\Books Identifier\source\books_output.json"
# INPUT_JSON = "books_output.json"
# with open(INPUT_JSON, "r", encoding="utf-8") as f:
#     books = json.load(f)
#
# output_dir = f"output_results"
# os.makedirs(output_dir, exist_ok=True)
#
# # 3. Define output file paths (Keys must match model classes)
# paths = {
#     "Photocopy_Pdf": os.path.join(output_dir, "photocopy_pdf.txt"),
#     "Real_Book": os.path.join(output_dir, "real_book.txt"),
#     "Unknown": os.path.join(output_dir, "Unknown.txt")
# }
#
# # Open all files with utf-8 encoding and line buffering
# files = {cls: open(path, "w", encoding="utf-8") for cls, path in paths.items()}
#
# print(f"Model Classes: {model.names}")
#
# try:
#     for idx, book in enumerate(books[10000:]):
#         print(f"\n[{idx + 1}/{len(books)}] Processing: {book['book_url']}")
#
#         img_name = f"temp_{idx}.webp"
#         try:
#             image = book.get("image_url", "")
#             response = requests.get(image, timeout=15)
#
#             if response.status_code !=200:
#                 print(f"Image {image} Not found!\n ")
#                 continue
#
#             with open(img_name, "wb") as f:
#                 f.write(response.content)
#         except Exception as e:
#             print(f"Skipping: Download failed - {e}")
#             continue
#
#         # Predict
#         results = model(img_name, verbose=False)
#         result = results[0]
#
#         # Get class name and clean it up to match our keys
#         # We use .title() to convert 'photocopy' -> 'Photocopy'
#         raw_pred = model.names[result.probs.top1]
#         pred_class = raw_pred.strip().title()
#
#         # Special case for PDF if your folder was lowercase 'pdf'
#         if pred_class == "Photocopy/Pdf": pred_class = "PHOTOCOPY/PDF"
#
#         conf = result.probs.top1conf.item()
#         print(f"→ Result: {pred_class} ({conf:.2%})")
#
#         # Write to file
#         if pred_class in files:
#             files[pred_class].write(f"{book['book_url']}\n")
#             files[pred_class].flush()  # Force write to file so it's not empty if crashed
#         else:
#             print(f"!! Warning: Model predicted '{pred_class}', but no output file exists for this name.")
#             print(f"Available keys in script: {list(files.keys())}")
#
#         # Clean up image
#         if os.path.exists(img_name):
#             try:
#                 time.sleep(0.2)
#                 os.remove(img_name)
#             except PermissionError:
#                 print(f"⚠️ Could not delete {img_name}, skipping...")
#
#
# finally:
#     # Safely close all files
#     for f in files.values():
#         f.close()
#
# print(f"\nDone! Check the '{output_dir}' folder for your .txt files.")
#
