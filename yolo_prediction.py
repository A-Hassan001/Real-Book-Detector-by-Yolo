from ultralytics import YOLO
import os

# 1. Load your CUSTOM trained weights
# Replace this path with the actual location of your best.pt
model_path = r'C:\office work\Books Identifier\runs\classify\train8\weights\best.pt'
model = YOLO(model_path)

# 2. Define your source (can be a single image, a folder, or a URL)
source_path = r'C:\office work\Books Identifier\new_scans'

# 3. Run prediction
# Using stream=True is critical for your 15,000 image dataset to avoid RAM crashes
results = model.predict(
    source=source_path,
    save=True,  # Saves the image with the label "Real_Book 99%" on it
    conf=0.5,  # Only show results with more than 50% confidence
    stream=True  # Efficient memory management
)

# 4. Process the results
print("--- Prediction Results ---")
for result in results:
    # Get the top class index and the name
    top_idx = result.probs.top1
    class_name = result.names[top_idx]
    confidence = result.probs.top1conf.item()

    # Print the filename and what the model thinks it is
    filename = os.path.basename(result.path)
    print(f"File: {filename} | Prediction: {class_name} ({confidence:.2%})")
    d=1