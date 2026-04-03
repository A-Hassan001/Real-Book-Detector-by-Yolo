from ultralytics import YOLO # You can still use YOLOv8-cls for the speed

model = YOLO('yolov8n-cls.pt')

# Start training with 3 classes: Real_Book, Photocopy/PDF, Unknown
# model.train(
#     data=r'C:\office work\Books Identifier\dataset',
#     epochs=100,               # Increased slightly for 3-class complexity
#     imgsz=224,
#     batch=32,
#     # --- Advanced Augmentations ---
#     hsv_s=0.0,               # Force model to ignore color ink vs black toner
#     hsv_v=0.4,               # Mimics the bright/dark scan variations
#     scale=0.6,               # Helps detect spiral rings at different distances
#     dropout=0.15,            # Higher dropout to prevent memorizing specific text
#     erasing=0.2,             # Mimics "bad scans" where parts are missing
#     label_smoothing=0.1      # Good for ambiguous reprints
# )


#   ============================ More efficient training code ====================
# from ultralytics import YOLO
#
# model = YOLO('yolov8n-cls.pt')
#
model.train(
    data=r'C:\office work\Books Identifier\dataset',
    epochs=100,
    imgsz=224,
    batch=-1,            # Auto-adjust batch size
    cache=True,          # Speed up training
    patience=15,         # Stop early if no improvement
    # --- Refined Augmentations ---
    hsv_h=0.0,
    hsv_s=0.0,           # Ignore color (focus on texture/grain)
    hsv_v=0.4,
    scale=0.5,
    fliplr=0.5,
    # blur=0.1,            # Mimics scanner blur
    erasing=0.3,         # Mimics missing scan data
    dropout=0.2,         # Stronger prevention of text-memorization
    label_smoothing=0.1
)

# Validation
#
# results = model.val(data='C:\office work\Books Identifier\dataset')
# # print(f"Top-1 Accuracy: {metrics.top1}")
# print(results)

