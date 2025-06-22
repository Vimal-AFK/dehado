import cv2
import torch
import os
import json
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# --- Load models ---
yolo_model = YOLO("./models/yolov8_latest_trained_model.pt")
processor = TrOCRProcessor.from_pretrained(".models/conservative_trocr_model/final")
model = VisionEncoderDecoderModel.from_pretrained(".models/conservative_trocr_model/final")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# --- Paths ---
test_images_dir = "IMAGES/"
save_dir = "predictions_of_IMAGES/"
os.makedirs(save_dir, exist_ok=True)

def sort_boxes(boxes):
    # Calculate y-center of each box
    y_centers = [(box[1] + box[3]) / 2 for box in boxes]
    
    # Cluster boxes into lines using y-center (with 20px tolerance)
    clusters = {}
    for i, y in enumerate(y_centers):
        matched = False
        for cluster_y in clusters:
            if abs(y - cluster_y) < 20:  # 20px tolerance for line grouping
                clusters[cluster_y].append(boxes[i])
                matched = True
                break
        if not matched:
            clusters[y] = [boxes[i]]
    
    # Sort clusters by y position (top to bottom)
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[0])
    
    # Sort boxes within each cluster by x position (left to right)
    sorted_boxes = []
    for _, cluster_boxes in sorted_clusters:
        sorted_boxes.extend(sorted(cluster_boxes, key=lambda b: b[0]))
    
    return sorted_boxes

def process_image(image_path):
    # --- Load image and run YOLO detection ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return []
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)[0]

    # Collect all detections
    boxes = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

    # Sort boxes in reading order
    sorted_boxes = sort_boxes(boxes)

    ocr_results = []
    for x1, y1, x2, y2, conf in sorted_boxes:
        crop = image_rgb[y1:y2, x1:x2]
        pil_crop = Image.fromarray(crop)

        inputs = processor(images=pil_crop, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(inputs["pixel_values"])
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        ocr_results.append({
            "text": text,
            "bbox": [x1, y1, x2, y2]
        })

    return ocr_results

def save_json_custom(data, filepath):
    """Custom JSON writer that matches the exact requested format"""
    with open(filepath, 'w') as f:
        f.write('[\n')
        for i, item in enumerate(data):
            f.write('\t{\n')
            f.write(f'\t    "text": {json.dumps(item["text"])},\n')
            f.write(f'\t    "bbox": {json.dumps(item["bbox"])}\n')
            f.write('\t}' + (',' if i < len(data)-1 else '') + '\n')
        f.write(']\n')

# --- Process all test images ---
for image_name in os.listdir(test_images_dir):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(save_dir, json_name)
        
        # Skip if JSON already exists
        if os.path.exists(json_path):
            print(f"Skipping {image_name} - output already exists")
            continue
            
        image_path = os.path.join(test_images_dir, image_name)
        
        try:
            # Process the image
            ocr_results = process_image(image_path)
            
            # Save JSON with same name as image
            save_json_custom(ocr_results, json_path)
            
            print(f"Processed {image_name} -> Saved to {json_path}")
      
            json_count = len([f for f in os.listdir(save_dir) if f.endswith('.json')])
            print(f"Number of JSON files: {json_count}")
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

print("Processing complete!")