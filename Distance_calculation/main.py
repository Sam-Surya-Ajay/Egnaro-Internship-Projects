#latest 14may
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
import math

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Paths and settings
image_folder = "/kaggle/input/dist-calc-imag2/dist_calc_imag/4.27mm"
output_path = "/kaggle/working/results/"
os.makedirs(output_path, exist_ok=True)

# Real-world object dimensions (width, height) in cm
object_dimensions = {
    "person": (50, 170),
    "cup": (9, 12),
    "cell phone": (7.5, 15),
    "bottle": (8, 25),
    "keyboard": (35, 15),
    "toothbrush": (2, 18),
    "backpack": (30, 45),
    "spoon": (3, 17),
    "fork": (2.5, 18),
    "remote": (5.5, 17),
    "knife": (2.7, 22),
    "book": (22, 30),
    "laptop": (38, 25),
    "chair": (50, 100),
    "clock": (20, 20),
    "tv": (120, 70),
    "potted plant": (35, 50),
    "sink": (50, 40),
    "toilet": (50, 70),
    "mirror": (40, 60)
}

# Apply tilt correction only to these
tilted_objects = {"sink", "toilet", "urinal", "indian toilet"}

# Camera parameters
SENSOR_WIDTH_MM = 5.04
FOCAL_LENGTH_MM = 4.27

def get_correction_factor(focal_mm):
    return -0.0566 * focal_mm + 1.591
    
CORRECTION_FACTOR = get_correction_factor(FOCAL_LENGTH_MM)

# Get camera angle from user, default to 45 if no input
try:
    user_input = input("Enter camera angle in degrees (press Enter for default 45°): ").strip()
    CAMERA_ANGLE_DEG = float(user_input) if user_input else 45.0
except ValueError:
    print("Invalid input. Defaulting to 45°.")
    CAMERA_ANGLE_DEG = 45.0

COS_THETA = math.cos(math.radians(CAMERA_ANGLE_DEG))


# Detection thresholds
conf_threshold = 0.5
iou_threshold = 0.45
min_pixel_size = 15
edge_margin = 2

# Visualization
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
TEXT_BG_COLOR = (200, 255, 200)
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2

def get_focal_length_px(image_width_px, focal_mm, sensor_mm):
    return (image_width_px * focal_mm) / sensor_mm

def calculate_distance(real_diag_cm, pixel_diag, focal_px, apply_angle_correction=False):
    if pixel_diag <= 0:
        return None
    distance_cm = (real_diag_cm * focal_px) / pixel_diag
    distance_cm *= CORRECTION_FACTOR
    if apply_angle_correction:
        distance_cm *= COS_THETA
    return round(distance_cm / 2.54, 1)

# Load images
image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")) +
                     glob(os.path.join(image_folder, "*.jpeg")) +
                     glob(os.path.join(image_folder, "*.png")))

successful_detections = 0

for image_path in image_paths:
    img_name = os.path.basename(image_path)
    print(f"\n--- Processing: {img_name} ---")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        continue

    img_height, img_width = img.shape[:2]
    focal_px = get_focal_length_px(img_width, FOCAL_LENGTH_MM, SENSOR_WIDTH_MM)

    results = model(img, conf=conf_threshold, iou=iou_threshold)
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_w = x2 - x1
            pixel_h = y2 - y1

            if pixel_w < min_pixel_size or class_name not in object_dimensions:
                continue

            if x1 < edge_margin or y1 < edge_margin or x2 > img_width - edge_margin or y2 > img_height - edge_margin:
                continue

            aspect_ratio = pixel_w / (pixel_h + 1e-6)
            if not (0.05 < aspect_ratio < 15):
                continue

            real_w, real_h = object_dimensions[class_name]
            real_diag_cm = np.sqrt(real_w**2 + real_h**2)
            pixel_diag = np.sqrt(pixel_w**2 + pixel_h**2)

            apply_correction = class_name.lower() in tilted_objects
            distance_in = calculate_distance(real_diag_cm, pixel_diag, focal_px, apply_correction)
            if distance_in is None:
                continue

            detected_objects.append({
                'class': class_name,
                'distance_in': distance_in,
                'confidence': confidence,
                'box': (x1, y1, x2, y2)
            })

    for obj in detected_objects:
        print(f"Class: {obj['class']} - Distance: {obj['distance_in']} in")

    # Draw boxes
    detected_objects.sort(key=lambda x: x['distance_in'], reverse=True)
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['box']
        label = f"{obj['class']}: {obj['distance_in']}in"
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), TEXT_BG_COLOR, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    if detected_objects:
        successful_detections += 1
        output_file = os.path.join(output_path, f"result_{img_name}")
        cv2.imwrite(output_file, img)
        print(f"Saved result to {output_file}")

        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Result - {img_name}")
        summary = "\n".join([f"{obj['class']}: {obj['distance_in']} in" for obj in detected_objects])
        plt.figtext(0.05, 0.05, summary, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.show()

print(f"\nProcessing complete. Successful detections: {successful_detections}")
