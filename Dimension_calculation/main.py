import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
from ultralytics import YOLO
from torchvision.ops import nms

# Load the custom YOLOv8 model from Kaggle input

#model_path = "/kaggle/input/obj-detect-model/150-best (4).pt"
#model = YOLO(model_path)

model = YOLO("yolov8x.pt")

# Dataset folder path from Kaggle input
image_folder = "/kaggle/input/dist-calc-imag2/dist_calc_imag/4.27mm"
image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")) +
                     glob(os.path.join(image_folder, "*.jpeg")) +
                     glob(os.path.join(image_folder, "*.png")))

# Process each image
for image_path in image_paths:
    print(f"\n--- Processing: {os.path.basename(image_path)} ---")
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Unable to read image.")
        continue

    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO model
    results = model(image_rgb)

    # Extract predictions
    boxes = results[0].boxes.xyxy.clone().detach()
    scores = results[0].boxes.conf.clone().detach()
    class_ids = results[0].boxes.cls.clone().detach()

    # Apply Non-Maximum Suppression (NMS)
    iou_threshold = 0.5
    nms_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    boxes = boxes[nms_indices].cpu().numpy()
    scores = scores[nms_indices].cpu().numpy()
    class_ids = class_ids[nms_indices].cpu().numpy()

    for box, score, class_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = map(int, box)
        label = f'{model.names[int(class_id)]}: {score:.2f}'

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        width = (xmax - xmin) * (0.0104166667) #Convert pixels to inches with 0.0104...
        height = (ymax - ymin) * (0.0104166667)
        print(f"Detected {model.names[int(class_id)]} -> Width: {width:.2f}in, Height: {height:.2f}in")

    # Save or display the result
    output_path = f"/kaggle/working/output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, frame)
