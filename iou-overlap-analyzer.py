#!/usr/bin/env python3
# analyze_iou_overlaps.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywhn, xywh2xyxy
from ultralytics.utils.metrics import bbox_iou

def load_annotations(label_path, img_shape):
    """Load ground truth annotations from YOLO format .txt file"""
    if not os.path.exists(label_path):
        return np.zeros((0, 5))
    
    with open(label_path, 'r') as f:
        annotations = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
    
    if len(annotations) == 0:
        return np.zeros((0, 5))
    
    # Convert normalized xywh to pixel xyxy for IoU calculation
    annotations[:, 1:5] = xywh2xyxy(annotations[:, 1:5])
    
    # Scale normalized coordinates to image dimensions
    h, w = img_shape
    annotations[:, 1] *= w  # x1
    annotations[:, 2] *= h  # y1
    annotations[:, 3] *= w  # x2
    annotations[:, 4] *= h  # y2
    
    return annotations

def analyze_model_iou(model_path, data_path, output_dir, conf_thres=0.25, iou_thres=0.5):
    """Analyze IoU overlaps between model predictions and ground truth annotations"""
    # Load model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find validation images
    with open(data_path, 'r') as f:
        import yaml
        data_dict = yaml.safe_load(f)
    
    # Determine image and label paths
    dataset_root = data_dict.get('path', os.path.dirname(data_path))
    val_img_path = os.path.join(dataset_root, data_dict.get('val', 'images'))
    val_images = list(Path(val_img_path).glob('**/*.jpg')) + list(Path(val_img_path).glob('**/*.png'))
    
    # Statistics
    iou_values = []
    class_iou_values = {}
    
    # Process each image
    for img_path in tqdm(val_images, desc="Analyzing images"):
        # Determine label path
        img_name = img_path.stem
        label_path = os.path.join(dataset_root, 'labels', f"{img_name}.txt")
        
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load ground truth annotations
        gt_boxes = load_annotations(label_path, (h, w))
        
        # Run model prediction
        results = model(img_rgb, conf=conf_thres, iou=iou_thres)[0]
        pred_boxes = results.boxes.cpu().numpy()
        
        # For visualization
        img_with_boxes = img.copy()
        
        # Calculate IoU for each prediction
        for i, pred in enumerate(pred_boxes):
            xyxy = pred.xyxy[0].astype(int)
            cls = int(pred.cls[0])
            conf = pred.conf[0]
            
            # Draw prediction box (red)
            cv2.rectangle(img_with_boxes, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            
            # Find matching ground truth box with highest IoU
            max_iou = 0
            max_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                gt_cls = int(gt[0])
                gt_xyxy = gt[1:5].astype(int)
                
                # Skip different classes if gt class matches prediction class
                if gt_cls != cls:
                    continue
                
                # Calculate IoU
                box1 = np.array([[xyxy[0], xyxy[1], xyxy[2], xyxy[3]]])
                box2 = np.array([[gt_xyxy[0], gt_xyxy[1], gt_xyxy[2], gt_xyxy[3]]])
                iou = bbox_iou(box1, box2, xywh=False).item()
                
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j
            
            # Store IoU value if match found
            if max_iou > 0:
                iou_values.append(max_iou)
                
                # Store by class
                if cls not in class_iou_values:
                    class_iou_values[cls] = []
                class_iou_values[cls].append(max_iou)
                
                # Draw ground truth box (green) if matched
                gt = gt_boxes[max_gt_idx]
                gt_xyxy = gt[1:5].astype(int)
                cv2.rectangle(img_with_boxes, 
                             (gt_xyxy[0], gt_xyxy[1]), 
                             (gt_xyxy[2], gt_xyxy[3]), 
                             (0, 255, 0), 2)
                
                # Draw IoU value
                cv2.putText(img_with_boxes, 
                           f"IoU: {max_iou:.2f}", 
                           (xyxy[0], xyxy[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 2)
        
        # Save visualization
        output_img_path = os.path.join(output_dir, f"{img_name}_iou.jpg")
        cv2.imwrite(output_img_path, img_with_boxes)
    
    # Generate IoU distribution histogram
    if iou_values:
        plt.figure(figsize=(10, 6))
        plt.hist(iou_values, bins=20, alpha=0.7, color='blue')
        plt.title('IoU Distribution')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'iou_distribution.png'))
        
        # Generate per-class IoU distribution
        plt.figure(figsize=(12, 8))
        for cls, ious in class_iou_values.items():
            plt.hist(ious, bins=20, alpha=0.5, label=f'Class {cls}')
        plt.title('IoU Distribution by Class')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'iou_distribution_by_class.png'))
        
        # Calculate statistics
        avg_iou = np.mean(iou_values)
        median_iou = np.median(iou_values)
        
        # Write summary
        with open(os.path.join(output_dir, 'iou_summary.txt'), 'w') as f:
            f.write(f"IoU Analysis Summary\n")
            f.write(f"Total predictions: {len(iou_values)}\n")
            f.write(f"Average IoU: {avg_iou:.4f}\n")
            f.write(f"Median IoU: {median_iou:.4f}\n\n")
            
            f.write(f"Per-Class IoU:\n")
            for cls, ious in class_iou_values.items():
                f.write(f"Class {cls}:\n")
                f.write(f"  Count: {len(ious)}\n")
                f.write(f"  Average IoU: {np.mean(ious):.4f}\n")
                f.write(f"  Median IoU: {np.median(ious):.4f}\n\n")
        
        print(f"Analysis complete. Results saved to {output_dir}")
        return avg_iou
    else:
        print("No valid predictions found.")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Analyze IoU overlaps between YOLO predictions and ground truth")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True, help='Path to validation data.yaml')
    parser.add_argument('--output', type=str, default='./iou_analysis', help='Output directory for analysis results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS')
    args = parser.parse_args()
    
    analyze_model_iou(args.model, args.data, args.output, args.conf, args.iou)

if __name__ == "__main__":
    main()
