#!/usr/bin/env python3
# train_final.py
from ultralytics import YOLO
import yaml

# Path to your model and data
MODEL_PATH = "/home/gamerock/scythe1/yolov12n.pt"
DATA_PATH = "/home/gamerock/scythe1/datasets/2025_1_98_manual/data.yaml"
DEVICE = "0"

# Load hyperparameters from YAML
with open('best_hyperparameters.yaml', 'r') as f:
    hyp = yaml.safe_load(f)

print("Loaded hyperparameters:")
for key, value in hyp.items():
    print(f"  {key}: {value}")

# Initialize the model
print(f"\nInitializing model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Train with best hyperparameters
print("\nStarting training with best hyperparameters...")
model.train(
    data=DATA_PATH,
    epochs=100,
    imgsz=640,
    device=DEVICE,
    project='./final_model',
    **hyp  # Unpack hyperparameters as keyword arguments
)

print("\nTraining complete!")
