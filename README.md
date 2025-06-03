# YOLO Hyperparameter Optimization Framework

A comprehensive pipeline for automatically optimizing YOLO model hyperparameters using Bayesian optimization.

## Overview

This framework provides an end-to-end solution for finding optimal hyperparameters for YOLO object detection models. It leverages Bayesian optimization (via Optuna) to efficiently search the hyperparameter space, trains a final model with the best parameters, and uses an external dataset (outside training) to analyze model performance (iou_analyzer). Based on the results, it self-adjust hyperparameters and re-trains with the new.

## Components

- `run-opt.sh`: Main entry script that orchestrates the entire optimization pipeline
- `bayesian-opt-yolo.py`: Implements Bayesian optimization to find optimal hyperparameters
- `iou-overlap-analyzer.py`: Evaluates model performance with detailed IoU analysis against a different annotated dataset
- `train-final.py`: Trains the final model using the best discovered hyperparameters from all runs

## Requirements

- Python 3.8+
- Required Python packages:
  - ultralytics
  - optuna
  - numpy
  - matplotlib
  - opencv-python
  - pyyaml
  - tqdm

## Usage

```bash
./run-opt.sh --data PATH_TO_DATA_YAML --external-val-data PATH_TO_VALIDATION_YAML --model MODEL_PATH --epochs EPOCHS --trials TRIALS --device DEVICE
```

### Arguments

- `--data`: Path to the training data YAML file (required)
- `--external-val-data`: Path to validation data YAML file (optional)
- `--model`: Initial YOLO model to optimize (default: "yolov12m.pt")
- `--epochs`: Number of epochs per trial (default: 20)
- `--trials`: Number of optimization trials (default: 20)
- `--device`: CUDA device index (default: "0")

### Example

```bash
./run-opt.sh --data ../datasets/2025_1_98_manual/data.yaml --external-val-data ../datasets/final_validation/data.yaml --model yolov12m.pt --epochs 50 --trials 30 --device 0
```

## Optimization Process

1. **Bayesian Optimization**: Systematically explores various hyperparameters to find the optimal combination.
2. **Final Training**: Trains a model using the best hyperparameters for an extended number of epochs.
3. **IoU Analysis**: Analyzes the final model's performance on a different validation dataset.

## Hyperparameters Optimized

- Learning rate (lr0)
- Momentum
- Weight decay
- Data augmentation parameters (HSV, rotation, translation, etc.)
- Batch size
- Image size

## Output

The optimization process creates a timestamped directory containing:

- `best_hyperparameters.yaml`: Best hyperparameter values discovered
- `optimization_results/`: Training results for each trial
- `final_model/`: Final model trained with the best hyperparameters
- `iou_analysis/`: IoU analysis results (if external validation data provided)
- Visualization plots in HTML format:
  - `optimization_history.html`
  - `param_importances.html`
  - `contour_plot.html`

## Using the Optimized Model

After optimization, you can use the final model in your Python code:

```python
from ultralytics import YOLO
model = YOLO('/path/to/final_model/best.pt')
results = model.predict('path/to/image.jpg')
```
