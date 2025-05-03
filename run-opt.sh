#!/bin/bash
# run_yolo_optimization.sh

# This script runs the complete pipeline for Bayesian optimization of YOLO hyperparameters

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }

# Parse command line arguments
DATA=""
EXTERNAL_VAL_DATA=""
MODEL="yolov12m.pt"
EPOCHS=20
TRIALS=20
DEVICE="0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data)
      DATA="$2"
      shift
      shift
      ;;
    --external-val-data)
      EXTERNAL_VAL_DATA="$2"
      shift
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --trials)
      TRIALS="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if data argument is provided
if [ -z "$DATA" ]; then
  echo "Error: --data argument is required."
  echo "Usage: ./run_yolo_optimization.sh --data path/to/data.yaml [--external-val-data path/to/external_val.yaml] [--model yolov12m.pt] [--epochs 20] [--trials 20] [--device 0]"
  exit 1
fi

# Ensure model path is absolute
if [[ "$MODEL" != /* ]] && [[ "$MODEL" != ./* ]]; then
  # If it's not an absolute path or relative path starting with ./
  # Check if it's a local file in the current directory
  if [ -f "$MODEL" ]; then
    MODEL="$(pwd)/$MODEL"
    echo "Converted model path to absolute: $MODEL"
  fi
  # Otherwise, assume it's a model name from the YOLO model hub
fi

# Ensure data path is absolute
if [[ "$DATA" != /* ]] && [[ "$DATA" != ./* ]]; then
  DATA="$(pwd)/$DATA"
  echo "Converted data path to absolute: $DATA"
fi

# Create project directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROJECT_DIR="yolo_optimization_${TIMESTAMP}"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

echo "========================================"
echo "YOLO Bayesian Optimization Pipeline"
echo "========================================"
echo "Data: $DATA"
if [ -n "$EXTERNAL_VAL_DATA" ]; then
  echo "External validation data: $EXTERNAL_VAL_DATA"
else
  echo "External validation data: None (will use validation split from data.yaml)"
fi
echo "Model: $MODEL"
echo "Epochs per trial: $EPOCHS"
echo "Number of trials: $TRIALS"
echo "Device: $DEVICE"
echo "Project directory: $(pwd)"
echo "========================================"

# Step 1: Run Bayesian hyperparameter optimization
echo "[1/3] Running Bayesian hyperparameter optimization..."
# Ensure external validation path is absolute if provided
if [ -n "$EXTERNAL_VAL_DATA" ]; then
  # Make external validation path absolute if it's not already
  if [[ "$EXTERNAL_VAL_DATA" != /* ]] && [[ "$EXTERNAL_VAL_DATA" != ./* ]]; then
    EXTERNAL_VAL_DATA="$(pwd)/$EXTERNAL_VAL_DATA"
    echo "Converted external validation path to absolute: $EXTERNAL_VAL_DATA"
  fi
  python3 ../bayesian-opt-yolo.py --data "$DATA" --external-val-data "$EXTERNAL_VAL_DATA" --model "$MODEL" --epochs $EPOCHS --trials $TRIALS --device $DEVICE --project ./optimization_results
else
  python3 ../bayesian-opt-yolo.py --data "$DATA" --model "$MODEL" --epochs $EPOCHS --trials $TRIALS --device $DEVICE --project ./optimization_results
fi

# Check if optimization completed successfully
if [ ! -f "./best_hyperparameters.yaml" ]; then
  echo "Error: Bayesian optimization failed or did not produce best hyperparameters."
  exit 1
fi
echo "Optimization completed successfully. Best hyperparameters saved to best_hyperparameters.yaml"

# Find the best model
BEST_MODEL=$(find ./optimization_results -name "best.pt" | sort -n | tail -1)
if [ -z "$BEST_MODEL" ]; then
  echo "Error: Could not find best model weights."
  exit 1
fi
echo "Best model found: $BEST_MODEL"

# Step 2: Train final model with best hyperparameters
echo "[2/3] Training final model with best hyperparameters..."

# Create a Python script with the best hyperparameters
echo "from ultralytics import YOLO
import yaml

# Load hyperparameters from YAML
with open('best_hyperparameters.yaml', 'r') as f:
    hyp = yaml.safe_load(f)

# Initialize the model with the exact path (not just model name)
model = YOLO('$MODEL')

# Train with best hyperparameters
model.train(
    data='$DATA',
    epochs=100,
    imgsz=640,
    device='$DEVICE',
    project='./final_model',
    **hyp  # Unpack hyperparameters as keyword arguments
)
" > train_final.py

# Run the Python script
python3 train_final.py

# Find the final model
FINAL_MODEL=$(find ./final_model -name "best.pt" | sort -n | tail -1)
if [ -z "$FINAL_MODEL" ]; then
  echo "Error: Could not find final model weights."
  exit 1
fi
echo "Final model trained: $FINAL_MODEL"

# Step 3: Analyze IoU overlaps if external validation data is provided
if [ -n "$EXTERNAL_VAL_DATA" ]; then
  echo "[3/3] Analyzing IoU overlaps on external validation dataset..."
  python3 ../iou-overlap-analyzer.py --model "$FINAL_MODEL" --data "$EXTERNAL_VAL_DATA" --output ./iou_analysis
  
  echo "========================================"
  echo "Optimization pipeline completed!"
  echo "========================================"
  echo "Results summary:"
  echo "- Best hyperparameters: ./best_hyperparameters.yaml"
  echo "- Final model: $FINAL_MODEL"
  echo "- IoU analysis: ./iou_analysis/iou_summary.txt"
  echo "- Optimization visualizations: ./optimization_history.html"
else
  echo "[3/3] No external validation dataset provided. Skipping IoU analysis."
  
  echo "========================================"
  echo "Optimization pipeline completed!"
  echo "========================================"
  echo "Results summary:"
  echo "- Best hyperparameters: ./best_hyperparameters.yaml"
  echo "- Final model: $FINAL_MODEL"
  echo "- Optimization visualizations: ./optimization_history.html"
fi

echo "========================================"
echo "To use the optimized model, run:"
echo "from ultralytics import YOLO"
echo "model = YOLO('$FINAL_MODEL')"
echo "========================================"