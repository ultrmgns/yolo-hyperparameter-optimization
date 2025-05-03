#!/usr/bin/env python3
# bayesian_yolo_optimizer.py

import os
import yaml
import optuna
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Default hyperparameter ranges - these will be passed directly to the YOLO train method
DEFAULT_HYP_RANGES = {
    'lr0': (0.001, 0.1),           # Initial learning rate
    'momentum': (0.8, 0.99),       # SGD momentum
    'weight_decay': (0.0001, 0.001),  # Weight decay
    'hsv_h': (0.0, 0.1),           # HSV hue augmentation
    'hsv_s': (0.0, 0.9),           # HSV saturation augmentation
    'hsv_v': (0.0, 0.9),           # HSV value augmentation
    'degrees': (0.0, 1.0),         # Rotation augmentation
    'translate': (0.0, 0.2),       # Translation augmentation
    'scale': (0.0, 0.9),           # Scale augmentation
    'fliplr': (0.0, 0.5),          # Horizontal flip probability
}

def objective(trial, args):
    """Define the objective function to be optimized."""
    # Sample hyperparameters
    train_args = {}
    
    # Add the base arguments
    train_args['data'] = args.data
    train_args['epochs'] = args.epochs
    train_args['device'] = args.device
    train_args['project'] = args.project
    train_args['name'] = f'trial_{trial.number}'
    train_args['val'] = True  # Always validate during training
    
    # Add hyperparameters that will be directly passed to train method
    for param_name, param_range in DEFAULT_HYP_RANGES.items():
        if param_name == 'lr0':
            # Log-uniform distribution for learning rate
            train_args[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
        else:
            train_args[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
    
    # Add specific parameters you might want to tune
    train_args['batch'] = trial.suggest_categorical('batch', [4, 8, 16, 32])
    train_args['imgsz'] = trial.suggest_categorical('imgsz', [416, 512, 640, 768])
    
    # Initialize the model
    try:
        # Use explicit model path from args directly
        model = YOLO(args.model)
        
        # Print model path being used for debugging
        print(f"Loading model from: {args.model}")
        
        # Train the model with the sampled hyperparameters directly passed
        results = model.train(**train_args)
        
        # Extract the best validation metrics
        # For YOLO, a good metric is mAP50-95 (mean Average Precision)
        metrics = results.results_dict
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
        
        # Optuna maximizes the objective by default, so we return the mAP
        return map50_95
    except Exception as e:
        print(f"Training failed with error: {e}")
        return 0.0  # Return a low score so this trial is not selected

def evaluate_best_model(best_trial, args):
    """Evaluate the best model on a separate validation dataset."""
    # If no external validation dataset is provided, skip evaluation
    if not args.external_val_data:
        print("No external validation dataset provided. Skipping final evaluation.")
        return None
    
    # Get the best hyperparameters
    best_params = {
        param: best_trial.params.get(param, (DEFAULT_HYP_RANGES[param][0] + DEFAULT_HYP_RANGES[param][1]) / 2)
        for param in DEFAULT_HYP_RANGES
    }
    
    # Add specific parameters that were tuned
    batch_size = best_trial.params.get('batch', 16)
    img_size = best_trial.params.get('imgsz', 640)
    
    # Save best hyperparameters to a YAML file for reference and reuse
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump(best_params, f)
    
    # Path to the best model from the optimization
    best_model_path = Path(args.project) / f'trial_{best_trial.number}' / 'weights' / 'best.pt'
    
    # Initialize the model with the best weights
    if best_model_path.exists():
        model = YOLO(str(best_model_path))
        
        # Validate on the external validation dataset
        results = model.val(
            data=args.external_val_data,
            batch=batch_size,
            imgsz=img_size,
            device=args.device,
            project=args.project,
            name='final_evaluation'
        )
        
        print(f"\nBest model evaluation on external validation dataset:")
        print(f"mAP50-95: {results.box.map}")
        print(f"mAP50: {results.box.map50}")
        print(f"Precision: {results.box.p}")
        print(f"Recall: {results.box.r}")
        
        return results
    else:
        print(f"Best model weights not found at {best_model_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Optimization for YOLO")
    parser.add_argument('--data', type=str, required=True, help='Path to training data.yaml (with train/val splits)')
    parser.add_argument('--external-val-data', type=str, default='', help='Path to external validation data.yaml for final evaluation')
    parser.add_argument('--model', type=str, default='yolov12m.pt', help='Initial model path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per trial')
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--project', type=str, default='runs/bayesian_opt', help='Project directory')
    args = parser.parse_args()
    
    # Print model path for debugging
    print(f"Using model: {args.model}")
    
    # Check if model path exists
    model_path = Path(args.model)
    if not model_path.exists() and not args.model.startswith('yolo'):
        print(f"WARNING: Model path {args.model} does not exist!")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="yolo_bayesian_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Start optimization
    print(f"Starting Bayesian Optimization with {args.trials} trials")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    # Print optimization results
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best hyperparameters
    best_params = best_trial.params
    best_hyp = {param: best_params.get(param, DEFAULT_HYP_RANGES[param][0]) 
                for param in DEFAULT_HYP_RANGES}
    
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump(best_hyp, f)
    
    print(f"\nBest hyperparameters saved to best_hyperparameters.yaml")
    
    # Evaluate the best model on the external validation dataset if provided
    if args.external_val_data:
        evaluate_best_model(best_trial, args)
    
    # Generate optimization visualization
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html('optimization_history.html')
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html('param_importances.html')
        
        fig = optuna.visualization.plot_contour(study)
        fig.write_html('contour_plot.html')
        
        print("\nOptimization visualizations saved as HTML files.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()