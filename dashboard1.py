#!/usr/bin/env python3
# fixed_dashboard.py

import os
import glob
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from collections import defaultdict

def find_optimization_directory():
    """Find the most recent optimization directory"""
    dirs = sorted(glob.glob("./yolo_optimization_*"), key=os.path.getmtime, reverse=True)
    if not dirs:
        dirs = ["."]  # Use current directory if no optimization directory found
    return dirs[0]

def collect_trial_data(base_dir):
    """Collect data from all trial directories"""
    print(f"Looking for trials in {base_dir}...")
    
    # Find all trial directories
    trial_dirs = glob.glob(os.path.join(base_dir, "optimization_results/trial_*"))
    if not trial_dirs:
        trial_dirs = glob.glob(os.path.join(base_dir, "*/trial_*"))
    
    if not trial_dirs:
        print("No trial directories found!")
        return None
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    # Collect data
    trials_data = []
    
    for trial_dir in trial_dirs:
        trial_num = int(os.path.basename(trial_dir).split("_")[-1])
        
        # Try to find results.csv
        results_files = glob.glob(os.path.join(trial_dir, "results.csv"))
        results_files.extend(glob.glob(os.path.join(trial_dir, "*.csv")))
        
        # Try to find args.yaml or hyp file
        args_files = glob.glob(os.path.join(trial_dir, "args.yaml"))
        args_files.extend(glob.glob(os.path.join(trial_dir, "hyp*.yaml")))
        args_files.extend(glob.glob(os.path.join(base_dir, f"hyp_trial_{trial_num}.yaml")))
        
        if not results_files:
            print(f"No results file found for trial {trial_num}")
            continue
            
        if not args_files:
            print(f"No args/hyp file found for trial {trial_num}")
            continue
        
        # Get final metrics
        try:
            results_df = pd.read_csv(results_files[0])
            final_map = results_df['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in results_df.columns else 0
            
            # Get hyperparameters
            with open(args_files[0], 'r') as f:
                args = yaml.safe_load(f)
            
            # Create trial data entry
            trial_data = {
                'trial': trial_num,
                'mAP': final_map,
                'epochs': len(results_df) if not results_df.empty else 0,
                'final_loss': results_df['train/box_loss'].iloc[-1] if 'train/box_loss' in results_df.columns and not results_df.empty else 0,
            }
            
            # Add hyperparameters
            for param, value in args.items():
                if param not in trial_data and isinstance(value, (int, float)):
                    trial_data[param] = value
            
            trials_data.append(trial_data)
            print(f"Processed trial {trial_num} - mAP: {final_map:.4f}")
            
        except Exception as e:
            print(f"Error processing trial {trial_num}: {e}")
    
    # Convert to DataFrame
    if trials_data:
        return pd.DataFrame(trials_data)
    else:
        print("No valid trial data found!")
        return None

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def create_optimization_history_plot(df):
    """Create optimization history plot"""
    print("Creating optimization history plot...")
    
    if df is None or df.empty:
        print("No data for optimization history plot")
        return None
    
    # Sort by trial number
    df = df.sort_values('trial')
    
    # Calculate best mAP so far
    best_so_far = df['mAP'].cummax()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['trial'], df['mAP'], 'o-', label='Trial mAP')
    ax.plot(df['trial'], best_so_far, 'r--', label='Best mAP so far')
    
    # Highlight best trial
    best_idx = df['mAP'].idxmax()
    best_trial = df.loc[best_idx, 'trial']
    best_map = df.loc[best_idx, 'mAP']
    ax.scatter([best_trial], [best_map], color='gold', s=200, zorder=5, label=f'Best Trial ({best_trial})', edgecolor='black')
    
    ax.set_title('Bayesian Optimization History')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('mAP50-95')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    return fig_to_base64(fig)

def create_param_importance_plot(df):
    """Create parameter importance plot"""
    print("Creating parameter importance plot...")
    
    if df is None or df.empty:
        print("No data for parameter importance plot")
        return None
    
    # Identify hyperparameters (excluding metrics and metadata)
    hyperparam_cols = [col for col in df.columns if col not in ['trial', 'mAP', 'epochs', 'final_loss']]
    
    if not hyperparam_cols:
        print("No hyperparameter columns found for importance analysis")
        return None
    
    # Calculate correlation with mAP
    correlations = []
    for param in hyperparam_cols:
        try:
            if df[param].nunique() > 1:  # Only consider parameters with variation
                corr = abs(df['mAP'].corr(df[param]))
                if not np.isnan(corr):
                    correlations.append((param, corr))
        except:
            pass
    
    if not correlations:
        print("No valid correlations found for parameter importance")
        return None
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    params = [x[0] for x in correlations]
    values = [x[1] for x in correlations]
    
    # Plot horizontal bars
    bars = ax.barh(params, values)
    
    # Color bars by importance
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(values[i]/max(values)))
    
    ax.set_title('Hyperparameter Importance (Correlation with mAP)')
    ax.set_xlabel('Absolute Correlation')
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def create_hyperparameter_scatter_plots(df):
    """Create scatter plots of hyperparameters vs mAP"""
    print("Creating hyperparameter scatter plots...")
    
    if df is None or df.empty:
        print("No data for hyperparameter scatter plots")
        return None
    
    # Identify hyperparameters (excluding metrics and metadata)
    hyperparam_cols = [col for col in df.columns if col not in ['trial', 'mAP', 'epochs', 'final_loss']]
    
    if not hyperparam_cols:
        print("No hyperparameter columns found for scatter plots")
        return None
    
    # Select top parameters by correlation
    correlations = []
    for param in hyperparam_cols:
        try:
            if df[param].nunique() > 1:
                corr = abs(df['mAP'].corr(df[param]))
                if not np.isnan(corr):
                    correlations.append((param, corr))
        except:
            pass
            
    # Sort and take top 6
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_params = [x[0] for x in correlations[:min(6, len(correlations))]]
    
    if not top_params:
        return None
    
    # Create figure with subplots
    n_params = len(top_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    axs = axs.flatten()
    
    # Create scatter plots
    for i, param in enumerate(top_params):
        if i < len(axs):
            ax = axs[i]
            try:
                sc = ax.scatter(df[param], df['mAP'], c=df['mAP'], cmap='viridis', alpha=0.8)
                ax.set_xlabel(param)
                ax.set_ylabel('mAP')
                ax.set_title(f'{param} vs mAP')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add best trial
                best_idx = df['mAP'].idxmax()
                best_param = df.loc[best_idx, param]
                best_map = df.loc[best_idx, 'mAP']
                ax.scatter([best_param], [best_map], color='red', s=100, zorder=5, edgecolor='black')
            except Exception as e:
                ax.text(0.5, 0.5, f"Could not plot {param}", horizontalalignment='center', verticalalignment='center')
                print(f"Error plotting {param}: {e}")
    
    # Hide empty subplots
    for i in range(len(top_params), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def get_improvement_summary(df):
    """Calculate improvement statistics"""
    if df is None or df.empty:
        return {}
    
    # Sort by trial number
    df_sorted = df.sort_values('trial')
    
    # Get best trial
    best_idx = df['mAP'].idxmax()
    best_trial = df.loc[best_idx, 'trial']
    best_map = df.loc[best_idx, 'mAP']
    
    # Calculate stats
    initial_map = df_sorted['mAP'].iloc[0] if not df_sorted.empty else 0
    
    # Get median of first 3 trials for baseline (more stable than just first)
    baseline_map = df_sorted.iloc[:min(3, len(df_sorted))]['mAP'].median()
    
    results = {
        'total_trials': len(df),
        'best_trial': int(best_trial),
        'best_map': best_map,
        'initial_map': initial_map,
        'baseline_map': baseline_map,
        'median_map': df['mAP'].median(),
        'mean_map': df['mAP'].mean(),
        'min_map': df['mAP'].min(),
        'max_map': df['mAP'].max(),
    }
    
    # Calculate improvements
    if baseline_map > 0:
        results['improvement_over_baseline'] = (best_map - baseline_map) / baseline_map * 100
    else:
        results['improvement_over_baseline'] = float('inf')
        
    if initial_map > 0:
        results['improvement_over_initial'] = (best_map - initial_map) / initial_map * 100
    else:
        results['improvement_over_initial'] = float('inf')
    
    # Get hyperparameters of best trial
    for col in df.columns:
        if col not in ['mAP', 'epochs', 'final_loss']:
            results[f'best_{col}'] = df.loc[best_idx, col]
    
    return results

def create_dashboard_html(df, base_dir):
    """Create a comprehensive HTML dashboard with all visualizations"""
    print("Creating dashboard HTML...")
    
    if df is None or df.empty:
        print("No data for dashboard")
        return False
    
    # Generate plots
    history_plot = create_optimization_history_plot(df)
    param_importance_plot = create_param_importance_plot(df)
    scatter_plots = create_hyperparameter_scatter_plots(df)
    
    # Get improvement summary
    summary = get_improvement_summary(df)
    
    # Format values for template
    total_trials = summary['total_trials']
    best_map = summary['best_map']
    best_trial = summary['best_trial']
    baseline_map = summary['baseline_map']
    improvement = summary['improvement_over_baseline'] if 'improvement_over_baseline' in summary and summary['improvement_over_baseline'] != float('inf') else 0
    initial_map = summary['initial_map']
    mean_map = summary['mean_map']
    
    # Create HTML using a template approach instead of % formatting
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Hyperparameter Optimization Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .improvement-summary {{
                background-color: #f0f8ff;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 0 5px rgba(0,0,0,0.05);
            }}
            .stats-container {{
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .stat-box {{
                background-color: #007BFF;
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                width: 22%;
                min-width: 200px;
                margin-bottom: 15px;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
            }}
            .stat-label {{
                font-size: 14px;
                opacity: 0.9;
            }}
            .plot-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .plot-container img {{
                max-width: 100%;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .best-params {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .top-models {{
                margin: 30px 0;
            }}
            footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 14px;
                color: #666;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO Hyperparameter Optimization Dashboard</h1>
            
            <div class="improvement-summary">
                <h2>Optimization Summary</h2>
                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-value">{total_trials}</div>
                        <div class="stat-label">Total Trials</div>
                    </div>
                    <div class="stat-box" style="background-color: #28a745;">
                        <div class="stat-value">{best_map:.4f}</div>
                        <div class="stat-label">Best mAP (Trial {best_trial})</div>
                    </div>
                    <div class="stat-box" style="background-color: #fd7e14;">
                        <div class="stat-value">{baseline_map:.4f}</div>
                        <div class="stat-label">Baseline mAP</div>
                    </div>
                    <div class="stat-box" style="background-color: #6f42c1;">
                        <div class="stat-value">{improvement:.1f}%</div>
                        <div class="stat-label">Improvement</div>
                    </div>
                </div>
                
                <h3>Performance Metrics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>Initial mAP</td>
                        <td>{initial_map:.4f}</td>
                        <td>Performance of first trial</td>
                    </tr>
                    <tr>
                        <td>Baseline mAP</td>
                        <td>{baseline_map:.4f}</td>
                        <td>Median of first 3 trials</td>
                    </tr>
                    <tr>
                        <td>Best mAP</td>
                        <td>{best_map:.4f}</td>
                        <td>Best performance achieved (Trial {best_trial})</td>
                    </tr>
                    <tr>
                        <td>Mean mAP</td>
                        <td>{mean_map:.4f}</td>
                        <td>Average performance across all trials</td>
                    </tr>
                    <tr>
                        <td>Improvement over baseline</td>
                        <td>{improvement:.1f}%</td>
                        <td>Percentage improvement from baseline to best</td>
                    </tr>
                </table>
            </div>
    """
    
    # Add optimization history plot
    if history_plot:
        html += f"""
            <div class="plot-container">
                <h2>Optimization History</h2>
                <img src="data:image/png;base64,{history_plot}" alt="Optimization History">
            </div>
        """
    
    # Add parameter importance plot
    if param_importance_plot:
        html += f"""
            <div class="plot-container">
                <h2>Hyperparameter Importance</h2>
                <img src="data:image/png;base64,{param_importance_plot}" alt="Hyperparameter Importance">
            </div>
        """
    
    # Add hyperparameter scatter plots
    if scatter_plots:
        html += f"""
            <div class="plot-container">
                <h2>Hyperparameter Effects on Performance</h2>
                <img src="data:image/png;base64,{scatter_plots}" alt="Hyperparameter Scatter Plots">
            </div>
        """
    
    # Add best hyperparameters
    html += f"""
            <div class="best-params">
                <h2>Best Hyperparameters (Trial {best_trial})</h2>
                <table>
                    <tr>
                        <th>Hyperparameter</th>
                        <th>Value</th>
                    </tr>
    """
    
    # Add hyperparameters of best model
    for key, value in summary.items():
        if key.startswith('best_') and key != 'best_trial' and key != 'best_map':
            param_name = key[5:]  # Remove 'best_' prefix
            html += f"""
                    <tr>
                        <td>{param_name}</td>
                        <td>{value}</td>
                    </tr>
            """
    
    html += """
                </table>
            </div>
            
            <div class="top-models">
                <h2>Top 5 Models</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Trial</th>
                        <th>mAP</th>
                        <th>Relative Performance</th>
                    </tr>
    """
    
    # Add top 5 models
    top_models = df.sort_values('mAP', ascending=False).head(min(5, len(df)))
    best_map_value = top_models['mAP'].iloc[0] if not top_models.empty else 0
    
    for i, (idx, row) in enumerate(top_models.iterrows()):
        relative = (row['mAP'] / best_map_value) * 100 if best_map_value > 0 else 0
        html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{int(row['trial'])}</td>
                        <td>{row['mAP']:.4f}</td>
                        <td>{relative:.1f}%</td>
                    </tr>
        """
    
    html += f"""
                </table>
            </div>
            
            <footer>
                Generated for optimization directory: {base_dir}
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open("yolo_optimization_dashboard.html", "w") as f:
        f.write(html)
    
    print("Dashboard created: yolo_optimization_dashboard.html")
    return True

def main():
    # Find optimization directory
    base_dir = find_optimization_directory()
    print(f"Using optimization directory: {base_dir}")
    
    # Collect trial data
    df = collect_trial_data(base_dir)
    
    if df is not None and not df.empty:
        # Create dashboard
        success = create_dashboard_html(df, base_dir)
        
        if success:
            print("\nDashboard generation complete!")
            print("Open yolo_optimization_dashboard.html in your browser to see the results.")
        else:
            print("Failed to create dashboard.")
    else:
        print("No trial data found. Dashboard generation failed.")

if __name__ == "__main__":
    main()
