import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score

def load_all_model_metrics(result_dirs):
    """모든 모델의 성능 지표 로드"""
    all_metrics = []
    
    for result_dir in result_dirs:
        # 결과 디렉토리 내의 모든 모델 폴더 찾기
        model_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
        
        for model_name in model_dirs:
            model_path = os.path.join(result_dir, model_name)
            metrics_file = os.path.join(model_path, "metrics_summary.json")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['model'] = model_name
                    metrics['result_dir'] = result_dir
                    all_metrics.append(metrics)
    
    if all_metrics:
        return pd.DataFrame(all_metrics)
    else:
        return None

def load_model_predictions(result_dirs):
    """모든 모델의 예측 결과 로드"""
    all_preds = []
    
    for result_dir in result_dirs:
        model_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
        
        for model_name in model_dirs:
            model_path = os.path.join(result_dir, model_name)
            pred_file = os.path.join(model_path, "all_predictions.csv")
            
            if os.path.exists(pred_file):
                preds = pd.read_csv(pred_file)
                preds['model'] = model_name
                preds['result_dir'] = result_dir
                all_preds.append(preds)
    
    if all_preds:
        return pd.concat(all_preds, axis=0, ignore_index=True)
    else:
        return None

def plot_roc_curves(predictions_df, output_dir):
    """모든 모델의 ROC 커브 플롯"""
    plt.figure(figsize=(10, 8))
    
    models = predictions_df['model'].unique()
    for model in models:
        model_preds = predictions_df[predictions_df['model'] == model]
        
        # 모든 폴드의 평균 ROC 커브 계산
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for fold in model_preds['fold'].unique():
            fold_data = model_preds[model_preds['fold'] == fold]
            fpr, tpr, _ = roc_curve(fold_data['y_true'], fold_data['y_prob'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, lw=2, 
                 label=f'{model} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'combined_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(predictions_df, output_dir):
    """모든 모델의 Precision-Recall 커브 플롯"""
    plt.figure(figsize=(10, 8))
    
    models = predictions_df['model'].unique()
    for model in models:
        model_preds = predictions_df[predictions_df['model'] == model]
        
        # 모든 폴드의 평균 PR 커브 계산
        precisions = []
        recalls = []
        pr_aucs = []
        mean_recall = np.linspace(0, 1, 100)
        
        for fold in model_preds['fold'].unique():
            fold_data = model_preds[model_preds['fold'] == fold]
            precision, recall, _ = precision_recall_curve(fold_data['y_true'], fold_data['y_prob'])
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)
            
            # 보간 (precision은 recall이 증가할 때 감소하므로 역순으로 보간)
            precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        
        mean_precision = np.mean(precisions, axis=0)
        mean_pr_auc = np.mean(pr_aucs)
        std_pr_auc = np.std(pr_aucs)
        
        plt.plot(mean_recall, mean_precision, lw=2, 
                 label=f'{model} (PR AUC = {mean_pr_auc:.3f} ± {std_pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'combined_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def load_all_cv_metrics(base_results_dir: Path):
    """Load mean and std metrics from metrics_summary.json for all models."""
    all_metrics_data = []
    model_dirs = [d for d in base_results_dir.iterdir() if d.is_dir() and 
                   not d.name.startswith('fold_') and 
                   not d.name.endswith('_shap_analysis') and
                   d.name != 'comparison_results']
                   
    if not model_dirs:
        print(f"Error: No model subdirectories found in {base_results_dir}.")
        return None, None

    print(f"Found model directories: {[d.name for d in model_dirs]}")
    
    processed_metrics_list = [] 

    for model_dir in model_dirs:
        model_name = model_dir.name
        metrics_file = model_dir / "metrics_summary.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics_summary = json.load(f)
                
                # Ensure model name is stored if not already present
                if 'model' not in metrics_summary:
                    metrics_summary['model'] = model_name 
                all_metrics_data.append(metrics_summary)

                # Prepare data for plotting (long format)
                for key, value in metrics_summary.items():
                    # Check if the key represents a mean value and is not None
                    if key.endswith('_mean') and value is not None: 
                        base_metric = key.replace('_mean', '')
                        std_key = key.replace('_std', '')
                        # Get std value, default to 0 if key missing or value is None
                        std_value = metrics_summary.get(std_key, 0)
                        if std_value is None: std_value = 0 

                        # Use more readable metric names
                        readable_metric = base_metric.replace('_', ' ').title()
                        # Standardize specific metric names
                        if readable_metric == 'Roc Auc': readable_metric = 'ROC AUC'
                        if readable_metric == 'Pr Auc': readable_metric = 'PR AUC'
                        if readable_metric == 'F1': readable_metric = 'F1 Score' # Ensure consistent naming
                        if readable_metric == 'Mcc': readable_metric = 'MCC'
                        if readable_metric == 'Balanced Accuracy': readable_metric = 'Balanced Accuracy' # Already good

                        processed_metrics_list.append({
                            'Model': model_name,
                            'Metric': readable_metric,
                            'Mean': value,
                            'Std': std_value
                        })

            except Exception as e:
                print(f"Error loading or processing {metrics_file}: {e}")
        else:
            print(f"Warning: metrics_summary.json not found for model {model_name}")

    if not all_metrics_data:
        return None, None

    metrics_raw_df = pd.DataFrame(all_metrics_data)
    metrics_plot_df = pd.DataFrame(processed_metrics_list)
    
    return metrics_raw_df, metrics_plot_df

def load_all_predictions(model_dir: Path):
    """Load and concatenate predictions from all folds for a given model."""
    all_preds_df = []
    prediction_files = sorted(model_dir.rglob('fold_*/predictions.csv')) 
    
    if not prediction_files:
        print(f"Warning: No 'predictions.csv' found in {model_dir}")
        return None

    # print(f"Found {len(prediction_files)} prediction files for {model_dir.name}")    
    for pred_file in prediction_files:
        try:
            fold_df = pd.read_csv(pred_file, index_col=0) 
            if {'y_true', 'y_pred', 'y_prob'}.issubset(fold_df.columns):
                 all_preds_df.append(fold_df)
            else: 
                 print(f"Warning: Skipping {pred_file}. Missing required columns.")
                 # Attempt to load even if y_pred is missing, if y_prob exists for curves
                 if {'y_true', 'y_prob'}.issubset(fold_df.columns) and 'y_pred' not in fold_df.columns:
                     print(f"  -> Proceeding with y_true, y_prob for curve plotting if needed.")
                     fold_df['y_pred'] = (fold_df['y_prob'] >= 0.5).astype(int) # Generate y_pred
                     all_preds_df.append(fold_df)

        except Exception as e: 
            print(f"Error loading {pred_file}: {e}")            
            
    if not all_preds_df: 
        print(f"Error: Could not load any valid prediction data for {model_dir.name}.")
        return None      
          
    combined_df = pd.concat(all_preds_df, ignore_index=False) 
    if combined_df.index.duplicated().any(): 
         print(f"Warning: Duplicate indices found for {model_dir.name}. Using first occurrence.")
         combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
         
    # print(f"Combined predictions for {model_dir.name}. Shape: {combined_df.shape}")
    return combined_df

def plot_cv_metric_comparison_means(metrics_plot_df, output_dir):
    """
    Plot grouped bar chart comparing key CV metrics (mean values only) across models.
    Includes MCC. Uses Seaborn for grouped bars.
    """
    if metrics_plot_df is None or metrics_plot_df.empty:
        print("No data available for plotting CV metric comparison (means only).")
        return

    # Define Metrics and Order
    key_metrics_readable = ['ROC AUC', 'PR AUC', 'F1 Score', 'Balanced Accuracy', 'MCC', 'Precision', 'Recall', 'Accuracy']

    plot_df_filtered = metrics_plot_df[metrics_plot_df['Metric'].isin(key_metrics_readable)].copy()

    if plot_df_filtered.empty:
        print(f"No data found for the specified key metrics: {key_metrics_readable}")
        return

    # Ensure 'Metric' column is categorical for ordering
    plot_df_filtered['Metric'] = pd.Categorical(plot_df_filtered['Metric'], categories=key_metrics_readable, ordered=True)
    plot_df_filtered.dropna(subset=['Mean'], inplace=True)
    plot_df_filtered.sort_values(['Metric', 'Model'], inplace=True)

    # Actual metrics present after filtering NaNs
    metrics_present = plot_df_filtered['Metric'].unique()
    if len(metrics_present) == 0:
        print("No valid metrics left to plot.")
        return

    plt.figure(figsize=(max(12, len(metrics_present) * 1.5), 7))
    # Use seaborn's barplot for easy grouping by hue (Model)
    sns.barplot(data=plot_df_filtered, x='Metric', y='Mean', hue='Model', palette='viridis') # Use 'Mean' column

    plt.title('Nested CV Performance Comparison (Mean Values)')
    plt.ylabel('Score (Mean)')
    plt.xlabel('Metric')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust y-limits based on mean values
    all_means = plot_df_filtered['Mean']
    if not all_means.empty:
        min_val = all_means.min()
        max_val = all_means.max()
        # Consider MCC range [-1, 1]
        y_min = min(0, min_val - 0.05) if min_val < 0 else 0
        y_max = max(1.0, max_val + 0.05)
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_path = output_dir / "cv_metrics_comparison_means.png" # Updated filename
    plt.savefig(plot_path)
    plt.close()
    print(f"CV Metrics comparison plot (Means only) saved to: {plot_path}")

def plot_overall_roc_curves(roc_data, output_dir):
    """Plot ROC curves for all models based on overall combined predictions."""
    plt.figure(figsize=(8, 8))
    plot_occurred = False
    for model_name, data in roc_data.items():
        if data.get('fpr') is not None and data.get('tpr') is not None and data.get('auc') is not None:
            plt.plot(data['fpr'], data['tpr'], lw=2, label=f'{model_name} (AUC = {data["auc"]:.3f})')
            plot_occurred = True
        else: 
            print(f"Skipping ROC plot for {model_name} due to missing data.")

    if not plot_occurred:
        print("No data available to plot overall ROC curves.")
        plt.close()
        return

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Overall Data'); plt.legend(loc="lower right"); plt.grid(alpha=0.5)    
    plot_path = output_dir / "overall_roc_curves.png" 
    plt.savefig(plot_path); plt.close()
    print(f"Overall ROC curve plot saved to: {plot_path}")

def plot_overall_pr_curves(pr_data, output_dir):
    """Plot Precision-Recall curves based on overall combined predictions."""
    plt.figure(figsize=(8, 8))
    plot_occurred = False
    for model_name, data in pr_data.items():
        if data.get('precision') is not None and data.get('recall') is not None and data.get('auc') is not None:
            plt.plot(data['recall'], data['precision'], lw=2, label=f'{model_name} (AUC = {data["auc"]:.3f})')
            plot_occurred = True
        else: 
             print(f"Skipping PR plot for {model_name} due to missing data.")
             
    if not plot_occurred:
        print("No data available to plot overall PR curves.")
        plt.close()
        return

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Overall Data'); plt.legend(loc="lower left"); plt.grid(alpha=0.5)
    plot_path = output_dir / "overall_pr_curves.png" 
    plt.savefig(plot_path); plt.close()
    print(f"Overall PR curve plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare Classification Model Results from Nested CV Framework.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the base directory containing model result subdirectories.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save comparison results. Defaults to a "comparison" subdirectory within results_dir.')
    
    args = parser.parse_args()

    base_results_dir = Path(args.results_dir)
    if not base_results_dir.is_dir():
        print(f"Error: Base results directory not found: {base_results_dir}")
        return

    if args.output_dir:
        comparison_output_dir = Path(args.output_dir)
    else:
        comparison_output_dir = base_results_dir / "comparison_results"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparison results will be saved to: {comparison_output_dir}")
    
    # --- Load CV Metrics ---
    metrics_raw_df, metrics_plot_df = load_all_cv_metrics(base_results_dir)

    if metrics_raw_df is not None:
        print("\n--- Nested CV Metrics Summary (Raw) ---")
        metrics_raw_df_display = metrics_raw_df.set_index('model') 
        print(metrics_raw_df_display)
        cv_summary_raw_path = comparison_output_dir / "cv_metrics_summary_raw.csv"
        try:
             metrics_raw_df_display.to_csv(cv_summary_raw_path)
             print(f"Raw CV metrics summary saved to: {cv_summary_raw_path}")
        except Exception as e:
             print(f"Error saving raw CV metrics summary: {e}")

        # --- Create and Save Formatted Metrics Summary (Mean ± Std) ---
        print("\n--- Formatting Metrics as Mean ± Std ---")
        formatted_metrics = metrics_raw_df.copy()
        # Use model name from the DataFrame if present, otherwise fallback to index if needed
        if 'model' in formatted_metrics.columns:
            formatted_metrics.set_index('model', inplace=True) 

        metric_columns = [col for col in formatted_metrics.columns if '_mean' in col]
        formatted_cols_dict = {} # Store formatted series temporarily

        for mean_col in metric_columns:
            std_col = mean_col.replace('_mean', '_std')
            base_metric_name = mean_col.replace('_mean', '')

            if std_col in formatted_metrics.columns:
                # Apply formatting, handle None/NaN values gracefully
                formatted_series = formatted_metrics.apply(
                    lambda row: f"{row[mean_col]:.3f} ± {row[std_col]:.3f}" if pd.notna(row[mean_col]) and pd.notna(row[std_col]) else \
                                (f"{row[mean_col]:.3f}" if pd.notna(row[mean_col]) else "N/A"),
                    axis=1
                )
                formatted_cols_dict[base_metric_name] = formatted_series
            else:
                # If only mean exists, just format mean
                 formatted_series = formatted_metrics.apply(
                    lambda row: f"{row[mean_col]:.3f}" if pd.notna(row[mean_col]) else "N/A",
                    axis=1
                )
                 formatted_cols_dict[base_metric_name] = formatted_series
                 
        # Create the final formatted DataFrame from the dictionary
        formatted_metrics_final = pd.DataFrame(formatted_cols_dict)

        # Reorder columns for better readability
        metric_order_preference = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc', 'roc_auc', 'pr_auc']
        # Use title case for display matching the plot if desired
        metric_order_preference_title = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy', 'MCC', 'ROC AUC', 'PR AUC']
        
        # Find columns present in the formatted data
        present_cols_base = [col for col in metric_order_preference if col in formatted_metrics_final.columns]
        other_cols = [col for col in formatted_metrics_final.columns if col not in present_cols_base]
        
        # Reorder using base names
        formatted_metrics_final = formatted_metrics_final[present_cols_base + other_cols]
        # Rename columns to Title Case for final CSV
        formatted_metrics_final.columns = [col.replace('_',' ').title().replace('Roc Auc','ROC AUC').replace('Pr Auc','PR AUC').replace('Mcc','MCC').replace('F1','F1 Score') for col in formatted_metrics_final.columns]


        print("\n--- CV Metrics Summary (Formatted as Mean ± Std) ---")
        print(formatted_metrics_final)
        formatted_summary_path = comparison_output_dir / "cv_metrics_summary_formatted.csv"
        try:
            formatted_metrics_final.to_csv(formatted_summary_path)
            print(f"Formatted CV metrics summary saved to: {formatted_summary_path}")
        except Exception as e:
            print(f"Error saving formatted CV metrics summary: {e}")

        # --- Create CV metrics comparison plot (Means only) ---
        # Ensure metrics_plot_df (long format) is available
        if metrics_plot_df is not None:
             plot_cv_metric_comparison_means(metrics_plot_df, comparison_output_dir)
        else:
             print("Plotting DataFrame (long format) was not created. Skipping means plot.")

    else:
        print("\nNo CV metrics loaded. Skipping CV comparison plot and formatted summary generation.")
        
    # --- Load Overall Predictions and Plot Overall Curves ---
    print("\n--- Processing Overall Predictions for ROC/PR Curves ---")
    roc_plot_data_overall = {}
    pr_plot_data_overall = {}
    
    # Find model directories again
    model_dirs = [d for d in base_results_dir.iterdir() if d.is_dir() and 
                   not d.name.startswith('fold_') and 
                   not d.name.endswith('_shap_analysis') and
                   d.name != 'comparison_results']

    if not model_dirs:
         print("No model directories found to process for overall curves.")
    else:
         for model_dir in model_dirs:
             model_name = model_dir.name
             # print(f"Loading combined predictions for {model_name}...") # Reduce verbosity
             combined_preds = load_all_predictions(model_dir)
             
             if combined_preds is not None and 'y_true' in combined_preds and 'y_prob' in combined_preds:
                 y_true_overall = combined_preds['y_true']
                 y_prob_overall = combined_preds['y_prob']
                 
                 # Calculate ROC AUC for overall data
                 try:
                      overall_roc_auc = roc_auc_score(y_true_overall, y_prob_overall)
                      fpr, tpr, _ = roc_curve(y_true_overall, y_prob_overall)
                      roc_plot_data_overall[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': overall_roc_auc}
                 except Exception as e:
                      print(f"Could not calculate overall ROC for {model_name}: {e}")
                      roc_plot_data_overall[model_name] = {} # Empty dict if fails

                 # Calculate PR AUC for overall data
                 try:
                      overall_pr_auc = average_precision_score(y_true_overall, y_prob_overall)
                      precision, recall, _ = precision_recall_curve(y_true_overall, y_prob_overall)
                      pr_plot_data_overall[model_name] = {'precision': precision, 'recall': recall, 'auc': overall_pr_auc}
                 except Exception as e:
                      print(f"Could not calculate overall PR for {model_name}: {e}")
                      pr_plot_data_overall[model_name] = {} # Empty dict if fails
             else:
                 print(f"Skipping overall curve calculation for {model_name} due to missing prediction data or columns.")
                 roc_plot_data_overall[model_name] = {} 
                 pr_plot_data_overall[model_name] = {}

         # Plot overall curves if any valid data was processed
         if any(roc_plot_data_overall.values()): # Check if any dict is non-empty
             plot_overall_roc_curves(roc_plot_data_overall, comparison_output_dir)
         else:
             print("No valid data for overall ROC curve plotting.")
             
         if any(pr_plot_data_overall.values()):
             plot_overall_pr_curves(pr_plot_data_overall, comparison_output_dir)
         else:
              print("No valid data for overall PR curve plotting.")

    print("\n--- Model Comparison Script Finished ---")

if __name__ == "__main__":
    main()
