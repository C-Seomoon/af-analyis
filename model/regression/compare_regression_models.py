import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer plots
import numpy as np
import traceback

# Define standard metrics to look for (adjust if needed based on calculate_regression_metrics output)
METRIC_KEYS_REGRESSION = ['r2', 'mse', 'rmse', 'mae']

def load_metrics(model_dir):
    """Loads metrics summary from a model's result directory."""
    metrics_file = os.path.join(model_dir, "metrics_summary.json")
    model_name = os.path.basename(model_dir) # Get model name from directory name

    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            # Add model name for easy identification
            metrics_data['model_name'] = model_name
            return metrics_data
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {metrics_file}")
            return None
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            return None
    else:
        # print(f"Metrics summary file not found in {model_dir}") # Be less verbose
        return None

def plot_comparison(df, metric_base, results_dir, add_error_bars=True):
    """
    Generates and saves a comparison vertical bar plot for a given metric.

    Args:
        df (pd.DataFrame): DataFrame containing model comparison metrics.
        metric_base (str): The base name of the metric to plot (e.g., 'r2', 'rmse').
        results_dir (str): Directory to save the plot.
        add_error_bars (bool): Whether to add standard deviation error bars. Defaults to True.
    """
    mean_col = f"{metric_base}_mean"
    std_col = f"{metric_base}_std"

    if mean_col not in df.columns:
        print(f"Metric '{mean_col}' not found in summary data. Skipping plot.")
        return

    # Check for std dev column only if error bars are requested
    std_exists = std_col in df.columns
    if add_error_bars and not std_exists:
        print(f"Standard deviation column '{std_col}' not found. Plotting means only.")
        add_error_bars = False # Disable error bars if std dev data is missing

    # Fill potential NaNs in std_col with 0 if it exists and we need error bars
    if add_error_bars and std_exists:
        df[std_col] = df[std_col].fillna(0)
    elif not std_exists: # Ensure std_col exists with 0 if not adding error bars but needed later
        df = df.assign(**{std_col: 0})


    # Sort by mean metric value (ascending for errors like RMSE/MAE, descending for R2)
    ascending_sort = metric_base not in ['r2'] # Sort R2 descending, others ascending
    df_sorted = df.sort_values(by=mean_col, ascending=ascending_sort)

    plt.figure(figsize=(10, 7)) # Adjusted size for vertical plot
    # Use seaborn barplot for the bars based on mean values
    # Address FutureWarning: assign x to hue, set legend=False for vertical plot
    try:
        # 1. Create the vertical bars using Seaborn
        ax = sns.barplot(x='model_name', y=mean_col, data=df_sorted, palette='viridis',
                         hue='model_name', legend=False, dodge=False, errorbar=None) # errorbar=None disables default CI

        # 2. Add error bars using matplotlib's errorbar function (optional)
        if add_error_bars:
            x_coords = np.arange(len(df_sorted))
            means = df_sorted[mean_col].values
            stds = df_sorted[std_col].values # NaNs already filled if std_exists

            # Add error bars vertically
            ax.errorbar(x=x_coords, y=means, yerr=stds, fmt='none', # 'none' means don't plot the data points
                        color='darkgrey', capsize=4, elinewidth=1.5, capthick=1.5) # Customize appearance
            y_label_text = f'Mean {metric_base.upper()} (+/- Std Dev)'
        else:
            y_label_text = f'Mean {metric_base.upper()}'


        plt.title(f'Model Comparison: Mean {metric_base.upper()} (Nested CV)')
        plt.xlabel('Model')
        plt.ylabel(y_label_text)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.6) # Add horizontal grid lines

        # Add value labels to bars (adjust logic for vertical bars)
        for i, (_, row) in enumerate(df_sorted.iterrows()):
             mean_val = row[mean_col]
             std_val = row[std_col] # Already filled NaN with 0 if needed

             # Calculate position slightly above the bar or error bar cap
             label_y_offset = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0]) # Small relative offset based on y-axis range
             if pd.notna(mean_val):
                 y_pos = mean_val + (std_val if add_error_bars else 0) + label_y_offset
                 ha = 'center'
                 va = 'bottom' # Place text above the bar/error bar

                 # Adjust position for potentially very low values close to 0 if needed
                 # if y_pos < ax.get_ylim()[0] + label_y_offset * 2:
                 #     y_pos = ax.get_ylim()[0] + label_y_offset * 2 # Prevent overlap with axis

                 plt.text(i, y_pos, f'{mean_val:.3f}', color='black', va=va, ha=ha, fontsize=9)


        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"comparison_{metric_base}_mean_vertical.png") # Indicate vertical in filename
        plt.savefig(plot_filename)
        print(f"Comparison plot saved to: {plot_filename}")
        plt.close()

    except Exception as e:
         print(f"Error generating plot for metric {metric_base}: {e}")
         print(traceback.format_exc()) # Print full traceback for debugging
         plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare regression model performance from Nested CV results.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the main results directory containing subdirectories for each model.")
    parser.add_argument("--sort_by", type=str, default="rmse_mean",
                        help="Metric column to sort the final comparison table by (e.g., 'r2_mean', 'rmse_mean').")

    args = parser.parse_args()

    print(f"--- Comparing Models in Directory: {args.results_dir} ---")

    all_metrics = []
    # Iterate through items in the results directory
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        # Check if it's a directory and potentially a model result folder
        if os.path.isdir(item_path):
            print(f"Checking directory: {item}")
            metrics = load_metrics(item_path)
            if metrics:
                all_metrics.append(metrics)
            else:
                print(f"Could not load metrics from {item_path} or it's not a model results directory.")

    if not all_metrics:
        print("Error: No model metrics summaries found in the specified directory.")
        return

    # Convert list of dictionaries to DataFrame
    comparison_df = pd.DataFrame(all_metrics)

    # Basic check if expected columns exist
    if 'model_name' not in comparison_df.columns:
         print("Error: 'model_name' column not found in loaded metrics.")
         return

    # Reorder columns for better readability (put model name first, then metrics)
    metric_cols_ordered = []
    for key in METRIC_KEYS_REGRESSION:
         mean_col = f"{key}_mean"
         std_col = f"{key}_std"
         if mean_col in comparison_df.columns: metric_cols_ordered.append(mean_col)
         if std_col in comparison_df.columns: metric_cols_ordered.append(std_col)

    # Add any other columns present (like best_inner_cv_score_mean etc.)
    other_cols = [col for col in comparison_df.columns if col not in ['model_name'] + metric_cols_ordered]
    final_cols = ['model_name'] + metric_cols_ordered + sorted(other_cols)

    # Filter DataFrame to only include existing columns
    final_cols_exist = [col for col in final_cols if col in comparison_df.columns]
    comparison_df = comparison_df[final_cols_exist]


    # Sort DataFrame
    sort_col = args.sort_by
    if sort_col not in comparison_df.columns:
        print(f"Warning: Sort column '{sort_col}' not found. Using 'model_name' for sorting.")
        sort_col = 'model_name'
        ascending_sort = True
    else:
         # Sort R2 descending, others ascending by default
        ascending_sort = not ('r2' in sort_col)

    try:
        comparison_df = comparison_df.sort_values(by=sort_col, ascending=ascending_sort).reset_index(drop=True)
    except Exception as e:
        print(f"Error sorting DataFrame by {sort_col}: {e}. Sorting by model_name.")
        comparison_df = comparison_df.sort_values(by='model_name').reset_index(drop=True)


    # Save comparison table
    output_csv_path = os.path.join(args.results_dir, "regression_models_comparison.csv")
    try:
        comparison_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"\nComparison table saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving comparison CSV: {e}")

    # Print results to console
    print("\n--- Model Comparison Summary ---")
    # Use pandas display options for better console output
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(comparison_df)

    # Generate comparison plots for standard metrics
    print("\n--- Generating Comparison Plots ---")
    for metric in METRIC_KEYS_REGRESSION:
        plot_comparison(comparison_df, metric, args.results_dir)

    print("\n--- Comparison Complete ---")

if __name__ == "__main__":
    main()
