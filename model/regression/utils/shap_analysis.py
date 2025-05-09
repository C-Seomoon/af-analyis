import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib # To save/load numpy arrays efficiently
import warnings
import traceback

plt.rcParams['figure.max_open_warning'] = 50 # Allow more figures open

def save_shap_results_regression(shap_values, X_explain, feature_names, output_dir, fold_num=None, expected_value=None):
    """
    Saves SHAP values and generates SHAP summary plots for a single fold in regression.
    Assumes shap_values is a pandas DataFrame or None.

    Args:
        shap_values (pd.DataFrame or None): SHAP values DataFrame (n_samples, n_features).
        X_explain (pd.DataFrame): Data used for SHAP explanation (aligned with shap_values).
        feature_names (list): List of feature names.
        output_dir (str): Directory to save the results.
        fold_num (int, optional): Fold number for filenames. Defaults to None.
        expected_value (float or np.ndarray, optional): The base value(s) (explainer's expected value). Defaults to None.
    """
    print(f"--- Saving SHAP results for Fold {fold_num if fold_num else 'N/A'} ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Validate inputs --- 
    if shap_values is None:
        print("Error: shap_values is None. Cannot save or plot SHAP results.")
        return
    if not isinstance(shap_values, pd.DataFrame):
        print(f"Error: Expected shap_values to be a pandas DataFrame, but got {type(shap_values)}. Cannot proceed.")
        return
    if shap_values.shape[1] != len(feature_names):
         print(f"Warning: SHAP values columns ({shap_values.shape[1]}) != number of feature names ({len(feature_names)}). Check consistency.")

    # --- 1. Save SHAP values and related info --- 
    shap_values_filename = f"shap_values_fold_{fold_num}.npy" if fold_num else "shap_values.npy"
    shap_values_path = os.path.join(output_dir, shap_values_filename)
    try:
        np.save(shap_values_path, shap_values.values)
        print(f"SHAP values saved to: {shap_values_path}")
    except Exception as e:
        print(f"Error saving SHAP values to {shap_values_path}: {e}")
        return

    x_explain_filename = f"X_explain_fold_{fold_num}.csv" if fold_num else "X_explain.csv"
    x_explain_path = os.path.join(output_dir, x_explain_filename)
    try:
        if list(X_explain.columns) != feature_names:
             print("Warning: X_explain columns don't match feature_names list. Reassigning columns.")
             X_explain.columns = feature_names
        X_explain.to_csv(x_explain_path, index=False)
        print(f"X_explain saved to (CSV): {x_explain_path}")
    except Exception as e:
        print(f"Error saving X_explain to CSV {x_explain_path}: {e}")

    if expected_value is not None:
        ev_filename = f"shap_expected_value_fold_{fold_num}.joblib" if fold_num else "shap_expected_value.joblib"
        ev_path = os.path.join(output_dir, ev_filename)
        try:
            joblib.dump(expected_value, ev_path)
            print(f"SHAP expected value saved to: {ev_path}")
        except Exception as e:
            print(f"Error saving SHAP expected value to {ev_path}: {e}")

    # --- 2. Generate and Save SHAP Summary Plots --- 
    plot_filenames = {
        'bar': f"shap_summary_bar_fold_{fold_num}.png" if fold_num else "shap_summary_bar.png",
        'dot': f"shap_summary_dot_fold_{fold_num}.png" if fold_num else "shap_summary_dot.png"
    }

    # Dot plot (preferred over default beeswarm for many features)
    dot_plot_path = os.path.join(output_dir, plot_filenames['dot'])
    try:
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        shap.summary_plot(
            shap_values.values,
            features=X_explain,
            feature_names=feature_names,
            show=False,
            plot_type='dot'
        )
        plt.title(f'SHAP Summary (Dot Plot) - Fold {fold_num}')
        plt.tight_layout()
        plt.savefig(dot_plot_path)
        plt.close()
        print(f"SHAP dot plot saved to: {dot_plot_path}")
    except Exception as e:
        print(f"Error generating SHAP dot plot: {e}")
        print(traceback.format_exc())
        plt.close()

    # Bar plot (mean absolute SHAP value)
    bar_plot_path = os.path.join(output_dir, plot_filenames['bar'])
    try:
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        shap.summary_plot(
            shap_values.values,
            features=X_explain,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title(f'SHAP Feature Importance (Mean Abs) - Fold {fold_num}')
        plt.tight_layout()
        plt.savefig(bar_plot_path)
        plt.close()
        print(f"SHAP bar plot saved to: {bar_plot_path}")
    except Exception as e:
        print(f"Error generating SHAP bar plot: {e}")
        print(traceback.format_exc())
        plt.close()

    # (Optional) Dependence Plots for top N features
    # try:
    #     # Calculate mean abs SHAP to find top features
    #     mean_abs_shap = np.abs(shap_values).mean(axis=0)
    #     feature_order = np.argsort(mean_abs_shap)[::-1]
    #     top_n = min(10, len(feature_names)) # Plot top 10 or fewer
    #     print(f"\nGenerating dependence plots for top {top_n} features...")
    #     dep_plot_dir = os.path.join(output_dir, "dependence_plots")
    #     os.makedirs(dep_plot_dir, exist_ok=True)

    #     for i in range(top_n):
    #         feature_idx = feature_order[i]
    #         feature_name = feature_names[feature_idx]
    #         try:
    #             plt.figure() # Create new figure for each plot
    #             shap.dependence_plot(
    #                 feature_idx,
    #                 shap_values,
    #                 X_explain,
    #                 feature_names=feature_names,
    #                 interaction_index=None, # Or 'auto' to find interaction
    #                 show=False
    #             )
    #             plt.title(f'SHAP Dependence Plot: {feature_name} - Fold {fold_num}')
    #             plt.tight_layout()
    #             plot_path = os.path.join(dep_plot_dir, f"shap_dependence_{feature_name}_fold_{fold_num}.png")
    #             plt.savefig(plot_path)
    #             plt.close()
    #         except Exception as inner_e:
    #              print(f"Error generating dependence plot for {feature_name}: {inner_e}")
    #              plt.close() # Ensure plot is closed

    # except Exception as e:
    #     print(f"Error generating dependence plots: {e}")
    #     print(traceback.format_exc())
    #     plt.close()

    print(f"--- SHAP saving completed for Fold {fold_num if fold_num else 'N/A'} ---")


def analyze_global_shap_regression(model_results_dir):
    """
    Analyzes aggregated SHAP results across all folds for a regression model.

    Args:
        model_results_dir (str): Path to the directory containing fold results
                                 (e.g., .../output_dir/xgb/). It should contain
                                 subdirectories like 'fold_1', 'fold_2', etc.
                                 Each fold directory must contain:
                                  - shap_values_fold_N.npy
                                  - X_explain_fold_N.csv
    """
    print(f"\n--- Analyzing Global SHAP for Model: {os.path.basename(model_results_dir)} ---")
    all_shap_values = []
    all_X_explain = []
    feature_names = None
    num_folds_found = 0

    # --- 1. Load SHAP values and X_explain from each fold ---
    for item in sorted(os.listdir(model_results_dir)):
        fold_dir = os.path.join(model_results_dir, item)
        if os.path.isdir(fold_dir) and item.startswith('fold_'):
            fold_num = item.split('_')[-1]
            shap_file = os.path.join(fold_dir, f"shap_values_fold_{fold_num}.npy")
            x_file = os.path.join(fold_dir, f"X_explain_fold_{fold_num}.csv")

            if os.path.exists(shap_file) and os.path.exists(x_file):
                print(f"Loading data from {item}...")
                try:
                    shap_vals = np.load(shap_file)
                    x_explain_df = pd.read_csv(x_file)

                    if feature_names is None:
                        feature_names = x_explain_df.columns.tolist()
                    elif feature_names != x_explain_df.columns.tolist():
                        print(f"Warning: Feature names mismatch in fold {fold_num}. Skipping fold.")
                        continue # Skip fold if features don't match

                    if shap_vals.shape[0] == x_explain_df.shape[0] and shap_vals.shape[1] == len(feature_names):
                        all_shap_values.append(shap_vals)
                        all_X_explain.append(x_explain_df)
                        num_folds_found += 1
                    else:
                        print(f"Warning: Shape mismatch between SHAP values {shap_vals.shape} and X_explain {x_explain_df.shape} in fold {fold_num}. Skipping.")

                except Exception as e:
                    print(f"Error loading data from {item}: {e}")
            else:
                print(f"SHAP values or X_explain file missing in {item}. Skipping.")

    if num_folds_found == 0 or not all_shap_values or not all_X_explain:
        print("Error: No valid SHAP data found across folds. Cannot perform global analysis.")
        return

    print(f"Loaded SHAP data from {num_folds_found} folds.")

    # --- 2. Concatenate results ---
    try:
        global_shap_values = np.concatenate(all_shap_values, axis=0)
        global_X_explain = pd.concat(all_X_explain, axis=0).reset_index(drop=True)
        print(f"Combined SHAP values shape: {global_shap_values.shape}")
        print(f"Combined X_explain shape: {global_X_explain.shape}")

        # Save combined data
        combined_dir = os.path.join(model_results_dir, "combined_shap")
        os.makedirs(combined_dir, exist_ok=True)
        np.save(os.path.join(combined_dir, "combined_shap_values.npy"), global_shap_values)
        global_X_explain.to_csv(os.path.join(combined_dir, "combined_X_explain.csv"), index=False)
        print(f"Combined SHAP data saved to (CSV for X_explain): {combined_dir}")

    except Exception as e:
        print(f"Error concatenating SHAP results: {e}")
        return


    # --- 3. Calculate and Save Global Mean Absolute SHAP ---
    try:
        mean_abs_shap = np.abs(global_shap_values).mean(axis=0)
        global_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)

        importance_path = os.path.join(combined_dir, "global_shap_feature_importance.csv")
        global_importance_df.to_csv(importance_path, index=False)
        print(f"Global SHAP feature importance saved to: {importance_path}")
        print("\nTop 10 Features by Global Mean Absolute SHAP:")
        print(global_importance_df.head(10))

    except Exception as e:
        print(f"Error calculating global SHAP importance: {e}")
        # Continue to plotting if possible


    # --- 4. Generate and Save Global Summary Plots ---
    plot_filenames = {
        'beeswarm': "combined_shap_summary_beeswarm.png",
        'bar': "combined_shap_summary_bar.png",
        'dot': "combined_shap_summary_dot.png"
    }

    # Combined Dot plot
    dot_plot_path = os.path.join(combined_dir, plot_filenames['dot'])
    try:
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        shap.summary_plot(
            global_shap_values,
            features=global_X_explain,
            feature_names=feature_names,
            show=False,
            plot_type='dot'
        )
        plt.title(f'Combined SHAP Summary (Dot Plot) - All Folds')
        plt.tight_layout()
        plt.savefig(dot_plot_path)
        plt.close()
        print(f"Combined SHAP dot plot saved to: {dot_plot_path}")
    except Exception as e:
        print(f"Error generating combined SHAP dot plot: {e}")
        print(traceback.format_exc())
        plt.close()

    # Combined Bar plot
    bar_plot_path = os.path.join(combined_dir, plot_filenames['bar'])
    try:
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
        shap.summary_plot(
            global_shap_values,
            features=global_X_explain,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title(f'Combined SHAP Feature Importance (Mean Abs) - All Folds')
        plt.tight_layout()
        plt.savefig(bar_plot_path)
        plt.close()
        print(f"Combined SHAP bar plot saved to: {bar_plot_path}")
    except Exception as e:
        print(f"Error generating combined SHAP bar plot: {e}")
        print(traceback.format_exc())
        plt.close()

    # (Optional) Global Dependence Plots
    # Similar loop as in save_shap_results_regression, but using global data
    # dep_plot_dir = os.path.join(combined_dir, "dependence_plots")
    # os.makedirs(dep_plot_dir, exist_ok=True)
    # top_n = min(10, len(feature_names))
    # feature_order = global_importance_df['feature'].tolist()[:top_n]
    # for feature_name in feature_order:
    #      try:
    #          plt.figure()
    #          shap.dependence_plot(
    #              feature_name, # Can use feature name directly with DataFrames
    #              global_shap_values,
    #              global_X_explain,
    #              # feature_names=feature_names, # Not needed if using string index
    #              interaction_index=None, # Or 'auto'
    #              show=False
    #          )
    #          plt.title(f'Combined SHAP Dependence Plot: {feature_name}')
    #          plt.tight_layout()
    #          plot_path = os.path.join(dep_plot_dir, f"combined_shap_dependence_{feature_name}.png")
    #          plt.savefig(plot_path)
    #          plt.close()
    #      except Exception as inner_e:
    #          print(f"Error generating combined dependence plot for {feature_name}: {inner_e}")
    #          plt.close()

    print(f"--- Global SHAP analysis completed for: {os.path.basename(model_results_dir)} ---")
