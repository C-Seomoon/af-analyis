import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates standard regression metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: Dictionary containing regression metrics (R2, MSE, RMSE, MAE).
              Returns empty dict if input arrays are invalid.
    """
    metrics = {}
    try:
        # Ensure inputs are array-like and have the same length
        if not hasattr(y_true, '__len__') or not hasattr(y_pred, '__len__') or len(y_true) != len(y_pred):
            print("Error: Invalid input arrays for metric calculation.")
            return {}
        if len(y_true) == 0:
            print("Warning: Empty arrays provided for metric calculation.")
            return {}
            
        # Handle potential NaNs or infinite values if they slipped through
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.all(valid_indices):
            print(f"Warning: Found non-finite values. Calculating metrics on {np.sum(valid_indices)} finite samples.")
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            if len(y_true) == 0:
                 print("Error: No finite samples left after filtering.")
                 return {}

        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        print(f"Calculated Metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    except Exception as e:
        print(f"Error calculating regression metrics: {e}")
        # Return partially filled metrics or empty dict in case of error
        return {}
        
    return metrics

def save_results(results_dict, output_dir, filename="results.json"):
    """
    Saves a dictionary (e.g., metrics, parameters) to a JSON file.

    Args:
        results_dict (dict): Dictionary containing the results to save.
        output_dir (str): Directory where the file will be saved.
        filename (str, optional): Name of the JSON file. Defaults to "results.json".
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        # Custom converter for numpy types that are not JSON serializable by default
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4, default=convert_numpy)
        print(f"Results saved to {filepath}")
    except TypeError as e:
        print(f"Error saving results to {filepath}: {e}. Check for non-serializable types.")
    except Exception as e:
        print(f"An unexpected error occurred while saving results to {filepath}: {e}")


def save_predictions(y_true, y_pred, output_dir, filename="predictions.csv", index=None):
    """
    Saves true values and predicted values to a CSV file.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        output_dir (str): Directory where the file will be saved.
        filename (str, optional): Name of the CSV file. Defaults to "predictions.csv".
        index (pd.Index or array-like, optional): Index to use for the DataFrame. Defaults to None.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        # Ensure inputs are consistent lengths
        if len(y_true) != len(y_pred):
             print(f"Error saving predictions: y_true (len {len(y_true)}) and y_pred (len {len(y_pred)}) have different lengths.")
             return
        if index is not None and len(index) != len(y_true):
             print(f"Error saving predictions: index (len {len(index)}) and y_true (len {len(y_true)}) have different lengths.")
             # Fallback to default index if provided index is mismatched
             index = None 
             print("Using default range index instead.")

        pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}, index=index)
        pred_df.to_csv(filepath, index=True, index_label='original_index' if index is not None else 'index')
        print(f"Predictions saved to {filepath} (shape: {pred_df.shape})")
    except Exception as e:
        print(f"Error saving predictions to {filepath}: {e}")
