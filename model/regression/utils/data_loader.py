import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path,
                             target_column='DockQ', # Target for regression
                             query_id_column='query', # Group ID column
                             default_features_to_drop=None,
                             user_features_to_drop=None):
    """
    Loads data for regression, selects target, identifies query IDs,
    drops rows with NaN in feature columns, and drops specified columns.

    Args:
        file_path (str): Path to the input CSV file.
        target_column (str, optional): Name of the continuous target variable column. Defaults to 'DockQ'.
        query_id_column (str, optional): Name of the column containing group IDs. Defaults to 'query'.
        default_features_to_drop (list, optional): List of default columns to always drop.
        user_features_to_drop (list, optional): List of additional columns specified by the user to drop.

    Returns:
        tuple: (X, y, query_ids, y_strat)
               X (pd.DataFrame): Processed features (NaN rows dropped).
               y (pd.Series): Continuous target variable.
               query_ids (pd.Series or None): Group IDs for GroupKFold, or None.
               y_strat (pd.Series or None): Stratification bins for regression stratified splitting, or None.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If target_column is not found or X is empty after processing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Input file not found at {file_path}")

    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")

    # --- Target Variable Selection ---
    if target_column not in data.columns:
        raise ValueError(f"Error: Target column '{target_column}' not found in the data.")

    # Ensure target is numeric, coerce errors, and drop rows where target is NaN
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    initial_rows_target_check = len(data)
    data.dropna(subset=[target_column], inplace=True)
    if len(data) < initial_rows_target_check:
         print(f"Dropped {initial_rows_target_check - len(data)} rows due to NaN in target column '{target_column}'.")
    print(f"Using '{target_column}' as the continuous target variable.")

    # --- Query ID Handling ---
    if query_id_column in data.columns:
        query_ids_temp = data[query_id_column].copy() # Store temporarily before dropping NaNs
    else:
        query_ids_temp = None
        print(f"Warning: Query ID column '{query_id_column}' not found. GroupKFold cannot be used.")

    # --- Feature Selection (Dropping Columns) ---
    cols_to_drop = set() # Start with an empty set for regression

    # Define default columns to drop (similar to classification, but without threshold-based target)
    if default_features_to_drop is None:
         default_features_to_drop = [
              'pdb', 'seed', 'sample', 'data_file', 'chain_iptm', 'chain_pair_iptm',
              'chain_pair_pae_min', 'chain_ptm', 'format', 'model_path', 'native_path',
              'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS', 'LIS', 'mpdockq_AH', 'mpdockq_AL', 'mpdockq_HL', 'mpdockq'
              # Add other classification-specific or less relevant columns here if needed
         ]
         # IMPORTANT: Unlike classification, DO NOT automatically add target_column to cols_to_drop here.
         # It's the value we want to predict! We will drop it from X later.

    valid_default_drops = [col for col in default_features_to_drop if col in data.columns and col != target_column] # Exclude target
    cols_to_drop.update(valid_default_drops)

    if user_features_to_drop:
        valid_user_drops = [col for col in user_features_to_drop if col in data.columns and col != target_column] # Exclude target
        cols_to_drop.update(valid_user_drops)

    # Add query ID column to drop list if it exists
    if query_id_column in data.columns:
        cols_to_drop.add(query_id_column)

    # --- Identify Potential Feature Columns (excluding target and query ID) ---
    potential_feature_cols = [col for col in data.columns if col != target_column and col not in cols_to_drop]
    print(f"Identified {len(potential_feature_cols)} potential feature columns.")
    if not potential_feature_cols:
         raise ValueError("Error: No potential feature columns identified after configuring drops.")

    # --- Handle Missing Values (NaN) in Features by Dropping Rows ---
    print(f"Checking for NaN values in potential feature columns...")
    original_rows_feature_check = len(data)
    # Drop rows where any of the potential feature columns have NaN
    data.dropna(subset=potential_feature_cols, inplace=True)
    rows_dropped = original_rows_feature_check - len(data)

    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NaN values in one or more feature columns.")
    else:
        print("No rows dropped due to NaN values in features.")

    # --- Create final X, y, query_ids from cleaned data ---
    y = data[target_column] # The actual continuous target values
    X = data[potential_feature_cols] # Select only the feature columns

    if query_ids_temp is not None:
        query_ids = query_ids_temp.loc[X.index] # Align query_ids with the final index of X
        if query_ids.isnull().any():
             print(f"Warning: Query ID column still contains NaN after row dropping. Filling with placeholder.")
             query_ids = query_ids.fillna('NaN_Placeholder_Group')
    else:
        query_ids = None

    # --- Stratification Bin Creation for Regression ---
    try:
        # Create 10 quantile-based bins of the target for stratification
        y_strat = pd.qcut(y, q=10, labels=False, duplicates='drop')
        print(f"Generated {y_strat.nunique()} stratification bins from target for regression.")
    except Exception as e:
        print(f"Warning: Unable to bin target for stratification: {e}")
        y_strat = None

    print(f"Processed Features (X) shape after NaN drop: {X.shape}")
    print(f"Processed Target (y) shape after NaN drop: {y.shape}")
    if query_ids is not None:
        print(f"Processed Query IDs shape after NaN drop: {query_ids.shape}")

    # Check for empty features dataframe
    if X.empty or X.shape[1] == 0:
        raise ValueError("Error: Feature set (X) is empty after dropping columns and NaN rows.")

    # Final check for NaNs in X (should be none)
    if X.isnull().sum().sum() > 0:
        print("Error: Final feature set X still contains NaN values after dropping rows!")
        print(X.isnull().sum()[X.isnull().sum() > 0])

    return X, y, query_ids
