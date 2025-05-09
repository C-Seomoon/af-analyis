import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
# Use pathlib for robust path operations
from pathlib import Path 
import traceback # Import traceback for better error details

def save_shap_results(shap_values, X_test, feature_names, output_dir, fold_num):
    """
    Saves SHAP values and the corresponding test data subset to CSV files.
    Ensures SHAP values are 2D and saves the DataFrame index.
    Raises exceptions on failure.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Validate SHAP values ---
    if shap_values is None:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: Input shap_values is None.")
    if not isinstance(shap_values, np.ndarray):
        raise TypeError(f"Cannot save SHAP values for fold {fold_num}: shap_values must be a NumPy array, got {type(shap_values)}.")
    if shap_values.ndim != 2:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: SHAP values must be 2-dimensional (samples, features), but got shape {shap_values.shape}.")
    if X_test is None:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: Input X_test is None.")
    if isinstance(X_test, pd.DataFrame) and X_test.index is None:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: X_test DataFrame is missing an index.")

    if shap_values.shape[0] != X_test.shape[0]:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: Mismatch between SHAP values rows ({shap_values.shape[0]}) and X_test rows ({X_test.shape[0]}).")
    if feature_names is None or len(feature_names) == 0:
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: feature_names list is missing or empty.")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(f"Cannot save SHAP values for fold {fold_num}: Mismatch between SHAP values columns ({shap_values.shape[1]}) and number of feature names ({len(feature_names)}).")
        
    # --- Create DataFrames ---
    try:
        # Ensure X_test is a DataFrame for consistent indexing
        if not isinstance(X_test, pd.DataFrame):
             print(f"Warning: X_test for fold {fold_num} is not a DataFrame. Attempting to create one.")
             # Try creating with a default index if X_test has none; requires length match checked above
             try:
                 x_test_index = pd.RangeIndex(start=0, stop=shap_values.shape[0])
                 X_test_df = pd.DataFrame(X_test, columns=feature_names, index=x_test_index)
             except Exception as convert_e:
                  print(f"Error converting X_test to DataFrame in fold {fold_num}: {convert_e}")
                  raise TypeError(f"Error converting X_test to DataFrame in fold {fold_num}: {convert_e}") from convert_e
        else:
             X_test_df = X_test

        # Create DataFrame for SHAP values using X_test's index
        shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_test_df.index)
        
        # Ensure X_test columns match feature_names and align if needed
        if not all(col in X_test_df.columns for col in feature_names):
             print(f"Warning: Some feature names missing in X_test columns for fold {fold_num}. Saving X_test with its original columns.")
             X_test_df_aligned = X_test_df # 경고만 하고 진행
        else:
             # Ensure column order matches feature_names for the saved file
             X_test_df_aligned = X_test_df[feature_names]

    except Exception as e:
        print(f"Error creating DataFrames for saving SHAP results in fold {fold_num}: {e}")
        print(traceback.format_exc())
        raise RuntimeError(f"Error creating DataFrames for saving SHAP results in fold {fold_num}: {e}") from e

    # --- Define File Paths ---
    shap_filename = f"shap_values_fold_{fold_num}.csv"
    data_filename = f"test_data_fold_{fold_num}.csv"
    shap_filepath = os.path.join(output_dir, shap_filename)
    data_filepath = os.path.join(output_dir, data_filename)

    # --- Save to CSV ---
    try:
        shap_df.to_csv(shap_filepath, index=True) # Save index
        print(f"SHAP values saved to {shap_filepath} (shape: {shap_df.shape})")
        # Save the aligned/original X_test data
        X_test_df_aligned.to_csv(data_filepath, index=True) # Save index
        print(f"Test data corresponding to SHAP values saved to {data_filepath} (shape: {X_test_df_aligned.shape})")
    except Exception as e:
        print(f"Error writing SHAP or test data CSV files for fold {fold_num}: {e}")
        print(traceback.format_exc())
        raise IOError(f"Error writing SHAP or test data CSV files for fold {fold_num}: {e}") from e


def analyze_global_shap(model_name, model, test_data_df, shap_values_np=None, feature_names_list=None, output_dir=None):
    """
    주어진 모델의 글로벌 SHAP 값을 분석하고 시각화합니다.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap
    import os
    
    if shap_values_np is None:
        print(f"SHAP values not provided for {model_name}. Cannot proceed.")
        return
    
    if feature_names_list is None:
        feature_names_list = test_data_df.columns.tolist()
    
    if output_dir is None:
        output_dir = '.'
    
    try:
        # 수동으로 바 플롯 생성
        mean_abs_shap = np.abs(shap_values_np).mean(axis=0)
        shap_importance = pd.Series(mean_abs_shap, index=feature_names_list)
        top_features = shap_importance.sort_values(ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        ax = top_features[::-1].plot(kind='barh', color='#1E88E5')
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{model_name} - Top 15 Features (SHAP Importance)')
        plt.tight_layout()
        
        bar_plot_path = os.path.join(output_dir, f"{model_name}_global_shap_bar_top15.png")
        plt.savefig(bar_plot_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating bar plot: {e}")
    
    try:
        # Beeswarm 플롯 생성
        if isinstance(test_data_df, np.ndarray):
            test_data_df_for_plot = pd.DataFrame(test_data_df, columns=feature_names_list)
        else:
            test_data_df_for_plot = test_data_df.copy()
        
        plt.figure(figsize=(10, 8))
        
        # SHAP library 내부 경고 메시지 억제
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(
                shap_values_np,
                test_data_df_for_plot,
                feature_names=feature_names_list,
                plot_type="dot",
                max_display=15,
                show=False
            )
        
        plt.title(f'{model_name} - SHAP Values (Beeswarm Plot)')
        plt.tight_layout()
        
        beeswarm_path = os.path.join(output_dir, f"{model_name}_global_shap_beeswarm_top15.png")
        plt.savefig(beeswarm_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating beeswarm plot: {e}")
        
    return True
