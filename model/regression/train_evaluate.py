import argparse
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

# Import utilities
try:
    from utils.data_loader import load_and_preprocess_data
    from utils.evaluation import save_results, save_predictions
except ImportError:
    print("Attempting fallback import for utils...")
    try:
        from model.regression.utils.data_loader import load_and_preprocess_data
        from model.regression.utils.evaluation import save_results, save_predictions
    except ImportError:
        raise ImportError("Could not import data_loader or evaluation utils.")

# Import model classes
try:
    from models.tree_models import RandomForestRegressorModel, LGBMRegressorModel, XGBoostRegressorModel
    from models.linear_models import LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel
except ImportError:
    print("Attempting fallback import for models...")
    try:
        from model.regression.models.tree_models import RandomForestRegressorModel, LGBMRegressorModel, XGBoostRegressorModel
        from model.regression.models.linear_models import LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel
    except ImportError:
        raise ImportError("Could not import model classes.")

MODEL_FACTORY = {
    "RandomForest": RandomForestRegressorModel,
    "LGBM": LGBMRegressorModel,
    "XGBoost": XGBoostRegressorModel,
    "LinearRegression": LinearRegressionModel,
    "Ridge": RidgeModel,
    "Lasso": LassoModel,
    "ElasticNet": ElasticNetModel
}

def main(args):
    """Main function to train and evaluate a regression model."""
    print(f"Starting training and evaluation process for {args.model_type}")
    print(f"Configuration: {vars(args)}")

    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Results will be saved in: {model_output_dir}")

    # --- 1. Load and Preprocess Data ---
    print("\n--- Loading and Preprocessing Data ---")
    try:
        X, y, query_ids = load_and_preprocess_data(
            file_path=args.input_file,
            target_column=args.target_column,
            query_id_column=args.query_id_column if args.query_id_column else None,
            user_features_to_drop=args.drop_features
        )
        feature_names = X.columns.tolist() # Store feature names
        print(f"Data loaded successfully. Features: {len(feature_names)}, Samples: {len(X)}")
        if query_ids is not None:
             print(f"Query IDs loaded for potential grouped splitting (not used in basic train/test).")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error during data loading: {e}")
        return # Exit if data loading fails

    # --- 2. Split Data ---
    # Simple train/test split. For grouped data, GroupShuffleSplit could be used.
    print("\n--- Splitting Data ---")
    try:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, X.index, # Pass index to keep track
            test_size=args.test_size,
            random_state=args.random_state,
            # shuffle=True is default
        )
        print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return

    # --- 3. Initialize Model ---
    print("\n--- Initializing Model ---")
    if args.model_type not in MODEL_FACTORY:
        print(f"Error: Invalid model type '{args.model_type}'. Choose from {list(MODEL_FACTORY.keys())}")
        return

    # Instantiate the model class, passing the specific model's output directory
    model_save_dir = os.path.join(model_output_dir, "model_files") # Subdir for joblib files etc.
    ModelClass = MODEL_FACTORY[args.model_type]
    # Add model-specific hyperparameters from args if needed, e.g.:
    model_params = {}
    if args.model_type in ["Ridge", "Lasso", "ElasticNet"] and args.alpha is not None:
         model_params['alpha'] = args.alpha
    if args.model_type == "ElasticNet" and args.l1_ratio is not None:
         model_params['l1_ratio'] = args.l1_ratio
    # Add other hyperparameters here as needed

    try:
        # Pass the specific directory for this model instance to save into
        model = ModelClass(save_dir=model_save_dir, **model_params)
    except Exception as e:
        print(f"Error initializing model {args.model_type}: {e}")
        return

    # --- 4. Train Model ---
    print("\n--- Training Model ---")
    try:
        train_start_time = time.time()
        # Pass additional fit params like early stopping if needed/supported
        # Example: eval_set for LGBM/XGBoost would require splitting train further
        model.fit(X_train, y_train)
        train_end_time = time.time()
        print(f"Model training completed in {train_end_time - train_start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # --- 5. Predict on Test Set ---
    print("\n--- Predicting on Test Set ---")
    try:
        y_pred = model.predict(X_test)
        print("Predictions generated successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # --- 6. Evaluate Model ---
    print("\n--- Evaluating Model ---")
    try:
        metrics = model.evaluate(X_test, y_test)
        if not metrics:
             print("Warning: Evaluation returned empty metrics.")
        else:
             print("Evaluation Metrics:")
             for key, value in metrics.items():
                 print(f"  {key.upper()}: {value:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        metrics = {} # Ensure metrics is a dict even on error

    # --- 7. Save Results ---
    print("\n--- Saving Results ---")
    # Prepare results dictionary
    results_summary = {
        "model_type": args.model_type,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "data_file": os.path.basename(args.input_file),
        "target_column": args.target_column,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "features_used": len(feature_names),
        "dropped_features_input": args.drop_features,
        # Add model hyperparameters used? model.get_params() could be large
        # "model_params": model.get_params(deep=False), # Get top-level params
        "evaluation_metrics": metrics,
        "training_time_seconds": train_end_time - train_start_time,
        "total_time_seconds": time.time() - start_time,
    }

    # Save metrics and parameters
    save_results(results_summary, model_output_dir, filename="evaluation_summary.json")

    # Save predictions (using original index from test set)
    save_predictions(y_test, y_pred, model_output_dir, filename="predictions.csv", index=idx_test)

    # --- 8. Save Model (Optional) ---
    if args.save_model:
        print("\n--- Saving Model ---")
        # The model instance already knows its save path based on save_dir passed during init
        model.save_model() # Saves to model_output_dir/model_files/<model_name>_regressor.joblib

    # --- 9. Save Feature Importances (Optional) ---
    print("\n--- Getting Feature Importances ---")
    try:
        feature_importance_df = model.get_feature_importances(feature_names)
        if feature_importance_df is not None:
            importance_filename = "feature_importances.csv"
            importance_path = os.path.join(model_output_dir, importance_filename)
            feature_importance_df.to_csv(importance_path, index=False)
            print(f"Feature importances saved to {importance_path}")
        else:
            print("Feature importances are not available for this model or could not be retrieved.")
    except Exception as e:
        print(f"Error getting or saving feature importances: {e}")

    print("\n--- Process Completed ---")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a regression model.")

    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV data file.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results, predictions, and model files.")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=list(MODEL_FACTORY.keys()),
                        help="Type of regression model to train.")
    parser.add_argument("--target-column", type=str, default="DockQ",
                        help="Name of the target variable column.")
    parser.add_argument("--query-id-column", type=str, default="query",
                        help="Name of the query ID column for grouping (optional, set to '' or None to disable). Default 'query'.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion of the dataset to use for the test set.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--drop-features", nargs='+', default=None,
                        help="List of feature names to drop from the dataset.")
    parser.add_argument("--save-model", action='store_true',
                        help="Save the trained model using joblib.")

    # Model-specific hyperparameters (optional examples)
    parser.add_argument("--alpha", type=float, default=None, # Let model defaults handle if None
                        help="Regularization strength (alpha) for Ridge, Lasso, ElasticNet.")
    parser.add_argument("--l1-ratio", type=float, default=None, # Let model defaults handle if None
                        help="L1 ratio for ElasticNet (0=Ridge, 1=Lasso).")
    # Add more hyperparameters as needed (e.g., n_estimators, max_depth for trees)

    args = parser.parse_args()

    # Handle empty string for query_id_column
    if args.query_id_column == '':
         args.query_id_column = None

    main(args)
