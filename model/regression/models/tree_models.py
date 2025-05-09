import os
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import shap # SHAP 임포트
import time
import traceback

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# Import metrics directly for potential use, though evaluate uses the helper
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform # 하이퍼파라미터 튜닝용 분포

# Import the evaluation helper from the utils directory
try:
    from ..utils.evaluation import calculate_regression_metrics
except ImportError:
    # Fallback for execution from different directories if needed
    from model.regression.utils.evaluation import calculate_regression_metrics


class BaseModelRegressor(ABC):
    """Abstract base class for regression models, updated for Nested CV and SHAP."""
    # model_registry_name: Short internal name (e.g., 'rf', 'lgbm')
    # display_name: User-friendly name (e.g., 'Random Forest', 'LightGBM')
    model_registry_name = "base_regressor"
    display_name = "Base Regressor"

    def __init__(self, model_registry_name, display_name, save_dir="saved_models"):
        self.model_registry_name = model_registry_name
        self.display_name = display_name
        self.model = None # Actual model instance (set by subclasses or loaded)
        self.save_dir = save_dir
        # Use the registry name for directory structure consistency
        self.specific_save_dir = os.path.join(self.save_dir, self.model_registry_name)
        os.makedirs(self.specific_save_dir, exist_ok=True)
        # Model filename uses registry name too
        self.model_path = os.path.join(self.specific_save_dir, f"{self.model_registry_name}_regressor.joblib")

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """Train the regression model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions with the regression model."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set using regression metrics.

        Args:
            X_test (pd.DataFrame or np.ndarray): Test features.
            y_test (pd.Series or np.ndarray): True test target values.

        Returns:
            dict: Dictionary containing regression metrics (R2, MSE, RMSE, MAE).
                  Returns empty dict if evaluation fails.
        """
        if self.model is None:
            print("Error: Model has not been trained or loaded yet for evaluation.")
            return {}
        try:
            y_pred = self.predict(X_test)

            # Ensure y_test and y_pred are numpy arrays for calculation robustness
            y_test = np.asarray(y_test)
            y_pred = np.asarray(y_pred)

            if y_test.shape != y_pred.shape:
                 print(f"Warning: Shape mismatch in evaluation. y_test: {y_test.shape}, y_pred: {y_pred.shape}")
                 # Attempt to reshape if possible and makes sense, otherwise raise error or return empty
                 try:
                     # Check if y_pred is squeeze-able to match y_test
                     if y_pred.squeeze().shape == y_test.shape:
                          y_pred = y_pred.squeeze()
                          print("Reshaped y_pred to match y_test shape.")
                     else:
                          print("Error: Cannot reshape y_pred to match y_test. Evaluation failed.")
                          return {}
                 except Exception as e:
                     print(f"Error during reshape attempt: {e}. Evaluation failed.")
                     return {}

            return calculate_regression_metrics(y_test, y_pred)

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            return {}


    def save_model(self, path=None):
        """Saves the trained model to a file using joblib."""
        if self.model is None:
            print("Error: No model to save. Train or load a model first.")
            return
        save_path = path if path else self.model_path
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            joblib.dump(self.model, save_path)
            print(f"Model saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving model to {save_path}: {e}")

    def load_model(self, path=None):
        """Loads a model from a file using joblib."""
        load_path = path if path else self.model_path
        if not os.path.exists(load_path):
             print(f"Error: Model file not found at {load_path}")
             self.model = None # Ensure model is None if file doesn't exist
             # raise FileNotFoundError(f"Model file not found at {load_path}") # Option to raise error
             return # Return instead of raising error to allow attempting training later
        try:
            self.model = joblib.load(load_path)
            print(f"Model loaded successfully from {load_path}")
            # Optional: Verify model type after loading if needed
            # if hasattr(self, '_expected_model_type') and not isinstance(self.model, self._expected_model_type):
            #     print(f"Warning: Loaded model type mismatch. Expected {self._expected_model_type}, got {type(self.model)}")
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            self.model = None # Ensure model is None if loading failed

    def get_params(self, deep=True):
        """Gets parameters for this estimator's model."""
        if self.model:
            # If model is a pipeline, we might want params of the final step
            if hasattr(self.model, 'named_steps'):
                 # Assuming the regressor is the last step
                 step_name = list(self.model.named_steps.keys())[-1]
                 return self.model.named_steps[step_name].get_params(deep=deep)
            else:
                 return self.model.get_params(deep=deep)
        print("Warning: Model not initialized, returning empty parameter dict.")
        return {}

    def set_params(self, **params):
        """Sets the parameters of this estimator's model."""
        if self.model:
            try:
                 # Handle pipelines correctly
                 if hasattr(self.model, 'named_steps'):
                      step_name = list(self.model.named_steps.keys())[-1]
                      # Scikit-learn pipeline parameter setting uses 'stepname__paramname'
                      pipeline_params = {f"{step_name}__{k}": v for k, v in params.items()}
                      self.model.set_params(**pipeline_params)
                 else:
                      self.model.set_params(**params)
            except ValueError as e:
                 print(f"Error setting parameters: {e}. Check parameter names/values.")
        else:
             print("Warning: Model not initialized. Parameters cannot be set.")
        return self

    def get_feature_importances(self, feature_names):
        """
        Returns feature importances if the underlying model supports it.

        Args:
            feature_names (list): List of feature names corresponding to the columns in X.

        Returns:
            pd.DataFrame or None: DataFrame with features and their importance scores,
                                   sorted descending. None if not supported or model not trained.
        """
        if self.model is None:
            print("Model not trained or loaded yet. Cannot get feature importances.")
            return None

        model_instance = self.model
        # If it's a pipeline, get the final estimator (the regressor)
        if hasattr(self.model, 'named_steps'):
            step_name = list(self.model.named_steps.keys())[-1]
            model_instance = self.model.named_steps[step_name]

        importances = None
        if hasattr(model_instance, 'feature_importances_'):
            importances = model_instance.feature_importances_
        elif hasattr(model_instance, 'coef_'): # Handle linear models here too if needed
            importances = model_instance.coef_
            # Coef might need absolute value for importance ranking
            # importances = np.abs(model_instance.coef_)
            # Handle multi-dim coef if necessary
            if importances.ndim > 1: importances = np.mean(np.abs(importances), axis=0)
        else:
             print(f"Model type {type(model_instance).__name__} does not provide feature_importances_ or coef_.")
             return None

        if importances is not None:
            if len(importances) != len(feature_names):
                 print(f"Warning: Mismatch ({len(importances)} vs {len(feature_names)}) in feature importance/coefficient count.")
                 return None # Avoid creating incorrect DataFrame
            try:
                 # Determine column name based on attribute
                 importance_col = 'importance' if hasattr(model_instance, 'feature_importances_') else 'coefficient'
                 feature_importance_df = pd.DataFrame({
                     'feature': feature_names,
                     importance_col: importances
                 })
                 # Sort by absolute value for ranking
                 sort_col = f'abs_{importance_col}'
                 feature_importance_df[sort_col] = feature_importance_df[importance_col].abs()
                 feature_importance_df = feature_importance_df.sort_values(by=sort_col, ascending=False).drop(columns=[sort_col]).reset_index(drop=True)
                 return feature_importance_df
            except Exception as e:
                 print(f"Error creating feature importance DataFrame: {e}")
                 return None
        else:
            # Already logged message above
            return None

    # --- New Methods for Nested CV and SHAP ---

    def get_model_name(self) -> str:
        """Returns the short, internal name of the model (e.g., 'rf')."""
        return self.model_registry_name

    def get_display_name(self) -> str:
        """Returns the user-friendly name of the model (e.g., 'Random Forest')."""
        return self.display_name

    @abstractmethod
    def get_hyperparameter_grid(self) -> dict:
        """
        Returns the hyperparameter grid for RandomizedSearchCV.
        Keys should match the model's parameter names.
        Values should be distributions from scipy.stats or lists.
        """
        pass

    @abstractmethod
    def create_model(self, params: dict = None, n_jobs: int = 1):
        """
        Creates a new instance of the underlying scikit-learn model.
        Used by RandomizedSearchCV as the base estimator.

        Args:
            params (dict, optional): Parameters to initialize the model with.
                                     If None, use model defaults.
            n_jobs (int): Number of jobs for model initialization (if applicable).

        Returns:
            An instance of the specific scikit-learn regressor (e.g., RandomForestRegressor).
            Note: This returns the *core* model, not necessarily the pipeline used internally by the class instance.
        """
        pass

    @abstractmethod
    def calculate_shap_values(self, trained_model, X_explain: pd.DataFrame, X_train_sample=None):
        """
        Calculates SHAP values for the given data using the trained model.

        Args:
            trained_model: The trained model instance (could be the result of RandomizedSearchCV.best_estimator_).
                           This should ideally be the core regressor, not the pipeline,
                           if the pipeline includes scaling.
            X_explain (pd.DataFrame): The data for which to explain predictions.
                                      Should be the original unscaled data if the model pipeline includes scaling.
            X_train_sample: This argument is accepted for compatibility but not used by TreeExplainer.

        Returns:
            np.ndarray or shap.Explanation: SHAP values, or None if calculation fails.
        """
        pass


# --- Tree-based Regressor Implementations ---

class RandomForestRegressorModel(BaseModelRegressor):
    """Random Forest Regressor model, adapted for Nested CV and SHAP."""
    model_registry_name = "rf" # Short name
    display_name = "Random Forest" # Display name

    def __init__(self, save_dir="saved_models", **kwargs):
        # Default RF params
        default_params = {
            'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'n_jobs': -1, 'random_state': 42, 'oob_score': False
        }
        final_params = {**default_params, **kwargs}
        super().__init__(model_registry_name=self.model_registry_name,
                         display_name=self.display_name,
                         save_dir=save_dir)
        # Store initial params used for this instance
        self.initial_params = final_params
        self.model = RandomForestRegressor(**final_params) # Keep internal model instance
        print(f"Initialized {self.display_name} ({self.model_registry_name}) with params: {self.model.get_params()}")

    def create_model(self, params: dict = None, n_jobs: int = 1):
        """Creates a RandomForestRegressor instance."""
        # Use provided params, otherwise use the defaults stored during __init__
        # Note: We ignore self.model here, creating a fresh instance
        model_params = params if params is not None else self.initial_params.copy()
        # Ensure n_jobs is correctly passed or defaulted
        model_params['n_jobs'] = n_jobs
        # Ensure random_state is consistent if not overridden by tuning params
        if 'random_state' not in model_params:
             model_params['random_state'] = self.initial_params.get('random_state', 42)
        return RandomForestRegressor(**model_params)

    def get_hyperparameter_grid(self) -> dict:
        """Hyperparameter grid for RandomForestRegressor."""
        return {
            'n_estimators': randint(100, 500),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None], # Note: 'auto' is same as 'sqrt' for RF regressor
            'bootstrap': [True, False] # Consider if bootstrapping is desired
        }

    def fit(self, X_train, y_train, **kwargs):
        """Fits the internal RandomForestRegressor model."""
        if self.model is None: self.model = self.create_model() # Recreate if None
        print(f"Fitting {self.display_name}...")
        try:
            if hasattr(y_train, 'ravel'): y_train = y_train.ravel()
            self.model.fit(X_train, y_train, **kwargs)
            print("Fitting completed.")
            if self.model.oob_score: print(f"OOB Score: {self.model.oob_score_:.4f}")
        except Exception as e: print(f"Error during {self.display_name} fitting: {e}"); raise e

    def predict(self, X):
        """Predicts using the internal RandomForestRegressor model."""
        if self.model is None: raise RuntimeError("Model not trained/loaded.")
        try: return self.model.predict(X)
        except Exception as e: print(f"Error during {self.display_name} prediction: {e}"); raise e

    def calculate_shap_values(self, trained_model, X_explain: pd.DataFrame, X_train_sample=None):
        """
        Calculates SHAP values using Explainer for RandomForest.
        Returns results as DataFrame along with base values.

        Args:
            trained_model (RandomForestRegressor): The trained model instance.
            X_explain (pd.DataFrame or np.ndarray): Data to explain.
            X_train_sample: Optional background data for explainer.

        Returns:
            tuple(pd.DataFrame, np.ndarray) or tuple(None, None):
                - SHAP values as a DataFrame.
                - SHAP base values (expected value).
        """
        print(f"Calculating SHAP values for {self.display_name} using SHAP Explainer...")
        if not isinstance(trained_model, RandomForestRegressor):
             print(f"Warning: Expected RandomForestRegressor, got {type(trained_model)}. SHAP might fail.")

        start_time = time.time()
        try:
            # Ensure X_explain is DataFrame for consistent handling
            original_columns = getattr(X_explain, 'columns', None)
            original_index = getattr(X_explain, 'index', None)
            if not isinstance(X_explain, pd.DataFrame):
                print("Warning: X_explain is not a DataFrame. Converting for SHAP calculation.")
                # Create temporary DataFrame for explainer if needed, preserve original info
                temp_cols = original_columns if original_columns is not None else [f'feature_{i}' for i in range(X_explain.shape[1])]
                X_explain_df = pd.DataFrame(X_explain, columns=temp_cols)
                if original_index is None:
                    original_index = X_explain_df.index
            else:
                X_explain_df = X_explain # Use the original DataFrame

            # 1) 배경 데이터 설정
            background = X_train_sample if X_train_sample is not None else None

            # 2) 통합된 Explainer API 사용
            explainer = shap.TreeExplainer(trained_model)
            
            # 3) explainer(X) 호출 방식 사용
            shap_exp = explainer(X_explain_df)
            
            # 4) .values로 numpy array 획득
            shap_values = shap_exp.values
            base_values = shap_exp.base_values

            # Robustly get index and columns for output DataFrame
            if original_columns is None:
                 print("Warning: Original X_explain had no columns. Using generic feature names for output.")
                 output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
            else:
                 if len(original_columns) != shap_values.shape[1]:
                      print(f"Warning: Mismatch between original columns ({len(original_columns)}) and SHAP features ({shap_values.shape[1]}). Using generic names.")
                      output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
                 else:
                      output_columns = original_columns

            if original_index is None:
                 original_index = pd.RangeIndex(start=0, stop=shap_values.shape[0], step=1)

            shap_values_df = pd.DataFrame(
                shap_values,
                index=original_index,
                columns=output_columns
            )

            end_time = time.time()
            print(f"SHAP calculation took {end_time - start_time:.2f} seconds.")
            return shap_values_df, base_values

        except Exception as e:
            print(f"Error calculating SHAP values for {self.display_name}: {e}")
            print(traceback.format_exc())
            return None, None

    # evaluate, save_model, load_model, get_params, set_params, get_feature_importances inherited


class LGBMRegressorModel(BaseModelRegressor):
    """LightGBM Regressor model, adapted for Nested CV and SHAP."""
    model_registry_name = "lgbm"
    display_name = "LightGBM"

    def __init__(self, save_dir="saved_models", **kwargs):
        default_params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 100,
            'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.0, 'reg_lambda': 0.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
        }
        final_params = {**default_params, **kwargs}
        super().__init__(model_registry_name=self.model_registry_name,
                         display_name=self.display_name,
                         save_dir=save_dir)
        self.initial_params = final_params
        self.model = LGBMRegressor(**final_params)
        print(f"Initialized {self.display_name} ({self.model_registry_name}) with params: {self.model.get_params()}")

    def create_model(self, params: dict = None, n_jobs: int = 1):
        """Creates an LGBMRegressor instance."""
        model_params = params if params is not None else self.initial_params.copy()
        model_params['n_jobs'] = n_jobs
        if 'random_state' not in model_params:
             model_params['random_state'] = self.initial_params.get('random_state', 42)
        # Ensure objective/metric are set if not in tuned params
        if 'objective' not in model_params: model_params['objective'] = self.initial_params.get('objective','regression_l1')
        if 'metric' not in model_params: model_params['metric'] = self.initial_params.get('metric','mae')
        return LGBMRegressor(**model_params)

    def get_hyperparameter_grid(self) -> dict:
        """Hyperparameter grid for LGBMRegressor."""
        return {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.2), # Sample from 0.01 to 0.21
            'num_leaves': randint(20, 60),
            'max_depth': [-1, 10, 20, 30],
            'min_child_samples': randint(10, 40),
            'subsample': uniform(0.6, 0.4), # Sample from 0.6 to 1.0
            'colsample_bytree': uniform(0.6, 0.4), # Sample from 0.6 to 1.0
            'reg_alpha': uniform(0, 1),       # L1 reg
            'reg_lambda': uniform(0, 1)       # L2 reg
        }

    def fit(self, X_train, y_train, **kwargs):
        """Fits the internal LGBMRegressor model. Supports early stopping via kwargs."""
        if self.model is None: self.model = self.create_model()
        print(f"Fitting {self.display_name}...")
        fit_params = {k: v for k, v in kwargs.items() if k in ['eval_set', 'callbacks', 'eval_metric']}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in fit_params}

        try:
            if hasattr(y_train, 'ravel'): y_train = y_train.ravel()
            self.model.fit(X_train, y_train, **fit_params, **other_kwargs)
            print("Fitting completed.")
            if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_:
                 print(f"Best iteration found: {self.model.best_iteration_}")
        except Exception as e: print(f"Error during {self.display_name} fitting: {e}"); raise e

    def predict(self, X):
        """Predicts using the internal LGBMRegressor model."""
        if self.model is None: raise RuntimeError("Model not trained/loaded.")
        try:
            best_iter = getattr(self.model, 'best_iteration_', None)
            return self.model.predict(X, num_iteration=best_iter)
        except Exception as e: print(f"Error during {self.display_name} prediction: {e}"); raise e

    def calculate_shap_values(self, trained_model, X_explain: pd.DataFrame, X_train_sample=None):
        """
        Calculates SHAP values using Explainer for LightGBM.
        Returns results as DataFrame along with base values.

        Args:
            trained_model (LGBMRegressor): The trained model instance.
            X_explain (pd.DataFrame or np.ndarray): Data to explain.
            X_train_sample: Optional background data for explainer.

        Returns:
            tuple(pd.DataFrame, np.ndarray) or tuple(None, None):
                - SHAP values as a DataFrame.
                - SHAP base values (expected value).
        """
        print(f"Calculating SHAP values for {self.display_name} using SHAP Explainer...")
        if not isinstance(trained_model, LGBMRegressor):
             print(f"Warning: Expected LGBMRegressor, got {type(trained_model)}. SHAP might fail.")

        start_time = time.time()
        try:
            # Ensure X_explain is DataFrame for consistent handling
            original_columns = getattr(X_explain, 'columns', None)
            original_index = getattr(X_explain, 'index', None)
            if not isinstance(X_explain, pd.DataFrame):
                print("Warning: X_explain is not a DataFrame. Converting for SHAP calculation.")
                temp_cols = original_columns if original_columns is not None else [f'feature_{i}' for i in range(X_explain.shape[1])]
                X_explain_df = pd.DataFrame(X_explain, columns=temp_cols)
                if original_index is None:
                    original_index = X_explain_df.index
            else:
                X_explain_df = X_explain

            # 배경 데이터 설정
            background = X_train_sample if X_train_sample is not None else None

            # 통합된 Explainer API 사용
            explainer = shap.TreeExplainer(trained_model)
            
            # explainer(X) 호출 방식 사용
            shap_exp = explainer(X_explain_df)
            
            # .values로 numpy array 획득
            shap_values = shap_exp.values
            base_values = shap_exp.base_values

            # Robustly get index and columns for output DataFrame
            if original_columns is None:
                 output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
            else:
                 if len(original_columns) != shap_values.shape[1]:
                      output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
                 else:
                      output_columns = original_columns

            if original_index is None:
                 original_index = pd.RangeIndex(start=0, stop=shap_values.shape[0], step=1)

            shap_values_df = pd.DataFrame(
                shap_values,
                index=original_index,
                columns=output_columns
            )

            end_time = time.time()
            print(f"SHAP calculation took {end_time - start_time:.2f} seconds.")
            return shap_values_df, base_values

        except Exception as e:
            print(f"Error calculating SHAP values for {self.display_name}: {e}")
            print(traceback.format_exc())
            return None, None

    # evaluate, save_model, load_model, get_params, set_params, get_feature_importances inherited


class XGBoostRegressorModel(BaseModelRegressor):
    """XGBoost Regressor model, adapted for Nested CV and SHAP."""
    model_registry_name = "xgb"
    display_name = "XGBoost"

    def __init__(self, save_dir="saved_models", **kwargs):
        default_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 100,
            'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'random_state': 42, 'n_jobs': -1,
            'tree_method': 'hist'
        }
        final_params = {**default_params, **kwargs}
        super().__init__(model_registry_name=self.model_registry_name,
                         display_name=self.display_name,
                         save_dir=save_dir)
        self.initial_params = final_params
        self.model = XGBRegressor(**final_params)
        print(f"Initialized {self.display_name} ({self.model_registry_name}) with params: {self.model.get_params()}")


    def create_model(self, params: dict = None, n_jobs: int = 1):
        """Creates an XGBRegressor instance."""
        model_params = params if params is not None else self.initial_params.copy()
        model_params['n_jobs'] = n_jobs
        if 'random_state' not in model_params:
             model_params['random_state'] = self.initial_params.get('random_state', 42)
        if 'objective' not in model_params: model_params['objective'] = self.initial_params.get('objective','reg:squarederror')
        if 'eval_metric' not in model_params: model_params['eval_metric'] = self.initial_params.get('eval_metric','rmse')
        # Handle potential issues with booster type if not specified
        if 'booster' not in model_params: model_params['booster'] = 'gbtree' # Default for regression
        return XGBRegressor(**model_params)

    def get_hyperparameter_grid(self) -> dict:
        """Hyperparameter grid for XGBRegressor."""
        return {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.2), # eta
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),          # Min loss reduction
            'reg_alpha': uniform(0, 1),        # L1 reg
            'reg_lambda': uniform(0, 2)        # L2 reg (often > 1 is useful)
        }

    def fit(self, X_train, y_train, **kwargs):
        """Fits the internal XGBRegressor model. Supports early stopping via kwargs."""
        if self.model is None: self.model = self.create_model()
        print(f"Fitting {self.display_name}...")
        fit_params = {k: v for k, v in kwargs.items() if k in ['eval_set', 'early_stopping_rounds', 'verbose']}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in fit_params}

        try:
            if hasattr(y_train, 'ravel'): y_train = y_train.ravel()
            self.model.fit(X_train, y_train, **fit_params, **other_kwargs)
            print("Fitting completed.")
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                 print(f"Best iteration found: {self.model.best_iteration}")
        except Exception as e: print(f"Error during {self.display_name} fitting: {e}"); raise e

    def predict(self, X):
        """Predicts using the internal XGBRegressor model."""
        if self.model is None: raise RuntimeError("Model not trained/loaded.")
        try:
            # Predict using best model if early stopping was used (handled by sklearn wrapper)
            return self.model.predict(X)
        except Exception as e: print(f"Error during {self.display_name} prediction: {e}"); raise e

    def calculate_shap_values(self, trained_model, X_explain: pd.DataFrame, X_train_sample=None):
        """
        Calculates SHAP values using Explainer for XGBoost.
        Returns results as DataFrame along with base values.

        Args:
            trained_model (XGBRegressor): The trained model instance.
            X_explain (pd.DataFrame or np.ndarray): Data to explain.
            X_train_sample: Optional background data for explainer.

        Returns:
            tuple(pd.DataFrame, np.ndarray) or tuple(None, None):
                - SHAP values as a DataFrame.
                - SHAP base values (expected value).
        """
        print(f"Calculating SHAP values for {self.display_name} using SHAP Explainer...")
        if not isinstance(trained_model, XGBRegressor):
             print(f"Warning: Expected XGBRegressor, got {type(trained_model)}. SHAP might fail.")

        start_time = time.time()
        try:
            # Ensure X_explain is DataFrame for consistent handling
            original_columns = getattr(X_explain, 'columns', None)
            original_index = getattr(X_explain, 'index', None)
            if not isinstance(X_explain, pd.DataFrame):
                print("Warning: X_explain is not a DataFrame. Converting for SHAP calculation.")
                temp_cols = original_columns if original_columns is not None else [f'feature_{i}' for i in range(X_explain.shape[1])]
                X_explain_df = pd.DataFrame(X_explain, columns=temp_cols)
                if original_index is None:
                    original_index = X_explain_df.index
            else:
                X_explain_df = X_explain

            # 배경 데이터 설정
            background = X_train_sample if X_train_sample is not None else None

            # 통합된 Explainer API 사용
            explainer = shap.TreeExplainer(trained_model)
            
            # explainer(X) 호출 방식 사용
            shap_exp = explainer(X_explain_df)
            
            # .values로 numpy array 획득
            shap_values = shap_exp.values
            base_values = shap_exp.base_values

            # Robustly get index and columns for output DataFrame
            if original_columns is None:
                 output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
            else:
                 if len(original_columns) != shap_values.shape[1]:
                      output_columns=[f'feature_{i}' for i in range(shap_values.shape[1])]
                 else:
                      output_columns = original_columns

            if original_index is None:
                 original_index = pd.RangeIndex(start=0, stop=shap_values.shape[0], step=1)

            shap_values_df = pd.DataFrame(
                shap_values,
                index=original_index,
                columns=output_columns
            )

            end_time = time.time()
            print(f"SHAP calculation took {end_time - start_time:.2f} seconds.")
            return shap_values_df, base_values

        except Exception as e:
            print(f"Error calculating SHAP values for {self.display_name}: {e}")
            print(traceback.format_exc())
            return None, None

    # evaluate, save_model, load_model, get_params, set_params, get_feature_importances inherited
