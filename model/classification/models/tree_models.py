# model/classification/models/tree_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

# Import the base class
from .base_model import ClassificationModel 

class RandomForestModel(ClassificationModel):
    """Random Forest 분류 모델 구현"""
    
    def __init__(self):
        super().__init__(name='rf', display_name='Random Forest')

    def get_hyperparameter_grid(self):
        """Random Forest 하이퍼파라미터 탐색 공간 정의"""
        # Define the parameter grid for RandomizedSearchCV
        return {
            'n_estimators': [100, 200, 300, 500, 700], # Number of trees
            'max_depth': [None, 10, 20, 30, 40, 50],    # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],           # Minimum samples required to split a node
            'min_samples_leaf': [1, 2, 4],             # Minimum samples required at each leaf node
            'max_features': ['sqrt', 'log2', None],    # Number of features to consider at every split
            'class_weight': ['balanced', 'balanced_subsample', None] # Handle class imbalance
        }

    def create_model(self, params=None, n_jobs=1):
        """Random Forest 모델 인스턴스 생성"""
        if params is None:
            params = {}
        
        # Ensure n_jobs is handled correctly
        model_n_jobs = n_jobs if n_jobs is not None and n_jobs > 0 else -1

        # Set default parameters and update with provided params
        model_params = {
            'random_state': 42,
            'n_jobs': model_n_jobs,
            **params # Overwrite defaults with tuned params
        }
        
        # Remove None class_weight if selected by hyperparameter tuning
        if model_params.get('class_weight') is None:
             # Check if it actually exists before deleting
             if 'class_weight' in model_params:
                 del model_params['class_weight']

        return RandomForestClassifier(**model_params)

    def calculate_shap_values(self, model, X_test, X_background=None):
        """
        Random Forest 모델에 대한 SHAP 값을 계산합니다. (반환값 형태 수정됨)

        Args:
            model: 학습된 RandomForestClassifier 모델
            X_test: SHAP 값을 계산할 데이터 (DataFrame 권장)
            X_background: 배경 데이터

        Returns:
            np.ndarray or None:
                - 양성 클래스(1)에 대한 SHAP 값 (n_samples, n_features) 2D numpy 배열.
                - 오류 발생 시 None.
        """
        print("Calculating SHAP values for RandomForest using TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_out = explainer.shap_values(X_test)

            # shap_out 형태 확인 및 양성 클래스(1) 추출 -> 2D 배열 반환
            if isinstance(shap_out, list) and len(shap_out) > 1:
                # 클래스별 리스트: [shap_class0, shap_class1] -> shap_class1 선택
                shap_vals = shap_out[1]
                print("Returning SHAP values for the positive class (from list).")
            elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
                # 3D 배열: (n_samples, n_features, n_classes) -> [:, :, 1] 슬라이싱
                shap_vals = shap_out[:, :, 1]
                print("Returning SHAP values for the positive class (from 3D array).")
            elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 2:
                 # 이미 2D 배열인 경우 (예상치 못한 경우일 수 있으나 일단 반환)
                 print("Warning: TreeExplainer returned a 2D array. Assuming it relates to the positive class.")
                 shap_vals = shap_out
            else:
                print(f"Error: Unexpected SHAP values format: {type(shap_out)}. Returning None.")
                return None

            # 최종 반환값이 2D인지 확인 (추가 검증)
            if shap_vals.ndim != 2:
                 print(f"Error: Calculated SHAP values are not 2D (shape: {shap_vals.shape}). Returning None.")
                 return None

            return shap_vals

        except Exception as e:
            print(f"Error calculating SHAP values for RandomForest: {e}")
            # traceback.print_exc()
            return None

    def fit(self, X, y, **kwargs):
        """
        Random Forest 모델을 학습합니다.
        
        Args:
            X (pd.DataFrame): 학습에 사용할 특성 데이터
            y: 학습에 사용할 타겟 값
            **kwargs: 모델 학습에 필요한 추가 인수
            
        Returns:
            self: 학습된 모델 인스턴스
        """
        model = self.create_model()
        model.fit(X, y, **kwargs)
        self.model = model
        return self
        
    def predict(self, X):
        """
        학습된 Random Forest 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터
            
        Returns:
            np.ndarray: 예측값
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit 메서드를 호출하세요.")
        return self.model.predict(X)

# --- XGBoost Model Implementation ---
class XGBoostModel(ClassificationModel):
    """XGBoost 분류 모델 구현"""

    def __init__(self):
        super().__init__(name='xgb', display_name='XGBoost')

    def get_hyperparameter_grid(self):
        """XGBoost 하이퍼파라미터 탐색 공간 정의"""
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0], # Fraction of samples used for fitting the trees
            'colsample_bytree': [0.6, 0.8, 1.0], # Fraction of features used per tree
            'gamma': [0, 0.1, 0.5, 1], # Minimum loss reduction required to make a further partition
            'scale_pos_weight': [1, 3, 5, 7] # Control the balance of positive and negative weights, useful for unbalanced classes
            # 'reg_alpha': [0, 0.01, 0.1], # L1 regularization (Lasso) - Optional
            # 'reg_lambda': [0.1, 1, 10]   # L2 regularization (Ridge) - Optional
        }

    def create_model(self, params=None, n_jobs=1):
        """XGBoost 모델 인스턴스를 생성합니다."""
        # 기본 파라미터
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'verbosity': 0,           # 기존 verbose 대신 verbosity 사용 (조용히)
            'eval_metric': 'logloss', # 평가 메트릭 명시 (use_label_encoder 경고 방지)
            'n_jobs': 1
        }
        
        # 사용자 지정 파라미터가 있으면 업데이트
        if params:
            default_params.update(params)
        
        # 모델 생성
        # use_label_encoder 파라미터 제거 (최신 버전에서는 불필요)
        model = xgb.XGBClassifier(**default_params)
        
        return model

    def calculate_shap_values(self, model, X_test, X_background=None):
        """
        XGBoost 모델에 대한 SHAP 값을 계산합니다. (반환값 형태 수정됨)

        Args:
            model: 학습된 XGBClassifier 모델
            X_test: SHAP 값을 계산할 데이터 (DataFrame 권장)
            X_background: 배경 데이터

        Returns:
            np.ndarray or None:
                - 양성 클래스(1)에 대한 SHAP 값 (n_samples, n_features) 2D numpy 배열.
                - 오류 발생 시 None.
        """
        print("Calculating SHAP values for XGBoost using TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_out = explainer.shap_values(X_test)

            # XGBoost는 보통 양성 클래스 log-odds 값만 반환 (2D 배열)
            if isinstance(shap_out, np.ndarray) and shap_out.ndim == 2:
                print("Returning SHAP values (expected 2D array for positive class log-odds).")
                shap_vals = shap_out
            elif isinstance(shap_out, list) and len(shap_out) > 1:
                 # 드물게 리스트 반환 시 양성 클래스 선택
                 print("Warning: TreeExplainer returned a list for XGBoost. Selecting SHAP values for the positive class.")
                 shap_vals = shap_out[1]
            elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
                 # 매우 드물게 3D 배열 반환 시 양성 클래스 선택
                 print("Warning: TreeExplainer returned a 3D array for XGBoost. Selecting SHAP values for the positive class.")
                 shap_vals = shap_out[:, :, 1]
            else:
                print(f"Error: Unexpected SHAP values format: {type(shap_out)}. Returning None.")
                return None

            # 최종 반환값이 2D인지 확인
            if shap_vals.ndim != 2:
                 print(f"Error: Calculated SHAP values are not 2D (shape: {shap_vals.shape}). Returning None.")
                 return None

            return shap_vals

        except Exception as e:
            print(f"Error calculating SHAP values for XGBoost: {e}")
            # traceback.print_exc()
            return None

    def fit(self, X, y, **kwargs):
        """
        XGBoost 모델을 학습합니다.
        
        Args:
            X (pd.DataFrame): 학습에 사용할 특성 데이터
            y: 학습에 사용할 타겟 값
            **kwargs: 모델 학습에 필요한 추가 인수
            
        Returns:
            self: 학습된 모델 인스턴스
        """
        model = self.create_model()
        model.fit(X, y, **kwargs)
        self.model = model
        return self
        
    def predict(self, X):
        """
        학습된 XGBoost 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터
            
        Returns:
            np.ndarray: 예측값
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit 메서드를 호출하세요.")
        return self.model.predict(X)

# --- LightGBM Model Implementation ---
class LightGBMModel(ClassificationModel):
    """LightGBM 분류 모델 구현"""

    def __init__(self):
        super().__init__(name='lgb', display_name='LightGBM')

    def get_hyperparameter_grid(self):
        """LightGBM 하이퍼파라미터 탐색 공간 정의"""
        return {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 40, 50, 60], # Max number of leaves in one tree
            'max_depth': [-1, 10, 20, 30],     # Max tree depth (-1 means no limit)
            'subsample': [0.6, 0.8, 1.0],      # Also known as bagging_fraction
            'colsample_bytree': [0.6, 0.8, 1.0],# Also known as feature_fraction
            'reg_alpha': [0, 0.01, 0.1, 0.5],  # L1 regularization
            'reg_lambda': [0, 0.1, 1, 5],      # L2 regularization
            'class_weight': ['balanced', None] # Handle class imbalance
        }

    def create_model(self, params=None, n_jobs=1):
        """LightGBM 모델 인스턴스 생성"""
        if params is None:
            params = {}

        # Ensure n_jobs is handled correctly
        model_n_jobs = n_jobs if n_jobs is not None and n_jobs > 0 else -1

        # Set default parameters and update with provided params
        model_params = {
            'objective': 'binary',     # For binary classification
            'metric': 'binary_logloss', # Evaluation metric
            'random_state': 42,
            'n_jobs': model_n_jobs,
            'verbose': -1,             # Suppress verbose output unless overridden
            **params # Overwrite defaults with tuned params
        }
        
        # Remove None class_weight if selected
        if model_params.get('class_weight') is None:
             # Check if it actually exists before deleting
             if 'class_weight' in model_params:
                 del model_params['class_weight']

        return lgb.LGBMClassifier(**model_params)

    def calculate_shap_values(self, model, X_test, X_background=None):
        """
        LightGBM 모델에 대한 SHAP 값을 계산합니다. (반환값 형태 수정됨)

        Args:
            model: 학습된 LGBMClassifier 모델
            X_test: SHAP 값을 계산할 데이터 (DataFrame 권장)
            X_background: 배경 데이터

        Returns:
            np.ndarray or None:
                - 양성 클래스(1)에 대한 SHAP 값 (n_samples, n_features) 2D numpy 배열.
                - 오류 발생 시 None.
        """
        print("Calculating SHAP values for LightGBM using TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_out = explainer.shap_values(X_test)

            # LightGBM은 보통 클래스별 리스트 반환: [shap_class0, shap_class1]
            if isinstance(shap_out, list) and len(shap_out) > 1:
                print("Returning SHAP values for the positive class (from list).")
                shap_vals = shap_out[1]
            elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
                 # 드물게 3D 배열 반환 시 양성 클래스 선택
                 print("Warning: TreeExplainer returned a 3D array for LightGBM. Selecting SHAP values for the positive class.")
                 shap_vals = shap_out[:, :, 1]
            elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 2:
                 # 드물게 2D 배열 반환 시 (이미 양성 클래스 값일 수 있음)
                 print("Warning: TreeExplainer returned a 2D array for LightGBM. Assuming it relates to the positive class.")
                 shap_vals = shap_out
            else:
                print(f"Error: Unexpected SHAP values format: {type(shap_out)}. Returning None.")
                return None

            # 최종 반환값이 2D인지 확인
            if shap_vals.ndim != 2:
                 print(f"Error: Calculated SHAP values are not 2D (shape: {shap_vals.shape}). Returning None.")
                 return None

            return shap_vals

        except Exception as e:
            print(f"Error calculating SHAP values for LightGBM: {e}")
            # traceback.print_exc()
            return None

    def fit(self, X, y, **kwargs):
        """
        LightGBM 모델을 학습합니다.
        
        Args:
            X (pd.DataFrame): 학습에 사용할 특성 데이터
            y: 학습에 사용할 타겟 값
            **kwargs: 모델 학습에 필요한 추가 인수
            
        Returns:
            self: 학습된 모델 인스턴스
        """
        model = self.create_model()
        model.fit(X, y, **kwargs)
        self.model = model
        return self
        
    def predict(self, X):
        """
        학습된 LightGBM 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터
            
        Returns:
            np.ndarray: 예측값
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit 메서드를 호출하세요.")
        return self.model.predict(X)