from .base_model import ClassificationModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import shap
import pandas as pd

class LogisticRegressionModel(ClassificationModel):
    def __init__(self):
        super().__init__(name='logistic', display_name='Logistic Regression')
    
    def get_hyperparameter_grid(self):
        """Logistic Regression 하이퍼파라미터 그리드를 반환합니다."""
        # Define C values on a logarithmic scale
        C_values = np.logspace(-4, 4, 10) # Reduced number for faster example if needed

        param_grid = [
            # L2 Regularization (Ridge) - Compatible solvers: 'lbfgs', 'liblinear', 'saga'
            {
                'model__penalty': ['l2'],
                'model__C': C_values,
                'model__solver': ['lbfgs', 'liblinear'], # saga removed temporarily if causing issues
                'model__class_weight': [None, 'balanced']
            },
            # L2 Regularization with 'saga' - Increase max_iter
            {
                'model__penalty': ['l2'],
                'model__C': C_values,
                'model__solver': ['saga'], 
                'model__class_weight': [None, 'balanced'],
                'model__max_iter': [1000, 2000] # <-- saga 솔버에 max_iter 추가
            },
            # L1 Regularization (Lasso) - Compatible solvers: 'liblinear', 'saga'
            {
                'model__penalty': ['l1'],
                'model__C': C_values,
                'model__solver': ['liblinear'],
                'model__class_weight': [None, 'balanced']
            },
            # L1 Regularization with 'saga' - Increase max_iter
             {
                'model__penalty': ['l1'],
                'model__C': C_values,
                'model__solver': ['saga'],
                'model__class_weight': [None, 'balanced'],
                'model__max_iter': [1000, 2000] # <-- saga 솔버에 max_iter 추가
            },
            # ElasticNet Regularization - Compatible solver: 'saga'
            {
                'model__penalty': ['elasticnet'],
                'model__C': C_values,
                'model__solver': ['saga'],
                'model__l1_ratio': np.linspace(0.1, 0.9, 5), # Example ratios
                'model__class_weight': [None, 'balanced'],
                'model__max_iter': [1000, 2000] # <-- saga 솔버에 max_iter 추가
            }
        ]
        return param_grid
    
    def create_model(self, params=None, n_jobs=1):
        """Logistic Regression 모델 인스턴스를 포함하는 Pipeline 생성"""
        if params is None:
            params = {}

        # Ensure n_jobs is handled correctly (LogisticRegression uses it directly)
        model_n_jobs = n_jobs if n_jobs is not None and n_jobs != 0 else None # None might use 1 core depending on sklearn version

        # Extract model-specific parameters from the input 'params' dict
        # Pipeline parameters look like 'step_name__parameter_name'
        model_specific_params = {k.split('__', 1)[1]: v for k, v in params.items() if k.startswith('model__')}

        # Set default parameters for LogisticRegression
        default_lr_params = {
            'random_state': 42,
            'max_iter': 1000,           # Increase max_iter for convergence, esp. with 'saga'
            'class_weight': 'balanced', # Handle class imbalance by default
            'multi_class': 'ovr',       # One-vs-Rest strategy for binary/multiclass
            'n_jobs': model_n_jobs
        }
        
        # Update defaults with the tuned parameters
        default_lr_params.update(model_specific_params)

        # Create the pipeline: StandardScaler -> LogisticRegression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(**default_lr_params))
        ])
        
        return pipeline
    
    def calculate_shap_values(self, model, X_test, X_background=None):
        """
        Logistic Regression (Pipeline) 모델에 대한 SHAP 값을 계산합니다.
        (수정됨 - predict_proba 함수 전달 방식)

        Args:
            model: 학습된 Pipeline 객체 (e.g., StandardScaler + LogisticRegression)
            X_test: SHAP 값을 계산할 원본 데이터 (DataFrame 권장)
            X_background: 배경 데이터 (원본 스케일, SHAP Explainer용 샘플)

        Returns:
            np.ndarray or None:
                - 양성 클래스(1)에 대한 SHAP 값 (n_samples, n_features) 2D numpy 배열.
                - 오류 발생 시 None.
        """
        print("Calculating SHAP values using shap.Explainer with predict_proba for Pipeline...")

        if not isinstance(model, Pipeline):
             print("Error: Expected a Pipeline object for SHAP calculation.")
             return None

        # 배경 데이터 준비
        if X_background is None:
            n_samples = 100 if len(X_test) > 100 else len(X_test)
            print(f"Warning: Background data (X_background) not provided. Using {n_samples} samples from X_test.")
            try: background_data = X_test.sample(n=n_samples, random_state=42)
            except ValueError: background_data = X_test.copy()
        else:
            background_data = X_background

        try:
            # 파이프라인의 predict_proba[:, 1] (양성 클래스 확률) 함수를 lambda로 전달
            # 배경 데이터는 masker로 전달하거나 Explainer가 처리하도록 함
            # masker = shap.maskers.Independent(background_data, max_samples=len(background_data))
            # explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], masker)

            # Masker 없이 배경 데이터만 전달 (Explainer가 내부 처리)
            explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], background_data)

            # SHAP 값 계산 (.values 속성 사용)
            # 이 경우 explainer는 양성 클래스에 대한 값만 계산할 가능성이 높음 (2D 배열 기대)
            shap_values_output = explainer(X_test)
            shap_values = shap_values_output.values

            # 반환값 형태 확인 및 2D 배열 반환
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                print("Returning 2D SHAP values array from shap.Explainer (predict_proba).")
                return shap_values
            # 혹시 Explainer가 클래스별로 반환했다면 (3D)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                 print("Warning: shap.Explainer (predict_proba) returned 3D array. Selecting positive class values.")
                 # 일반적으로 predict_proba[:,1]을 사용하면 2D가 반환되지만 안전장치로 추가
                 return shap_values[:, :, 1]
            else:
                 print(f"Error: Unexpected SHAP values format from explainer(predict_proba): {type(shap_values)}. Returning None.")
                 return None

        except Exception as e:
            print(f"Error calculating SHAP values for Logistic Regression Pipeline using explainer(predict_proba): {e}")
            # traceback.print_exc() # 디버깅 시 활성화
            return None
            
    def fit(self, X, y, **kwargs):
        """
        로지스틱 회귀 모델을 학습합니다.
        
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
        학습된 로지스틱 회귀 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터
            
        Returns:
            np.ndarray: 예측값
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. 먼저 fit 메서드를 호출하세요.")
        return self.model.predict(X)
