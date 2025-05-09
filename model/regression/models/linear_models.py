import os
import joblib
import numpy as np
import pandas as pd
import shap
import logging
from scipy.stats import loguniform

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# BaseModelRegressor 임포트
from .tree_models import BaseModelRegressor

# 로깅 설정
logger = logging.getLogger(__name__)

class BaseLinearModelWithSHAP(BaseModelRegressor):
    """
    선형 모델을 위한 공통 SHAP 계산 기능이 있는 기본 클래스.
    선형 모델에 최적화된 LinearExplainer를 사용하여 SHAP 값을 효율적으로 계산합니다.
    """
    
    def _calculate_shap_values_common(self, trained_model, X_explain: pd.DataFrame, X_train_sample: pd.DataFrame = None):
        """
        모든 선형 모델에서 공통으로 사용하는 SHAP 값 계산 메서드.
        
        Args:
            trained_model: 학습된 모델 (Pipeline 또는 단일 모델)
            X_explain (pd.DataFrame): SHAP 값을 계산할 데이터셋
            X_train_sample (pd.DataFrame, optional): 배경 데이터용 훈련 샘플
        
        Returns:
            tuple: (shap_values_df: pd.DataFrame, expected_value: float or np.ndarray) - SHAP 값과 기준점
        """
        try:
            # 배경 데이터 샘플링 (최대 100개)
            if X_train_sample is None:
                if X_explain.shape[0] <= 100:
                    background = X_explain.copy()
                else:
                    n_bg = min(100, X_explain.shape[0])
                    background = shap.sample(X_explain, n_bg, random_state=42)
                logger.warning("No X_train_sample provided, using X_explain for background")
            else:
                n_bg = min(100, X_train_sample.shape[0])
                background = shap.sample(X_train_sample, n_bg, random_state=42)

            # Pipeline에서 순수 회귀 모델만 추출
            if isinstance(trained_model, Pipeline):
                # Pipeline에서 'reg' 스텝의 순수 모델 추출
                linear_model = trained_model.named_steps['reg']
                
                # 순수 linear model에 대해 explainer 생성
                explainer = shap.Explainer(
                    linear_model,  # 순수 LinearRegression 객체 전달
                    background,
                    algorithm="linear"
                )
                
                # 이미 스케일된 X_explain으로 SHAP 값 계산
                if 'scaler' in trained_model.named_steps:
                    scaler = trained_model.named_steps['scaler']
                    X_explain_scaled = pd.DataFrame(
                        scaler.transform(X_explain),
                        index=X_explain.index,
                        columns=X_explain.columns
                    )
                    shap_values = explainer(X_explain_scaled)
                else:
                    shap_values = explainer(X_explain)
            else:
                # 단일 모델의 경우 직접 사용
                explainer = shap.Explainer(
                    trained_model,  # 모델 객체 자체 전달
                    background,
                    algorithm="linear"
                )
                shap_values = explainer(X_explain)
            
            # shap_values 객체에서 값과 기준값 추출
            values = shap_values.values
            base_value = shap_values.base_values
            
            # 차원 확인 및 처리 (base_values가 배열인 경우 처리)
            if isinstance(base_value, np.ndarray) and base_value.ndim > 0:
                base_value = float(base_value.mean())

            # numpy array를 DataFrame으로 변환
            shap_df = pd.DataFrame(
                values,
                index=X_explain.index,
                columns=X_explain.columns
            )

            return shap_df, base_value

        except Exception as e:
            logger.error(f"Error calculating SHAP values with Explainer: {e}")
            # KernelExplainer로 폴백
            logger.info("Trying fallback to KernelExplainer...")
            try:
                # 샘플 크기 줄여서 속도 향상
                bg_samples = min(50, X_train_sample.shape[0] if X_train_sample is not None else X_explain.shape[0])
                background = shap.sample(X_train_sample if X_train_sample is not None else X_explain, bg_samples, random_state=42)
                
                # 예측 함수 래핑
                f = lambda x: trained_model.predict(x)
                
                # KernelExplainer 생성
                kernel_explainer = shap.KernelExplainer(f, background)
                
                # SHAP 값 계산
                kernel_shap_values = kernel_explainer.shap_values(X_explain)
                
                # numpy array를 DataFrame으로 변환
                shap_df = pd.DataFrame(
                    kernel_shap_values,
                    index=X_explain.index,
                    columns=X_explain.columns
                )
                
                logger.info("Successfully calculated SHAP values using KernelExplainer")
                return shap_df, kernel_explainer.expected_value
                
            except Exception as fallback_error:
                logger.error(f"Fallback to KernelExplainer also failed: {fallback_error}")
                if logger.level <= logging.DEBUG:
                    import traceback
                    traceback.print_exc()
                return None, None

    def fit(self, X_train, y_train, **kwargs):
        """
        회귀 모델을 훈련합니다.
        
        Args:
            X_train (pd.DataFrame): 훈련 특성
            y_train (pd.Series): 훈련 목표값
            **kwargs: 추가 훈련 매개변수
            
        Returns:
            self: 훈련된 모델 인스턴스
        """
        if self.model is None:
            self.model = self.create_model()
        
        logger.info(f"Fitting {self.get_display_name()}...")
        try:
            if hasattr(y_train, 'ravel'):
                y_train = y_train.ravel()
            self.model.fit(X_train, y_train, **kwargs)
            logger.info("Fitting completed.")
        except Exception as e:
            logger.error(f"Error during {self.get_display_name()} fitting: {e}")
            raise e
        return self

    def predict(self, X):
        """
        훈련된 모델로 예측을 수행합니다.
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터
            
        Returns:
            numpy.ndarray: 예측 결과
        """
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during {self.get_display_name()} prediction: {e}")
            raise e


class LinearRegressionModel(BaseLinearModelWithSHAP):
    """
    선형 회귀 모델 구현. 
    스케일링된 특성을 사용하여 선형 관계를 학습합니다.
    """
    
    def __init__(self, save_dir="saved_models"):
        """
        LinearRegressionModel 초기화
        
        Args:
            save_dir (str): 모델 저장 디렉토리
        """
        super().__init__(
            model_registry_name="lr",
            display_name="Linear Regression",
            save_dir=save_dir
        )
    
    def get_model_name(self):
        """모델 레지스트리에 등록할 짧은 식별자"""
        return self.model_registry_name
        
    def get_display_name(self):
        """UI 표시용 모델 이름"""
        return self.display_name
        
    def create_model(self, params=None, n_jobs=1):
        """
        스케일러가 포함된 파이프라인 모델 생성.
        
        Args:
            params (dict, optional): 모델 파라미터
            n_jobs (int): 병렬 처리를 위한 작업 수
        
        Returns:
            Pipeline: 스케일러와 선형 회귀 모델이 포함된 파이프라인
        """
        model_params = params or {}
        lr_params = {}
        # 'reg__' 접두사를 제거하여 LinearRegression에 전달
        for k, v in model_params.items():
            if k.startswith('reg__'):
                lr_params[k.replace('reg__', '')] = v
            else:
                lr_params[k] = v
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', LinearRegression(**lr_params))
        ])

    def get_hyperparameter_grid(self):
        """RandomizedSearchCV용 하이퍼파라미터 탐색 범위"""
        return {
            'reg__fit_intercept': [True, False]
        }
    
    def calculate_shap_values(self, trained_model, X_explain, X_train_sample=None):
        """
        선형 회귀 모델의 SHAP 값을 계산합니다.
        
        Args:
            trained_model: 학습된 모델 (Pipeline 또는 단일 모델)
            X_explain (pd.DataFrame): SHAP 값을 계산할 데이터셋
            X_train_sample (pd.DataFrame, optional): 배경 데이터용 훈련 샘플
            
        Returns:
            tuple: (shap_values_df: pd.DataFrame, expected_value: float or np.ndarray) - SHAP 값과 기준점
        """
        return self._calculate_shap_values_common(trained_model, X_explain, X_train_sample)


class RidgeModel(BaseLinearModelWithSHAP):
    """
    L2 정규화가 적용된 Ridge 회귀 모델 구현.
    다중공선성 문제를 완화하고 모델 과적합을 방지합니다.
    """
    
    def __init__(self, save_dir="saved_models"):
        """
        RidgeModel 초기화
        
        Args:
            save_dir (str): 모델 저장 디렉토리
        """
        super().__init__(
            model_registry_name="ridge",
            display_name="Ridge Regression",
            save_dir=save_dir
        )
    
    def get_model_name(self):
        """모델 레지스트리에 등록할 짧은 식별자"""
        return self.model_registry_name
        
    def get_display_name(self):
        """UI 표시용 모델 이름"""
        return self.display_name
        
    def create_model(self, params=None, n_jobs=1):
        """
        스케일러가 포함된 Ridge 회귀 파이프라인 생성.
        
        Args:
            params (dict, optional): 모델 파라미터
            n_jobs (int): 병렬 처리를 위한 작업 수
        
        Returns:
            Pipeline: 스케일러와 Ridge 모델이 포함된 파이프라인
        """
        model_params = params or {}
        ridge_params = {}
        for k, v in model_params.items():
            if k.startswith('reg__'):
                ridge_params[k.replace('reg__', '')] = v
            else:
                ridge_params[k] = v
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge(**ridge_params))
        ])
            
    def get_hyperparameter_grid(self):
        """
        RandomizedSearchCV용 하이퍼파라미터 탐색 범위 - 연속 분포 사용
        
        Returns:
            dict: 연속 분포가 포함된 하이퍼파라미터 그리드
        """
        return {
            'reg__alpha': loguniform(1e-3, 1e3),  # 0.001 ~ 1000 사이의 로그 스케일 분포
            'reg__fit_intercept': [True, False],
            'reg__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        }
    
    def calculate_shap_values(self, trained_model, X_explain, X_train_sample=None):
        """
        Ridge 회귀 모델의 SHAP 값을 계산합니다.
        
        Args:
            trained_model: 학습된 모델 (Pipeline 또는 단일 모델)
            X_explain (pd.DataFrame): SHAP 값을 계산할 데이터셋
            X_train_sample (pd.DataFrame, optional): 배경 데이터용 훈련 샘플
            
        Returns:
            tuple: (shap_values_df: pd.DataFrame, expected_value: float or np.ndarray) - SHAP 값과 기준점
        """
        return self._calculate_shap_values_common(trained_model, X_explain, X_train_sample)


class LassoModel(BaseLinearModelWithSHAP):
    """
    L1 정규화가 적용된 Lasso 회귀 모델 구현.
    특성 선택 기능이 있어 불필요한 특성의 계수를 0으로 만들어 희소 모델을 생성합니다.
    """
    
    def __init__(self, save_dir="saved_models"):
        """
        LassoModel 초기화
        
        Args:
            save_dir (str): 모델 저장 디렉토리
        """
        super().__init__(
            model_registry_name="lasso",
            display_name="Lasso Regression",
            save_dir=save_dir
        )
    
    def get_model_name(self):
        """모델 레지스트리에 등록할 짧은 식별자"""
        return self.model_registry_name
        
    def get_display_name(self):
        """UI 표시용 모델 이름"""
        return self.display_name
        
    def create_model(self, params=None, n_jobs=1):
        """
        스케일러가 포함된 Lasso 회귀 파이프라인 생성.
        
        Args:
            params (dict, optional): 모델 파라미터
            n_jobs (int): 병렬 처리를 위한 작업 수
        
        Returns:
            Pipeline: 스케일러와 Lasso 모델이 포함된 파이프라인
        """
        model_params = params or {}
        lasso_params = {}
        for k, v in model_params.items():
            if k.startswith('reg__'):
                lasso_params[k.replace('reg__', '')] = v
            else:
                lasso_params[k] = v
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Lasso(**lasso_params))
        ])
            
    def get_hyperparameter_grid(self):
        """
        RandomizedSearchCV용 하이퍼파라미터 탐색 범위 - 연속 분포 사용
        
        Returns:
            dict: 연속 분포가 포함된 하이퍼파라미터 그리드
        """
        return {
            'reg__alpha': loguniform(1e-3, 1e2),  # 0.001 ~ 100 사이의 로그 스케일 분포
            'reg__fit_intercept': [True, False],
            'reg__max_iter': [1000, 3000]  # Lasso는 수렴에 더 많은 반복이 필요할 수 있음
        }
    
    def calculate_shap_values(self, trained_model, X_explain, X_train_sample=None):
        """
        Lasso 회귀 모델의 SHAP 값을 계산합니다.
        
        Args:
            trained_model: 학습된 모델 (Pipeline 또는 단일 모델)
            X_explain (pd.DataFrame): SHAP 값을 계산할 데이터셋
            X_train_sample (pd.DataFrame, optional): 배경 데이터용 훈련 샘플
            
        Returns:
            tuple: (shap_values_df: pd.DataFrame, expected_value: float or np.ndarray) - SHAP 값과 기준점
        """
        return self._calculate_shap_values_common(trained_model, X_explain, X_train_sample)


class ElasticNetModel(BaseLinearModelWithSHAP):
    """
    L1과 L2 정규화를 함께 사용하는 ElasticNet 회귀 모델 구현.
    Lasso와 Ridge의 장점을 결합하여 일부 특성 선택과 함께 다중공선성 완화 효과를 제공합니다.
    """
    
    def __init__(self, save_dir="saved_models"):
        """
        ElasticNetModel 초기화
        
        Args:
            save_dir (str): 모델 저장 디렉토리
        """
        super().__init__(
            model_registry_name="en",
            display_name="Elastic Net",
            save_dir=save_dir
        )
    
    def get_model_name(self):
        """모델 레지스트리에 등록할 짧은 식별자"""
        return self.model_registry_name
        
    def get_display_name(self):
        """UI 표시용 모델 이름"""
        return self.display_name
        
    def create_model(self, params=None, n_jobs=1):
        """
        스케일러가 포함된 ElasticNet 회귀 파이프라인 생성.
        
        Args:
            params (dict, optional): 모델 파라미터
            n_jobs (int): 병렬 처리를 위한 작업 수
        
        Returns:
            Pipeline: 스케일러와 ElasticNet 모델이 포함된 파이프라인
        """
        model_params = params or {}
        en_params = {}
        for k, v in model_params.items():
            if k.startswith('reg__'):
                en_params[k.replace('reg__', '')] = v
            else:
                en_params[k] = v
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ElasticNet(**en_params))
        ])
            
    def get_hyperparameter_grid(self):
        """
        RandomizedSearchCV용 하이퍼파라미터 탐색 범위 - 연속 분포 사용
        
        Returns:
            dict: 연속 분포가 포함된 하이퍼파라미터 그리드
        """
        return {
            'reg__alpha': loguniform(1e-3, 1e2),  # 0.001 ~ 100 사이의 로그 스케일 분포
            'reg__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # L1 가중치 비율
            'reg__fit_intercept': [True, False],
            'reg__max_iter': [1000, 3000]
        }
    
    def calculate_shap_values(self, trained_model, X_explain, X_train_sample=None):
        """
        ElasticNet 회귀 모델의 SHAP 값을 계산합니다.
        
        Args:
            trained_model: 학습된 모델 (Pipeline 또는 단일 모델)
            X_explain (pd.DataFrame): SHAP 값을 계산할 데이터셋
            X_train_sample (pd.DataFrame, optional): 배경 데이터용 훈련 샘플
            
        Returns:
            tuple: (shap_values_df: pd.DataFrame, expected_value: float or np.ndarray) - SHAP 값과 기준점
        """
        return self._calculate_shap_values_common(trained_model, X_explain, X_train_sample)
