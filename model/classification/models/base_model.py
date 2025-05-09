from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd

class ClassificationModel(ABC):
    """
    모든 분류 모델 구현을 위한 추상 기본 클래스 (Abstract Base Class).
    이 클래스는 특정 모델 구현에 필요한 공통 인터페이스를 정의합니다.
    """
    
    def __init__(self, name, display_name):
        """
        모델의 이름과 표시 이름을 초기화합니다.

        Args:
            name (str): 모델의 짧은 식별자 (예: 'rf', 'logistic').
            display_name (str): 보고서나 플롯에 사용될 모델의 전체 이름 (예: 'Random Forest', 'Logistic Regression').
        """
        self.name = name
        self.display_name = display_name

    @abstractmethod
    def get_hyperparameter_grid(self):
        """
        이 모델에 대한 하이퍼파라미터 탐색 공간을 정의합니다.
        RandomizedSearchCV 또는 GridSearchCV와 호환되는 형식으로 반환해야 합니다.

        Returns:
            dict or list[dict]: 하이퍼파라미터 이름과 탐색할 값 목록을 포함하는 사전 또는 사전 목록.
        """
        pass
    
    @abstractmethod
    def create_model(self, params=None, n_jobs=1):
        """
        주어진 하이퍼파라미터로 모델 인스턴스를 생성합니다.
        필요한 경우 전처리 단계를 포함하는 Scikit-learn Pipeline을 반환할 수 있습니다.

        Args:
            params (dict, optional): 모델 생성에 사용할 하이퍼파라미터. Defaults to None.
            n_jobs (int, optional): 병렬 처리에 사용할 CPU 코어 수. Defaults to 1.

        Returns:
            object: Scikit-learn 호환 모델 또는 파이프라인 인스턴스.
        """
        pass
    
    @abstractmethod
    def calculate_shap_values(
        self,
        model,
        X_test: pd.DataFrame,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습된 모델과 테스트 데이터를 사용하여 SHAP 값을 계산합니다.
        모델 유형(트리, 선형 등)에 적합한 SHAP explainer를 사용해야 합니다.

        Args:
            model (object): 학습된 Scikit-learn 호환 모델 또는 파이프라인.
            X_test (pd.DataFrame): SHAP 값을 계산할 테스트 데이터.
            **kwargs: SHAP explainer 또는 값 계산에 필요한 추가 인수.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - shap_values (np.ndarray): SHAP 값 배열, 보통 (n_samples, n_features)
                - base_values (np.ndarray or float): 기대값 (explainer.base_values)
        """
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y, **kwargs):
        """
        제공된 훈련 데이터로 모델을 학습합니다.

        Args:
            X (pd.DataFrame): 학습에 사용할 특성 데이터
            y: 학습에 사용할 타겟 값
            **kwargs: 모델 학습에 필요한 추가 인수

        Returns:
            self: 학습된 모델 인스턴스
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """
        학습된 모델로 예측을 수행합니다.

        Args:
            X (pd.DataFrame): 예측할 특성 데이터

        Returns:
            np.ndarray: 예측값
        """
        pass

    def get_model_name(self):
        """모델의 짧은 식별자를 반환합니다."""
        return self.name

    def get_display_name(self):
        """모델의 표시 이름을 반환합니다."""
        return self.display_name
