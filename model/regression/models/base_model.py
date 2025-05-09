from abc import ABC, abstractmethod

class RegressionModel(ABC):
    """
    회귀 모델을 위한 추상 베이스 클래스.
    모든 회귀 모델 클래스는 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, name: str, display_name: str):
        """
        모델 초기화.

        Args:
            name (str): 모델의 내부 식별자 (예: 'rf_reg', 'xgb_reg').
            display_name (str): 로그 및 결과 표시에 사용될 이름 (예: 'Random Forest Regressor').
        """
        self._name = name
        self._display_name = display_name

    def get_model_name(self) -> str:
        """모델의 내부 식별자를 반환합니다."""
        return self._name

    def get_display_name(self) -> str:
        """모델의 표시 이름을 반환합니다."""
        return self._display_name

    @abstractmethod
    def get_hyperparameter_grid(self) -> dict:
        """
        하이퍼파라미터 튜닝을 위한 파라미터 그리드를 반환합니다.
        Scikit-learn의 RandomizedSearchCV 또는 GridSearchCV 형식이어야 합니다.

        Returns:
            dict or list[dict]: 하이퍼파라미터 분포 또는 리스트.
        """
        pass

    @abstractmethod
    def create_model(self, params: dict = None, n_jobs: int = 1):
        """
        주어진 파라미터로 회귀 모델 인스턴스를 생성합니다.
        파이프라인 (예: 스케일러 포함)을 반환할 수도 있습니다.

        Args:
            params (dict, optional): 모델에 설정할 하이퍼파라미터. Defaults to None.
            n_jobs (int, optional): 병렬 처리에 사용할 CPU 코어 수. Defaults to 1.

        Returns:
            model object: 학습 가능한 Scikit-learn 호환 모델 객체.
        """
        pass

    @abstractmethod
    def calculate_shap_values(self, model, X_test):
        """
        학습된 모델과 테스트 데이터에 대한 SHAP 값을 계산합니다.
        회귀 모델의 경우, 일반적으로 단일 출력(예측값 자체에 대한 기여도)을 가집니다.

        Args:
            model: 학습된 모델 객체 (create_model에서 반환된 타입).
            X_test (pd.DataFrame or np.ndarray): SHAP 값을 계산할 테스트 데이터.

        Returns:
            np.ndarray or None: SHAP 값 배열 (보통 2D: 샘플 수 x 특성 수). 계산 실패 시 None.
        """
        pass
