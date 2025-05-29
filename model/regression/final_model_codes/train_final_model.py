import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import json
import logging
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# regression 디렉토리를 기본 import 경로로 추가
REGRESSION_DIR = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression"
sys.path.append(REGRESSION_DIR)

# 상대 경로로 모듈 임포트
from models.tree_models import RandomForestRegressorModel
from models.linear_models import ElasticNetModel
from utils.data_loader import load_and_preprocess_data
from utils.evaluation import calculate_regression_metrics
from utils.visualization import (
    plot_actual_vs_predicted,
    plot_residuals_vs_predicted,
    plot_residual_distribution,
    plot_qq
)
from utils.shap_analysis import save_shap_results_regression
import shap

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_final_model(X_train, y_train, model_type, params, output_dir):
    """
    최종 모델을 학습하고 저장하는 함수
    
    Args:
        X_train (pd.DataFrame): 학습 데이터
        y_train (pd.Series): 학습 타겟
        model_type (str): 'rf' 또는 'en'
        params (dict): 모델 파라미터
        output_dir (str): 결과 저장 디렉토리
    
    Returns:
        trained_model: 학습된 모델
    """
    logger.info(f"\n=== Training Final {model_type.upper()} Model ===")
    
    # 모델 객체 생성
    if model_type == 'rf':
        model = RandomForestRegressorModel(save_dir=output_dir)
    elif model_type == 'en':
        model = ElasticNetModel(save_dir=output_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 모델 생성 및 학습
    trained_model = model.create_model(params=params)
    model.model = trained_model  # 모델 객체에 trained_model 할당
    model.fit(X_train, y_train)  # 모델 학습
    
    # 모델 저장
    model_path = os.path.join(output_dir, f"{model_type}_final_model.joblib")
    joblib.dump(trained_model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return trained_model

def evaluate_final_model(model, X_test, y_test, output_dir, model_type):
    """
    최종 모델을 평가하고 결과를 저장하는 함수
    
    Args:
        model: 학습된 모델
        X_test (pd.DataFrame): 테스트 데이터
        y_test (pd.Series): 테스트 타겟
        output_dir (str): 결과 저장 디렉토리
        model_type (str): 'rf' 또는 'en'
    """
    logger.info(f"\n=== Evaluating Final {model_type.upper()} Model ===")
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    metrics = calculate_regression_metrics(y_test, y_pred)
    
    # 결과 저장
    metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # 예측 결과 저장
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    }, index=X_test.index)
    predictions_path = os.path.join(output_dir, f"{model_type}_predictions.csv")
    predictions_df.to_csv(predictions_path)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # SHAP 분석
    logger.info(f"\n=== Performing SHAP Analysis for {model_type.upper()} Model ===")
    shap_dir = os.path.join(output_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)
    
    try:
        # 모델 타입에 따라 적절한 explainer 선택
        if model_type == 'rf':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:  # en
            # Pipeline에서 실제 모델 추출
            if isinstance(model, Pipeline):
                actual_model = model.named_steps['reg']  # 'reg' 스텝에서 모델 추출
                # 스케일링된 데이터로 SHAP 계산
                X_test_scaled = model.named_steps['scaler'].transform(X_test)
                explainer = shap.LinearExplainer(actual_model, X_test_scaled)
                shap_values = explainer.shap_values(X_test_scaled)
            else:
                explainer = shap.LinearExplainer(model, X_test)
                shap_values = explainer.shap_values(X_test)
        
        # SHAP 결과 저장 및 시각화
        save_shap_results_regression(
            shap_values=pd.DataFrame(shap_values, columns=X_test.columns),
            X_explain=X_test,
            feature_names=X_test.columns.tolist(),
            output_dir=shap_dir
        )
        logger.info(f"SHAP analysis results saved to: {shap_dir}")
        
    except Exception as e:
        logger.error(f"Error during SHAP analysis: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 기존 시각화
    plot_actual_vs_predicted(y_test, y_pred, output_dir, model_type)
    plot_residuals_vs_predicted(y_test, y_pred, output_dir, model_type)
    plot_residual_distribution(y_test, y_pred, output_dir, model_type)
    residuals = y_test - y_pred
    plot_qq(residuals, output_dir, model_type)
    
    return metrics

def main():
    # 데이터 경로
    train_data_path = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/data/final_data_with_rosetta_scaledRMSD_20250423.csv"
    test_data_path = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/data/ABAG_final_test_dataset_20250512.csv"
    
    # 출력 디렉토리 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = f"/home/cseomoon/appl/af_analysis-0.1.4/model/regression/final_models/{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 각 모델별 출력 디렉토리 생성
    rf_output_dir = os.path.join(base_output_dir, 'rf')
    en_output_dir = os.path.join(base_output_dir, 'en')
    os.makedirs(rf_output_dir, exist_ok=True)
    os.makedirs(en_output_dir, exist_ok=True)
    
    # 데이터 로드
    logger.info("Loading data...")
    X_train, y_train, _, _ = load_and_preprocess_data(train_data_path)
    X_test, y_test, _, _ = load_and_preprocess_data(test_data_path)
    
    # feature 순서 맞추기
    logger.info("Aligning feature order...")
    common_features = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_features]
    X_test = X_test[common_features]
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Number of common features: {len(common_features)}")
    
    # 최적 파라미터
    final_rf_params = {
        'bootstrap': True,
        'max_depth': 10,
        'max_features': 'log2',
        'min_samples_leaf': 2,
        'min_samples_split': 4,
        'n_estimators': 204
    }
    
    final_elasticnet_params = {
        'alpha': 0.09724339337447589,
        'fit_intercept': True,
        'l1_ratio': 0.3,
        'max_iter': 3000
    }
    
    # Random Forest 모델 학습 및 평가
    rf_model = train_final_model(X_train, y_train, 'rf', final_rf_params, rf_output_dir)
    rf_metrics = evaluate_final_model(rf_model, X_test, y_test, rf_output_dir, 'rf')
    
    # Elastic Net 모델 학습 및 평가
    en_model = train_final_model(X_train, y_train, 'en', final_elasticnet_params, en_output_dir)
    en_metrics = evaluate_final_model(en_model, X_test, y_test, en_output_dir, 'en')
    
    # 최종 결과 요약
    logger.info("\n=== Final Results Summary ===")
    logger.info("\nRandom Forest Metrics:")
    logger.info(json.dumps(rf_metrics, indent=4))
    logger.info("\nElastic Net Metrics:")
    logger.info(json.dumps(en_metrics, indent=4))

if __name__ == "__main__":
    main()
