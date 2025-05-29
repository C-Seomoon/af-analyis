import os
import sys
import json
import pandas as pd

# classification 디렉토리를 Python 경로에 추가
classification_dir = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification"
sys.path.append(classification_dir)

from utils.final_model_evaluation import ModelEvaluator

def analyze_results(comparison_dir):
    """저장된 모델 결과 분석"""
    # 모델별 디렉토리
    model_dirs = ['rf', 'logistic']
    
    for model_name in model_dirs:
        model_dir = os.path.join(comparison_dir, model_name)
        if not os.path.exists(model_dir):
            continue
            
        # 예측 결과 로드
        predictions = pd.read_csv(os.path.join(model_dir, "predictions.csv"))
        with open(os.path.join(model_dir, "metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        # 평가 수행
        evaluator = ModelEvaluator(model_dir)
        evaluator.evaluate_model(
            model_name=model_name,
            y_true=predictions['true_label'],
            y_pred=predictions['optimal_pred_label'],
            y_pred_proba=predictions['pred_proba'],
            metrics=metrics
        )

if __name__ == "__main__":
    comparison_dir = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635"
    analyze_results(comparison_dir)
