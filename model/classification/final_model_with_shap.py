import os
import sys
import numpy as np
import pandas as pd
import json

# 현재 스크립트의 디렉토리를 기준으로 상위 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # af_analysis-0.1.4 디렉토리
sys.path.append(parent_dir)

from model.classification.utils.shap_analysis import analyze_global_shap

# SHAP 값 로드
rf_shap_values = np.load('/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635/rf/shap_values.npy')
logistic_shap_values = np.load('/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635/logistic/shap_values.npy')

# Feature columns 로드
with open('/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635/rf/feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# 테스트 데이터의 feature 값들 로드
test_data_features = pd.read_csv('/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635/rf/test_data_features.csv')

# 데이터 shape 확인
print("RF SHAP values shape:", rf_shap_values.shape)
print("Logistic SHAP values shape:", logistic_shap_values.shape)
print("Test data features shape:", test_data_features.shape)
print("Feature columns:", feature_columns)
print("Number of features:", len(feature_columns))

# 출력 디렉토리 설정
output_dir = '/home/cseomoon/appl/af_analysis-0.1.4/model/classification/final_models/comparison_20250513_164635'

# Random Forest 모델 SHAP 분석
print("\nAnalyzing RF model...")
analyze_global_shap(
    model_name='rf',
    model=None,
    test_data_df=test_data_features,
    shap_values_np=rf_shap_values,
    feature_names_list=feature_columns,
    output_dir=output_dir + '/rf'
)

# Logistic Regression 모델 SHAP 분석
print("\nAnalyzing Logistic model...")
analyze_global_shap(
    model_name='logistic',
    model=None,
    test_data_df=test_data_features,
    shap_values_np=logistic_shap_values,
    feature_names_list=feature_columns,
    output_dir=output_dir + '/logistic'
)
