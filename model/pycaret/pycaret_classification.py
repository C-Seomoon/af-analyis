# calc_classification.py

import pandas as pd
import numpy as np

# PyCaret 임포트
from pycaret.classification import (
    setup, compare_models, finalize_model,
    predict_model, pull
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, confusion_matrix, classification_report
)

# 사용자 정의 데이터 로더 (위에서 정의한 함수)
from your_loader_module import load_and_preprocess_data

def main():
    # 1) 경로 정의
    train_fp = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/train/ABAG_final_h3_plddt_20250522.csv"
    test_fp  = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/test/ABAG_final_h3_plddt_20250522.csv"
    
    # 2) 데이터 로드 및 전처리
    #    레이블: DockQ>=0.23 → 1, else 0
    X_train, y_train, groups = load_and_preprocess_data(
        train_fp,
        target_column='DockQ',
        threshold=0.23,
        query_id_column='query'
    )
    X_test,  y_test,  _      = load_and_preprocess_data(
        test_fp,
        target_column='DockQ',
        threshold=0.23,
        query_id_column='query'
    )
    
    # 3) PyCaret용 데이터프레임 준비
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    
    test_df  = X_test.copy()
    test_df ['target'] = y_test.values
    
    # 4) PyCaret setup
    clf_setup = setup(
        data=train_df,
        target='target',
        # seed 고정
        session_id=42,
        # 그룹 단위 KFold
        fold_strategy='groupkfold',
        fold=5,
        groups=groups,
        # 스케일링, 인코딩 등은 PyCaret 기본에 맡김
        normalize=True,
        silent=True,
        verbose=False
    )
    
    # 5) 모델 비교 (상위 3개 모델만 살펴보기)
    top3 = compare_models(n_select=3, sort='AUC')
    print("=== Compare Models ===")
    print(pull().head(5))
    
    # 6) 최상위 모델 최종화
    best = top3[0]
    final_model = finalize_model(best)
    
    # 7) 외부 테스트셋 예측
    pred = predict_model(final_model, data=test_df)
    
    # 8) 평가 지표 계산
    y_pred_proba = pred['Score']  # 양성 클래스 확률
    y_pred       = pred['Label']  # 0/1 예측값
    
    auc  = roc_auc_score(y_test, y_pred_proba)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n=== External Test Set Evaluation ===")
    print(f"AUC      : {auc:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()