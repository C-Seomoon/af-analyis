import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, average_precision_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, ParameterSampler, cross_val_score
from scipy.stats import randint, loguniform
import argparse

# 현재 스크립트의 디렉토리를 기준으로 상위 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # af_analysis-0.1.4 디렉토리
sys.path.append(parent_dir)

# 이제 model 패키지를 import할 수 있습니다
from model.classification.models.tree_models import RandomForestModel
from model.classification.models.linear_models import LogisticRegressionModel
from model.classification.utils.shap_analysis import analyze_global_shap
from model.classification.utils.evaluation import calculate_classification_metrics as calculate_metrics

# 설정
TRAIN_DATA_PATH = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/final_data_with_rosetta_scaledRMSD_20250423.csv"
TEST_DATA_PATH = "/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/ABAG_final_test_dataset_20250512.csv"
TARGET_COL = "DockQ"
QUERY_ID_COL = "query"
THRESHOLD = 0.23  # 분류 기준값
OUTPUT_DIR_BASE = f"model/classification/final_models/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RANDOM_STATE = 42



### Threshold 0.23
# # RF 최적 하이퍼파라미터
# RF_FINAL_PARAMS = {
#     'n_estimators': 200,        # 중간값 선택
#     'min_samples_split': 5,     # 가장 많이 선택됨 (3/5)
#     'min_samples_leaf': 2,      # 가장 많이 선택됨 (3/5)
#     'max_features': 'log2',     # 모든 fold에서 일관되게 선택됨
#     'max_depth': 40,            # 가장 많이 선택됨 (3/5)
#     'class_weight': 'balanced_subsample',  # 가장 많이 선택된 유효값 (2/5)
#     'n_jobs': -1  # 모든 CPU 코어 사용
# }

# # Logistic Regression 하이퍼파라미터
# LR_FINAL_PARAMS = {
#     'C': 0.005995,         # 가장 많이 선택됨 (3/5)
#     'penalty': 'l1',       # 가장 많이 선택됨 (3/5)
#     'solver': 'liblinear', # l1 penalty와 호환되는 solver 중 가장 많이 선택됨
#     'max_iter': 2000,      # 가장 많이 선택됨
#     'class_weight': None,  # 모든 fold에서 선택됨
#     'n_jobs': -1  # 모든 CPU 코어 사용
# }

###Threshold 0.49, medium model
RF_FINAL_PARAMS = {
    'n_estimators': 300,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'max_depth': 10,
    'class_weight': None,
    'n_jobs': -1
}

LR_FINAL_PARAMS = {
    'solver': 'saga',
    'penalty': 'elasticnet',
    'max_iter': 1000,
    'l1_ratio': 0.30,
    'class_weight': 'balanced',
    'C': 0.005994842503189409,
    'n_jobs': -1
}




from utils.data_loader import load_and_preprocess_data

def create_output_dir(output_dir):
    """출력 디렉토리 생성"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

def load_data(train_data_path, test_data_path, target_col, query_id_col, threshold):
    """데이터 로드 및 전처리"""
    print(f"Loading and preprocessing training data from {train_data_path}")
    X_train, y_train, _ = load_and_preprocess_data(
        file_path=train_data_path,
        target_column=target_col,
        threshold=threshold,
        query_id_column=query_id_col
    )
    
    print(f"Loading and preprocessing test data from {test_data_path}")
    X_test, y_test, _ = load_and_preprocess_data(
        file_path=test_data_path,
        target_column=target_col,
        threshold=threshold,
        query_id_column=query_id_col
    )
    
    # 특성 순서 확인 및 정렬
    train_features = set(X_train.columns)
    test_features = set(X_test.columns)
    
    if train_features != test_features:
        print("Warning: Training and test datasets have different features!")
        print("Features in training but not in test:", train_features - test_features)
        print("Features in test but not in training:", test_features - train_features)
        
        # 공통 특성만 사용
        common_features = list(train_features.intersection(test_features))
        X_train = X_train[common_features]
        X_test = X_test[common_features]
    
    # 특성 순서를 학습 데이터와 동일하게 맞춤
    X_test = X_test[X_train.columns]
    
    # 특성 스케일링
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 스케일링된 데이터를 DataFrame으로 변환
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(model_class, params, model_name, X_train, y_train, X_test, y_test, output_dir):
    """모델 학습 및 평가"""
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 모델 초기화 및 학습
    model = model_class()
    model_instance = model.create_model(params=params)
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    print(f"{model_name} model training completed")
    
    # 모델 저장
    model_path = os.path.join(model_output_dir, f"{model_name}_final_model.joblib")
    try:
        model.save_model(path=model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # 하이퍼파라미터 저장
    with open(os.path.join(model_output_dir, "best_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    # 모델 평가
    print(f"Evaluating {model_name} model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    
    # 성능 지표 계산
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    
    # 최적 임계값 찾기
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    # PR AP 계산 및 추가 (average_precision_score 사용)
    pr_ap = average_precision_score(y_test, y_pred_proba)
    metrics["pr_ap"] = pr_ap
    
    metrics["optimal_threshold"] = optimal_threshold
    metrics["optimal_f1"] = optimal_f1
    
    # 성능 지표 저장
    with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"{model_name} model evaluation completed")
    print(f"Metrics: {metrics}")
    
    # ROC 곡선 그리기
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_output_dir, "roc_curve.png"))
    
    # PR 곡선 그리기
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR curve (AP = {pr_ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.axhline(y=sum(y_test)/len(y_test), linestyle='--', color='r', label=f'Baseline (No Skill)')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(model_output_dir, "pr_curve.png"))
    
    # 예측 확률값 저장 (나중에 ROC/PR curve 비교를 위해)
    predictions = pd.DataFrame({
        'true_label': y_test,
        'pred_proba': y_pred_proba
    })
    predictions.to_csv(os.path.join(model_output_dir, "prediction_probabilities.csv"), index=False)
    
    # ROC curve 데이터 저장
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })
    roc_data.to_csv(os.path.join(model_output_dir, "roc_curve_data.csv"), index=False)
    
    # PR curve 데이터 저장
    pr_data = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'thresholds': np.append(thresholds, 1.0)  # thresholds 길이 맞추기
    })
    pr_data.to_csv(os.path.join(model_output_dir, "pr_curve_data.csv"), index=False)
    
    return model, metrics, y_pred_proba, model_output_dir

def analyze_model(model, model_name, X, y, y_pred_proba, output_dir):
    """모델 분석"""
    print(f"Analyzing {model_name} model...")
    
    # 특성 중요도 (RF만 해당)
    if hasattr(model.model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.model.feature_importances_
        })
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        feature_importances.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)
        
        # 특성 중요도 시각화
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importances['feature'][:15], feature_importances['importance'][:15])
        plt.xlabel('Importance')
        plt.title(f'{model_name} Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importances.png"))
    
    # SHAP 값 계산 및 저장 (가능한 경우)
    if hasattr(model, 'calculate_shap_values'):
        print(f"Calculating SHAP values for {model_name}...")
        try:
            shap_values = model.calculate_shap_values(model.model, X)
            if shap_values is not None:
                # SHAP 값 저장
                np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
                # Feature columns 저장
                feature_columns = X.columns.tolist()
                with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
                    json.dump(feature_columns, f)
                # 테스트 데이터의 feature 값들 저장
                X.to_csv(os.path.join(output_dir, "test_data_features.csv"), index=False)
                print("SHAP values, feature columns, and test data features saved")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
    
    # 예측 결과 저장
    predictions = pd.DataFrame({
        'true_label': y,
        'pred_proba': y_pred_proba,
        'pred_label': (y_pred_proba >= 0.5).astype(int)
    })
    
    # 최적 임계값 적용 (metrics.json에서 로드)
    with open(os.path.join(output_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    predictions['optimal_pred_label'] = (y_pred_proba >= metrics["optimal_threshold"]).astype(int)
    predictions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    print(f"{model_name} model analysis completed")

def compare_models(rf_metrics, lr_metrics, output_dir):
    """모델 간 성능 비교"""
    print("Comparing model performance...")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
               'balanced_accuracy', 'mcc', 'optimal_f1', 'optimal_threshold', 'pr_ap']
    
    comparison = pd.DataFrame({
        'Metric': metrics,
        'RandomForest': [rf_metrics.get(m, 'N/A') for m in metrics],
        'LogisticRegression': [lr_metrics.get(m, 'N/A') for m in metrics]
    })
    
    comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # 모델 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'optimal_f1']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    rf_values = [rf_metrics.get(m, 0) for m in metrics_to_plot]
    lr_values = [lr_metrics.get(m, 0) for m in metrics_to_plot]
    
    plt.bar(x - width/2, rf_values, width, label='Random Forest')
    plt.bar(x + width/2, lr_values, width, label='Logistic Regression')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    print("Model comparison completed")

def fine_tune_model(model_class, base_params, X_train, y_train, model_name, output_dir, random_state=42, n_iter=20, cv_splits=3):
    """
    특정 모델의 하이퍼파라미터를 fine-tuning합니다.
    """
    print(f"\n=== Fine-tuning {model_name} model ===")
    
    # 1) 출력 폴더 및 기본 설정
    tune_dir = os.path.join(output_dir, model_name, "fine_tuning")
    os.makedirs(tune_dir, exist_ok=True)
    
    base_params_copy = base_params.copy()
    if 'random_state' not in base_params_copy:
        base_params_copy['random_state'] = random_state
    
    # 2) CV 설정
    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = 'roc_auc'
    
    # 3) estimator & param_dist 정의
    if model_name == 'rf':
        # RF 모델 및 파라미터 분포 정의
        estimator = model_class().create_model(params=base_params_copy)
        param_dist = {
            'n_estimators': randint(int(base_params_copy['n_estimators']*0.7), 
                                  int(base_params_copy['n_estimators']*1.3)),
            'min_samples_split': randint(max(2, int(base_params_copy['min_samples_split']*0.7)),
                                       int(base_params_copy['min_samples_split']*1.3)),
            'min_samples_leaf': randint(1, int(base_params_copy['min_samples_leaf']*2)+1),
            'max_depth': randint(max(5, int(base_params_copy['max_depth']*0.7)),
                               int(base_params_copy['max_depth']*1.3)),
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'random_state': [random_state]
        }
    else:  # logistic
        # Logistic 모델 인스턴스 생성
        model_instance = model_class()
        estimator = model_instance.create_model(params=base_params_copy)
        
        # 파이프라인 prefix 확인
        param_prefix = ""
        if hasattr(estimator, 'steps'):
            for step_name, _ in estimator.steps:
                if step_name == 'model':
                    param_prefix = "model__"
                    break
        
        # C 값 설정
        C0 = base_params_copy['C'] if 'C' in base_params_copy else base_params_copy.get(f'{param_prefix}C', 0.005995)
        
        # 파라미터 분포 정의 - liblinear solver로 통일
        param_dist = {
            f'{param_prefix}solver': ['liblinear'],
            f'{param_prefix}penalty': ['l1', 'l2'],
            f'{param_prefix}C': loguniform(C0*0.1, C0*10),
            f'{param_prefix}max_iter': [2000],
            f'{param_prefix}class_weight': [None, 'balanced'],
            f'{param_prefix}random_state': [random_state]
        }
    
    # 4) 베이스라인 성능 계산
    print("Calculating baseline performance with original parameters...")
    baseline_scores = cross_val_score(estimator, X_train, y_train, 
                                    cv=inner_cv, scoring=scoring, n_jobs=-1)
    baseline_score = baseline_scores.mean()
    print(f"Baseline {scoring} score: {baseline_score:.4f}")
    
    # 5) RandomizedSearchCV 생성 및 실행
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=inner_cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    
    print(f"Starting RandomizedSearchCV for {model_name} with {n_iter} iterations...")
    search.fit(X_train, y_train)
    best_score = search.best_score_
    print(f"Fine-tuning completed. Best score: {best_score:.4f}")
    
    # 성능 개선 계산
    improvement = best_score - baseline_score
    improvement_percent = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
    print(f"Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")
    
    # 6) 결과 저장
    best_params = search.best_params_
    print(f"Best parameters: {best_params}")
    
    # prefix 제거 (logistic regression의 경우)
    if model_name == 'logistic' and param_prefix:
        clean_best_params = {}
        for k, v in best_params.items():
            if k.startswith(param_prefix):
                clean_key = k[len(param_prefix):]
                clean_best_params[clean_key] = v
            else:
                clean_best_params[k] = v
        best_params = clean_best_params
    
    # 결과 저장
    with open(os.path.join(tune_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # CV 결과 저장
    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(tune_dir, "search_results.csv"), index=False
    )
    
    # 베이스라인과 최적 점수 비교 막대 그래프
    plt.figure(figsize=(8, 6))
    plt.bar(['Baseline', 'Fine-tuned'], [baseline_score, best_score])
    plt.title(f'Performance Improvement ({scoring})')
    plt.ylabel(f'{scoring} Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.annotate(f'+{improvement_percent:.2f}%', 
                xy=(1, best_score), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(tune_dir, "performance_improvement.png"))
    plt.close()
    
    # 결과 정보 종합
    tuning_results = {
        "baseline_score": baseline_score,
        "best_score": best_score,
        "improvement": improvement,
        "improvement_percent": improvement_percent,
        "best_params": best_params,
        "scoring": scoring,
        "cv_splits": cv_splits,
        "n_iterations": n_iter
    }
    
    with open(os.path.join(tune_dir, "tuning_results.json"), 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    # 성능이 개선되지 않은 경우 원래 파라미터 반환
    if improvement <= 0:
        print(f"Warning: Fine-tuning did not improve performance. Keeping original parameters.")
        return base_params_copy
    
    return best_params

class ModelTrainer:
    def __init__(self, output_dir, random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        
    def add_model(self, name, model_class, params):
        """모델 추가"""
        self.models[name] = {
            'class': model_class,
            'params': params
        }
    
    def fine_tune_models(self, X_train, y_train, n_iter=20, cv_splits=3):
        """모든 모델의 하이퍼파라미터를 fine-tuning"""
        print("\n=== Starting model fine-tuning ===")
        
        for model_name, model_info in self.models.items():
            try:
                best_params = fine_tune_model(
                    model_info['class'], model_info['params'],
                    X_train, y_train, model_name, self.output_dir, 
                    random_state=self.random_state,
                    n_iter=n_iter, cv_splits=cv_splits
                )
                
                # 최적 파라미터 업데이트
                self.models[model_name]['params'].update(best_params)
                print(f"{model_name} model parameters updated with fine-tuned values")
                
                # 업데이트된 파라미터 저장
                model_output_dir = os.path.join(self.output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                with open(os.path.join(model_output_dir, "fine_tuned_params.json"), "w") as f:
                    json.dump(self.models[model_name]['params'], f, indent=4)
                
            except Exception as e:
                print(f"Error in fine-tuning {model_name}: {str(e)}")
                print(f"Using original parameters for {model_name}")
        
        print("\n=== Model fine-tuning completed ===")

    def train_single_model(self, model_name, X_train, y_train, X_test, y_test):
        """단일 모델 학습 및 평가"""
        try:
            model_info = self.models[model_name]
            model = model_info['class']()
            model_instance = model.create_model(params=model_info['params'])
            
            # 모델별 출력 디렉토리 생성
            model_output_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # 모델 학습
            print(f"Training {model_name} model...")
            model.fit(X_train, y_train)
            
            # 모델 저장
            self._save_model(model.model, model_name, model_output_dir)
            
            # 모델 평가
            metrics, y_pred_proba = self._evaluate_model(
                model, X_test, y_test, model_name, model_output_dir
            )
            
            # 모델 분석
            self._analyze_model(
                model, model_name, X_test, y_test, y_pred_proba, model_output_dir
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error in training {model_name}: {str(e)}")
            return None
    
    def _save_model(self, model, model_name, output_dir):
        """모델 저장"""
        import joblib
        model_path = os.path.join(output_dir, f"{model_name}_final_model.joblib")
        try:
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _evaluate_model(self, model, X_test, y_test, model_name, output_dir):
        """모델 평가"""
        print(f"Evaluating {model_name} model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.model.predict_proba(X_test)[:, 1]
        
        # 성능 지표 계산
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        
        # 최적 임계값 찾기
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        # PR AP 계산 및 추가 (average_precision_score 사용)
        pr_ap = average_precision_score(y_test, y_pred_proba)
        metrics["pr_ap"] = pr_ap
        
        metrics["optimal_threshold"] = optimal_threshold
        metrics["optimal_f1"] = optimal_f1
        
        # 성능 지표 저장
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"{model_name} model evaluation completed")
        print(f"Metrics: {metrics}")
        
        # ROC 곡선 그리기
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        
        # PR 곡선 그리기
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR curve (AP = {pr_ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.axhline(y=sum(y_test)/len(y_test), linestyle='--', color='r', label=f'Baseline (No Skill)')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(output_dir, "pr_curve.png"))
        
        # 예측 확률값 저장 (나중에 ROC/PR curve 비교를 위해)
        predictions = pd.DataFrame({
            'true_label': y_test,
            'pred_proba': y_pred_proba
        })
        predictions.to_csv(os.path.join(output_dir, "prediction_probabilities.csv"), index=False)
        
        # ROC curve 데이터 저장
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr
        })
        roc_data.to_csv(os.path.join(output_dir, "roc_curve_data.csv"), index=False)
        
        # PR curve 데이터 저장
        pr_data = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(thresholds, 1.0)  # thresholds 길이 맞추기
        })
        pr_data.to_csv(os.path.join(output_dir, "pr_curve_data.csv"), index=False)
        
        return metrics, y_pred_proba
    
    def _analyze_model(self, model, model_name, X, y, y_pred_proba, output_dir):
        """모델 분석"""
        print(f"Analyzing {model_name} model...")
        
        # 특성 중요도 (RF만 해당)
        if hasattr(model.model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.model.feature_importances_
            })
            feature_importances = feature_importances.sort_values('importance', ascending=False)
            feature_importances.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)
            
            # 특성 중요도 시각화
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importances['feature'][:15], feature_importances['importance'][:15])
            plt.xlabel('Importance')
            plt.title(f'{model_name} Top 15 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importances.png"))
        
        # SHAP 값 계산 및 저장 (가능한 경우)
        if hasattr(model, 'calculate_shap_values'):
            print(f"Calculating SHAP values for {model_name}...")
            try:
                shap_values = model.calculate_shap_values(model.model, X)
                if shap_values is not None:
                    # SHAP 값 저장
                    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
                    # Feature columns 저장
                    feature_columns = X.columns.tolist()
                    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
                        json.dump(feature_columns, f)
                    # 테스트 데이터의 feature 값들 저장
                    X.to_csv(os.path.join(output_dir, "test_data_features.csv"), index=False)
                    print("SHAP values, feature columns, and test data features saved")
            except Exception as e:
                print(f"Error calculating SHAP values: {e}")
        
        # 예측 결과 저장
        predictions = pd.DataFrame({
            'true_label': y,
            'pred_proba': y_pred_proba,
            'pred_label': (y_pred_proba >= 0.5).astype(int)
        })
        
        # 최적 임계값 적용 (metrics.json에서 로드)
        with open(os.path.join(output_dir, "metrics.json"), "r") as f:
            metrics = json.load(f)
        
        predictions['optimal_pred_label'] = (y_pred_proba >= metrics["optimal_threshold"]).astype(int)
        predictions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        
        print(f"{model_name} model analysis completed")
    
    def _compare_models(self):
        """모델 간 성능 비교"""
        if len(self.metrics) < 2:
            return
            
        print("Comparing model performance...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                   'balanced_accuracy', 'mcc', 'optimal_f1', 'optimal_threshold', 'pr_ap']
        
        comparison = pd.DataFrame({
            'Metric': metrics,
            **{name: [metrics.get(m, 'N/A') for m in metrics] 
               for name, metrics in self.metrics.items()}
        })
        
        comparison.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
        
        # 모델 성능 비교 시각화
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'optimal_f1']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(self.metrics)
        
        for i, (name, metrics) in enumerate(self.metrics.items()):
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            plt.bar(x + i*width, values, width, label=name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*(len(self.metrics)-1)/2, metrics_to_plot)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        
        print("Model comparison completed")

        # 모든 모델의 예측 확률값을 하나의 파일로 저장
        all_predictions = {}
        for model_name in self.metrics:
            pred_file = os.path.join(self.output_dir, model_name, "prediction_probabilities.csv")
            if os.path.exists(pred_file):
                all_predictions[model_name] = pd.read_csv(pred_file)
        
        if all_predictions:
            combined_predictions = pd.DataFrame({
                'true_label': all_predictions[list(all_predictions.keys())[0]]['true_label']
            })
            for model_name, preds in all_predictions.items():
                combined_predictions[f'{model_name}_pred_proba'] = preds['pred_proba']
            
            combined_predictions.to_csv(
                os.path.join(self.output_dir, "all_models_predictions.csv"), 
                index=False
            )
        
        # 모든 모델의 ROC curve 데이터를 하나의 파일로 저장
        all_roc_data = {}
        for model_name in self.metrics:
            roc_file = os.path.join(self.output_dir, model_name, "roc_curve_data.csv")
            if os.path.exists(roc_file):
                all_roc_data[model_name] = pd.read_csv(roc_file)
        
        if all_roc_data:
            combined_roc_data = pd.DataFrame()
            for model_name, roc_data in all_roc_data.items():
                combined_roc_data[f'{model_name}_fpr'] = roc_data['fpr']
                combined_roc_data[f'{model_name}_tpr'] = roc_data['tpr']
            
            combined_roc_data.to_csv(
                os.path.join(self.output_dir, "all_models_roc_data.csv"), 
                index=False
            )
        
        # 모든 모델의 PR curve 데이터를 하나의 파일로 저장
        all_pr_data = {}
        for model_name in self.metrics:
            pr_file = os.path.join(self.output_dir, model_name, "pr_curve_data.csv")
            if os.path.exists(pr_file):
                all_pr_data[model_name] = pd.read_csv(pr_file)
        
        if all_pr_data:
            combined_pr_data = pd.DataFrame()
            for model_name, pr_data in all_pr_data.items():
                combined_pr_data[f'{model_name}_precision'] = pr_data['precision']
                combined_pr_data[f'{model_name}_recall'] = pr_data['recall']
                combined_pr_data[f'{model_name}_thresholds'] = pr_data['thresholds']
            
            combined_pr_data.to_csv(
                os.path.join(self.output_dir, "all_models_pr_data.csv"), 
                index=False
            )

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """모든 모델 학습 및 평가"""
        print("\n=== Starting model training and evaluation ===")
        
        for model_name in self.models:
            print(f"\n=== Training {model_name} model ===")
            metrics = self.train_single_model(
                model_name, X_train, y_train, X_test, y_test
            )
            if metrics:
                self.metrics[model_name] = metrics
                print(f"\n{model_name} model training and evaluation completed")
            else:
                print(f"\n{model_name} model training failed")
        
        # 모델 비교
        if len(self.metrics) > 1:
            print("\n=== Comparing model performance ===")
            self._compare_models()
        
        print("\n=== All models training and evaluation completed ===")
        print(f"Results saved to: {self.output_dir}")
        
        # 각 모델의 출력 디렉토리 출력
        for model_name in self.models:
            model_dir = os.path.join(self.output_dir, model_name)
            print(f"{model_name} model directory: {model_dir}")

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Train and evaluate classification models')
    
    # 데이터 경로
    parser.add_argument('--train-data', type=str, 
                       default="/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/final_data_with_rosetta_scaledRMSD_20250423.csv",
                       help='Path to training data CSV file')
    parser.add_argument('--test-data', type=str,
                       default="/home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/ABAG_final_test_dataset_20250512.csv",
                       help='Path to test data CSV file')
    
    # 모델 설정
    parser.add_argument('--target-col', type=str, default='DockQ',
                       help='Target column name')
    parser.add_argument('--query-id-col', type=str, default='query',
                       help='Query ID column name')
    parser.add_argument('--threshold', type=float, default=0.23,
                       help='Classification threshold')
    
    # 출력 설정
    parser.add_argument('--output-dir', type=str,
                       help='Output directory path (default: auto-generated)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # 모델 선택
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['rf', 'logistic'],
                       choices=['rf', 'logistic'],
                       help='Models to train (rf and/or logistic)')
    
    # CPU 코어 수 설정
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of CPU cores to use (-1 for all cores)')
    
    # Fine-tuning 관련 인자 추가
    parser.add_argument('--fine-tune', action='store_true',
                       help='Perform hyperparameter fine-tuning')
    parser.add_argument('--fine-tune-iter', type=int, default=20,
                       help='Number of parameter configurations to try in RandomizedSearchCV')
    parser.add_argument('--fine-tune-cv', type=int, default=3,
                       help='Number of cross-validation splits for fine-tuning')
    
    return parser.parse_args()

def main():
    # 명령행 인자 파싱
    args = parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "final_models", 
                                f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 데이터 로드
    X_train, X_test, y_train, y_test = load_data(
        args.train_data, args.test_data, 
        args.target_col, args.query_id_col, 
        args.threshold
    )
    
    # 모델 트레이너 초기화
    trainer = ModelTrainer(output_dir, random_state=args.random_state)
    
    # 모델 추가
    if 'rf' in args.models:
        rf_params = RF_FINAL_PARAMS.copy()
        rf_params['n_jobs'] = args.n_jobs
        trainer.add_model("rf", RandomForestModel, rf_params)
    
    if 'logistic' in args.models:
        lr_params = LR_FINAL_PARAMS.copy()
        lr_params['n_jobs'] = args.n_jobs
        trainer.add_model("logistic", LogisticRegressionModel, lr_params)
    
    # Fine-tuning 수행 (요청된 경우)
    if args.fine_tune:
        trainer.fine_tune_models(X_train, y_train, 
                                n_iter=args.fine_tune_iter, 
                                cv_splits=args.fine_tune_cv)
    
    # 모델 학습 및 평가
    trainer.train_all_models(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
