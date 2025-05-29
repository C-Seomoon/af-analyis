import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, StratifiedGroupKFold
import json
import traceback # For detailed error logging
import time # time 모듈 임포트
import warnings # warnings 모듈 임포트
from sklearn.exceptions import ConvergenceWarning # ConvergenceWarning 임포트 (선택적 필터링용)
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve
from collections import Counter

# --- 경고 필터링 설정 강화 ---
# 모든 사용자 경고(UserWarning) 무시
warnings.filterwarnings("ignore", category=UserWarning)

# 모든 향후 경고(FutureWarning) 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 모든 수렴 경고(ConvergenceWarning) 무시
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# SHAP 관련 모든 경고 무시
warnings.filterwarnings("ignore", module="shap")

# 특정 XGBoost 경고도 추가로 무시
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
warnings.filterwarnings("ignore", message=".*label encoder.*")

# 내부 모듈 임포트
from models.tree_models import RandomForestModel, XGBoostModel, LightGBMModel
from models.linear_models import LogisticRegressionModel
from utils.data_loader import load_and_preprocess_data
from utils.evaluation import calculate_classification_metrics, save_results, save_predictions
from utils.shap_analysis import save_shap_results, analyze_global_shap

# --- 경고 필터링 설정 ---
# 특정 경고를 무시하도록 설정합니다. 
# 실제 운영 환경에서는 주의해서 사용해야 합니다.

# SHAP LightGBM 관련 UserWarning 필터링 (이미 코드에서 처리됨)
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed*", category=UserWarning)

# SHAP LinearExplainer 관련 FutureWarning 필터링
warnings.filterwarnings("ignore", message="The feature_perturbation option is now deprecated*", category=FutureWarning)

# LogisticRegression multi_class 관련 FutureWarning 필터링
warnings.filterwarnings("ignore", message="'multi_class' was deprecated in version*", category=FutureWarning)

# ConvergenceWarning은 max_iter 증가로 해결 시도하므로 일단 필터링하지 않음
# 필요 시 아래 주석 해제하여 필터링 가능
# warnings.filterwarnings("ignore", category=ConvergenceWarning) 

def get_available_models():
    """사용 가능한 모델 객체들을 딕셔너리로 반환합니다."""
    return {
        'rf': RandomForestModel(),
        'xgb': XGBoostModel(),
        'lgb': LightGBMModel(),
        'logistic': LogisticRegressionModel()
        # Add new model classes here
    }

def run_nested_cv_classification_evaluation_only(X, y, query_ids, model_obj, output_dir, outer_folds=5, inner_folds=3, 
                 random_iter=50, n_jobs=1, random_state=42):
    """
    Nested Cross-Validation을 통한 성능 평가만 수행 (SHAP 계산 제외).
    
    Returns:
        dict: 평균 성능 지표
    """
    model_name = model_obj.get_model_name()
    model_display_name = model_obj.get_display_name()
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"\n--- Running Nested CV Evaluation for: {model_display_name} ({model_name}) ---")
    print(f"Output directory: {model_output_dir}")
    
    # 결과 저장을 위한 리스트 초기화
    all_fold_metrics = []
    all_fold_predictions = []
    feature_names = X.columns.tolist()
    
    # 외부 CV 설정
    if query_ids is not None:
        print(f"Using StratifiedGroupKFold for outer CV with {outer_folds} folds based on query IDs and stratified by y.")
        outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y, groups=query_ids)
    else:
        print(f"Using KFold for outer CV with {outer_folds} folds (no query IDs provided).")
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y)
    
    fold_times = []
    
    # --- Outer Loop ---
    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        fold_start_time = time.time()
        fold_num = fold + 1
        print(f"\n-- Processing Outer Fold {fold_num}/{outer_folds} --")
        
        # 외부 폴드 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_output_dir = os.path.join(model_output_dir, f"fold_{fold_num}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

        # 내부 CV 설정
        inner_cv_groups = query_ids.iloc[train_idx] if query_ids is not None else None
        if inner_cv_groups is not None:
            print(f"Using StratifiedGroupKFold for inner CV with {inner_folds} folds.")
            inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
            inner_cv_iterable = list(inner_cv.split(X_train, y_train, groups=inner_cv_groups))
        else:
            print(f"Using KFold for inner CV with {inner_folds} folds.")
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
            inner_cv_iterable = inner_cv
            
        # 하이퍼파라미터 튜닝
        tuning_start_time = time.time()
        try:
            print("Starting hyperparameter tuning (RandomizedSearchCV)...")
            param_grid = model_obj.get_hyperparameter_grid()
            base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs)
            
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=random_iter,
                cv=inner_cv_iterable,  
                scoring={
                    'roc_auc': 'roc_auc',
                    'pr_auc': 'average_precision',
                    'f1': 'f1',
                    'balanced_acc': 'balanced_accuracy'
                },
                refit='roc_auc',
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            best_params = search.best_params_
            best_score = search.best_score_
            best_estimator = search.best_estimator_

            print(f"Best Params found: {best_params}")
            print(f"Best Inner CV PR-AUC score: {best_score:.4f}")
            
            # 최적 파라미터 저장
            save_results(best_params, fold_output_dir, filename="best_params.json")
            
            # CV 지표 저장
            cv_res = search.cv_results_
            best_idx = search.best_index_
            cv_metrics = {
                'best_params': best_params,
                'pr_auc': cv_res['mean_test_pr_auc'][best_idx],
                'roc_auc': cv_res['mean_test_roc_auc'][best_idx],
                'f1': cv_res['mean_test_f1'][best_idx],
                'balanced_acc': cv_res['mean_test_balanced_acc'][best_idx]
            }
            save_results(cv_metrics, fold_output_dir, filename="cv_metrics.json")
            
        except Exception as e:
            print(f"Error during RandomizedSearchCV in fold {fold_num}: {e}")
            print(traceback.format_exc())
            continue
        
        tuning_end_time = time.time()
        print(f"Hyperparameter Tuning Duration: {tuning_end_time - tuning_start_time:.2f} seconds")

        # 예측 및 평가
        predict_eval_start_time = time.time()
        try:
            print("Predicting on outer test set...")
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            print("Calculating performance metrics...")
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            metrics['best_inner_cv_pr_auc'] = best_score
            all_fold_metrics.append(metrics)
            
            # 결과 저장
            save_results(metrics, fold_output_dir, filename="metrics.json")
            save_predictions(y_test, y_pred, y_prob, fold_output_dir, 
                           filename="predictions.csv", index=X_test.index)

            # 폴드 예측 결과 수집
            fold_df = pd.DataFrame({
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'fold': fold
            }, index=X_test.index)
            all_fold_predictions.append(fold_df)

            # 최적 임계값 찾기
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]

            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
            print(f"Default threshold (0.5): F1={metrics['f1']:.4f}")
            print(f"Optimal threshold ({optimal_threshold:.4f}): F1={optimal_f1:.4f}")

            # 최적 임계값 메트릭 계산
            metrics_optimal = calculate_classification_metrics(y_test, y_pred_optimal, y_prob)

            # 임계값 정보 추가
            metrics['default_threshold'] = 0.5
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_f1'] = metrics_optimal['f1']
            metrics['threshold_improvement'] = metrics_optimal['f1'] - metrics['f1']

            # 최적 임계값이 상당히 좋으면 사용
            if metrics['threshold_improvement'] > 0.05:
                print(f"Using optimal threshold ({optimal_threshold:.4f}) for better F1 score")
                metrics = metrics_optimal
                metrics['threshold_used'] = optimal_threshold
                y_pred = y_pred_optimal
            else:
                metrics['threshold_used'] = 0.5
            
        except Exception as e:
            print(f"Error during prediction or evaluation in fold {fold_num}: {e}")
            print(traceback.format_exc())
            continue
        
        predict_eval_end_time = time.time()
        print(f"Prediction & Evaluation Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        print(f"-- Outer Fold {fold_num} finished. Duration: {fold_duration:.2f} seconds --")

    # 결과 집계
    if not all_fold_metrics:
        print(f"\nError: No metrics were collected for model {model_name}.")
        return {'error': 'No metrics collected'}

    print(f"\n--- Aggregating results for: {model_display_name} ({model_name}) ---")
    avg_metrics = {}
    for metric_key in all_fold_metrics[0].keys():
        valid_values = [m[metric_key] for m in all_fold_metrics if m.get(metric_key) is not None]
        if valid_values:
            avg_metrics[f"{metric_key}_mean"] = np.mean(valid_values)
            avg_metrics[f"{metric_key}_std"] = np.std(valid_values)
        else:
            avg_metrics[f"{metric_key}_mean"] = None
            avg_metrics[f"{metric_key}_std"] = None

    # NumPy 스칼라를 Python 기본 타입으로 변환
    avg_metrics = {
        k: (v.item() if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in avg_metrics.items()
    }

    print("Average Metrics across folds:")
    print(json.dumps(avg_metrics, indent=4))
    
    # 평균 성능 지표 저장
    save_results(avg_metrics, model_output_dir, filename="metrics_summary.json")

    # 모든 폴드 예측 결과 합치기
    if all_fold_predictions:
        combined_preds = pd.concat(all_fold_predictions, ignore_index=False)
        combined_preds.to_csv(
            os.path.join(model_output_dir, "all_predictions.csv"),
            index=True, index_label="original_index"
        )
        print(f"Combined predictions saved to: {os.path.join(model_output_dir, 'all_predictions.csv')}")

    if fold_times:
        print(f"Average time per outer fold: {np.mean(fold_times):.2f} seconds")

    print(f"--- Nested CV completed for: {model_display_name} ({model_name}) ---")
    return avg_metrics

def get_best_hyperparameters_separate_cv(X, y, query_ids, model_obj, inner_folds=3, 
                                        random_iter=50, n_jobs=1, random_state=42):
    """
    전체 학습 데이터에서 별도의 CV로 최적 하이퍼파라미터를 찾습니다.
    """
    print("Finding optimal hyperparameters using separate cross-validation...")
    
    # 내부 CV 설정
    if query_ids is not None:
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        inner_cv_iterable = list(inner_cv.split(X, y, groups=query_ids))
    else:
        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        inner_cv_iterable = inner_cv
    
    # 하이퍼파라미터 탐색
    param_grid = model_obj.get_hyperparameter_grid()
    base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs)
    
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=random_iter,
        cv=inner_cv_iterable,
        scoring={
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision',
            'f1': 'f1',
            'balanced_acc': 'balanced_accuracy'
        },
        refit='pr_auc',
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # 전체 데이터로 튜닝
    search.fit(X, y)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV PR-AUC score: {search.best_score_:.4f}")
    
    return search.best_params_, search.best_score_

def train_final_classification_model_with_shap(X, y, model_obj, best_params, output_dir, n_jobs=1):
    """
    최종 분류 모델을 전체 데이터로 학습하고 SHAP 값을 계산합니다.
    """
    model_name = model_obj.get_model_name()
    print(f"\n--- Training Final Classification Model: {model_obj.get_display_name()} ---")
    print(f"Using parameters: {best_params}")
    
    # 최종 모델 생성 및 학습
    final_model = model_obj.create_model(params=best_params, n_jobs=n_jobs)
    
    print("Training final model on entire dataset...")
    start_time = time.time()
    final_model.fit(X, y)
    training_time = time.time() - start_time
    print(f"Final model training completed in {training_time:.2f} seconds")
    
    # 전체 데이터에 대한 예측 및 메트릭 계산
    print("Evaluating final model on training data...")
    y_prob_final = final_model.predict_proba(X)[:, 1]
    y_pred_final = (y_prob_final >= 0.5).astype(int)
    final_metrics = calculate_classification_metrics(y, y_pred_final, y_prob_final)
    
    # 최종 모델 메트릭 저장
    save_results(final_metrics, output_dir, filename="final_model_metrics.json")
    print(f"Final model ROC-AUC: {final_metrics['roc_auc']:.4f}, PR-AUC: {final_metrics['pr_auc']:.4f}")
    
    # 최종 예측 결과 저장
    save_predictions(y, y_pred_final, y_prob_final, output_dir, 
                    filename="final_model_predictions.csv", index=X.index)
    
    # SHAP 값 계산
    print("Calculating SHAP values for final model...")
    shap_start_time = time.time()
    try:
        shap_values = model_obj.calculate_shap_values(final_model, X)
        
        if shap_values is not None:
            feature_names = X.columns.tolist()
            
            # SHAP 결과 저장 (fold_num 없이)
            save_shap_results(shap_values, X, feature_names, output_dir, fold_num=None)
            
            # 글로벌 SHAP 분석
            analyze_global_shap(
                model_name=model_name,
                model=final_model,
                test_data_df=X,
                shap_values_np=shap_values,
                feature_names_list=feature_names,
                output_dir=output_dir
            )
            print("SHAP analysis completed successfully")
        else:
            print("SHAP values calculation failed or returned None")
            
    except Exception as e:
        print(f"Error during SHAP calculation: {e}")
        print(traceback.format_exc())
    
    shap_end_time = time.time()
    print(f"SHAP calculation completed in {shap_end_time - shap_start_time:.2f} seconds")
    
    return final_model

def compare_classification_performances(nested_cv_metrics, test_metrics):
    """
    Nested CV와 Test set 성능을 비교하고 해석합니다.
    """
    comparison = {}
    
    # 주요 메트릭들 비교
    for metric in ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'balanced_accuracy']:
        if f'{metric}_mean' in nested_cv_metrics and metric in test_metrics:
            cv_mean = nested_cv_metrics[f'{metric}_mean']
            cv_std = nested_cv_metrics[f'{metric}_std']
            test_val = test_metrics[metric]
            
            # 95% 신뢰구간 계산
            cv_lower = cv_mean - 1.96 * cv_std
            cv_upper = cv_mean + 1.96 * cv_std
            
            # Test 성능이 신뢰구간 내에 있는지 확인
            is_within_ci = cv_lower <= test_val <= cv_upper
            
            comparison[metric] = {
                'nested_cv_mean': cv_mean,
                'nested_cv_std': cv_std,
                'nested_cv_95ci': [cv_lower, cv_upper],
                'test_performance': test_val,
                'within_confidence_interval': is_within_ci,
                'difference': test_val - cv_mean,
                'relative_difference_pct': ((test_val - cv_mean) / abs(cv_mean)) * 100 if cv_mean != 0 else 0
            }
            
            # 해석
            if is_within_ci:
                interpretation = "✅ Good: Test performance within expected range"
            elif test_val > cv_upper:
                interpretation = "⚠️ Suspicious: Test performance unexpectedly high"
            else:
                interpretation = "⚠️ Concerning: Test performance below expected range"
            
            comparison[metric]['interpretation'] = interpretation
            
            print(f"{metric.upper()} Comparison:")
            print(f"  Nested CV: {cv_mean:.4f} ± {cv_std:.4f} (95% CI: [{cv_lower:.4f}, {cv_upper:.4f}])")
            print(f"  Test Set:  {test_val:.4f}")
            print(f"  {interpretation}")
    
    return comparison

def comprehensive_classification_evaluation(X_train, y_train, X_test, y_test, query_ids_train, 
                                          model_obj, args, output_dir):
    """
    포괄적인 분류 모델 평가를 수행합니다.
    """
    results = {}
    model_name = model_obj.get_model_name()
    
    # 1. Nested CV로 일반화 성능 추정
    print("=== Step 1: Nested CV Performance Estimation ===")
    nested_cv_metrics = run_nested_cv_classification_evaluation_only(
        X=X_train, y=y_train, query_ids=query_ids_train,
        model_obj=model_obj,
        output_dir=output_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        random_iter=args.random_iter,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )
    
    results['nested_cv'] = nested_cv_metrics
    
    # 2. 하이퍼파라미터 튜닝 (전체 train 데이터)
    print("=== Step 2: Hyperparameter Tuning ===")
    best_params, best_cv_score = get_best_hyperparameters_separate_cv(
        X=X_train, y=y_train, query_ids=query_ids_train,
        model_obj=model_obj,
        inner_folds=args.inner_folds,
        random_iter=args.random_iter,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )
    
    results['best_params'] = best_params
    results['hyperparameter_tuning_cv_score'] = best_cv_score
    
    # 3. 최종 모델 학습 및 SHAP
    print("=== Step 3: Final Model Training ===")
    model_output_dir = os.path.join(output_dir, model_name)
    final_model = train_final_classification_model_with_shap(
        X=X_train, y=y_train,
        model_obj=model_obj,
        best_params=best_params,
        output_dir=model_output_dir,
        n_jobs=args.n_jobs
    )
    
    # 4. Test set 평가 (만약 제공된 경우)
    if X_test is not None and y_test is not None:
        print("=== Step 4: Test Set Evaluation ===")
        y_test_prob = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_prob)
        results['test_performance'] = test_metrics
        
        # Test set 예측 결과 저장
        save_predictions(y_test, y_test_pred, y_test_prob, model_output_dir, 
                        filename="test_predictions.csv", index=X_test.index)
        
        # 5. 성능 비교
        print("=== Step 5: Performance Comparison ===")
        comparison = compare_classification_performances(nested_cv_metrics, test_metrics)
        results['performance_comparison'] = comparison
        
        # 비교 결과 저장
        save_results(comparison, model_output_dir, filename="performance_comparison.json")
    
    return results, final_model

def main():
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(description='Comprehensive Classification Framework with Nested CV')
    
    # --- Input/Output Arguments ---
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input CSV data file.')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save all results. If None, creates a timestamped directory.')
    parser.add_argument('--target_column', type=str, default='DockQ', 
                        help='Name of the column to generate the target variable from.')
    parser.add_argument('--threshold', type=float, default=0.23, 
                        help='Threshold on target_column to create binary classes.')
    parser.add_argument('--query_id_column', type=str, default='query', 
                        help='Name of the column containing group IDs for GroupKFold.')
    parser.add_argument('--drop_features', nargs='*', default=None,
                        help='List of additional feature columns to drop from X.')
    
    # --- Test Set Arguments ---
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to separate test set CSV file (optional).')
                        
    # --- Model Selection ---
    available_model_names = list(get_available_models().keys())
    parser.add_argument('--models', type=str, default=','.join(available_model_names), 
                        help=f'Comma-separated list of models to train. Available: {", ".join(available_model_names)}')

    # --- CV and Tuning Arguments ---
    parser.add_argument('--outer_folds', type=int, default=5, help='Number of outer CV folds.')
    parser.add_argument('--inner_folds', type=int, default=3, help='Number of inner CV folds.')
    parser.add_argument('--random_iter', type=int, default=50, 
                        help='Number of iterations for RandomizedSearchCV.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    
    # --- Computation Arguments ---
    parser.add_argument('--n_jobs', type=int, default=0, 
                        help='Number of CPU cores for parallel processing.')

    args = parser.parse_args()
    
    # Handle n_jobs
    n_jobs = args.n_jobs
    if n_jobs == 0:
        import multiprocessing
        try:
            n_cores = multiprocessing.cpu_count()
            n_jobs = max(1, int(n_cores * 0.75))
            print(f"Using {n_jobs} CPU cores (75% of available {n_cores}).")
        except NotImplementedError:
            n_jobs = 1
            print("Could not detect number of CPU cores. Using 1 core.")
    elif n_jobs < 0:
        n_jobs = -1
        print("Using all available CPU cores (n_jobs=-1).")
    else:
        print(f"Using {n_jobs} CPU cores.")
    
    # 중요: args.n_jobs를 업데이트해야 함
    args.n_jobs = n_jobs
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'comprehensive_classification_results_{timestamp}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Save arguments
    save_results(vars(args), args.output_dir, filename="run_arguments.json")
    
    # --- Data Loading ---
    print("\n--- Loading Data ---")
    data_load_start_time = time.time()
    try:
        X_train, y_train, query_ids_train = load_and_preprocess_data(
            file_path=args.input_file,
            target_column=args.target_column,
            threshold=args.threshold,
            query_id_column=args.query_id_column,
            user_features_to_drop=args.drop_features 
        )
        
        # 별도 테스트 세트 로딩 (선택적)
        X_test, y_test, query_ids_test = None, None, None
        if args.test_file:
            print(f"Loading separate test set from: {args.test_file}")
            X_test, y_test, query_ids_test = load_and_preprocess_data(
                file_path=args.test_file,
                target_column=args.target_column,
                threshold=args.threshold,
                query_id_column=args.query_id_column,
                user_features_to_drop=args.drop_features
            )
            print(f"Test set loaded: {X_test.shape}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        print(traceback.format_exc())
        return
    
    data_load_end_time = time.time()
    print(f"Data Loading Duration: {data_load_end_time - data_load_start_time:.2f} seconds")
        
    # --- Model Training Loop ---
    print("\n--- Starting Comprehensive Model Evaluation ---")
    all_models_summary = []
    final_models = {}
    
    selected_model_names = [name.strip() for name in args.models.split(',')]
    available_models_map = get_available_models()
    
    # Filter for valid models
    models_to_run = []
    for name in selected_model_names:
        if name in available_models_map:
            models_to_run.append(available_models_map[name])
        else:
            print(f"Warning: Model '{name}' not found. Skipping.")
            
    if not models_to_run:
        print("Error: No valid models selected. Exiting.")
        return

    # Run comprehensive evaluation for each model
    for model_obj in models_to_run:
        model_run_start_time = time.time()
        model_name = model_obj.get_model_name()
        
        try:
            print(f"\n{'='*60}")
            print(f"Processing model: {model_obj.get_display_name()} ({model_name})")
            print(f"{'='*60}")
            
            # 포괄적 모델 평가
            results, final_model = comprehensive_classification_evaluation(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                query_ids_train=query_ids_train,
                model_obj=model_obj,
                args=args,
                output_dir=args.output_dir
            )
            
            # 최종 모델 저장
            if final_model is not None:
                final_models[model_name] = {
                    'model': final_model,
                    'results': results
                }
            
            # 결과 요약
            nested_cv_results = results.get('nested_cv', {})
            model_summary = {'model_name': model_name, 'status': 'completed', **nested_cv_results}
            if 'test_performance' in results:
                for key, value in results['test_performance'].items():
                    model_summary[f'test_{key}'] = value
            
            all_models_summary.append(model_summary)
            
        except Exception as e:
            print(f"CRITICAL ERROR for model {model_name}: {e}")
            print(traceback.format_exc())
            all_models_summary.append({'model_name': model_name, 'status': 'error', 'error': str(e)})
            continue
            
        model_run_end_time = time.time()
        print(f"Total time for model '{model_name}': {model_run_end_time - model_run_start_time:.2f} seconds")

    # --- Final Summary ---
    summary_start_time = time.time()
    print("\n--- Overall Summary ---")
    
    if all_models_summary:
        # 성공한 모델들
        successful_models = [m for m in all_models_summary if m.get('status') == 'completed']
        failed_models = [m for m in all_models_summary if m.get('status') == 'error']
        
        if successful_models:
            summary_df = pd.DataFrame(successful_models)
            summary_filename = os.path.join(args.output_dir, "model_comparison_summary.csv")
            summary_df.to_csv(summary_filename, index=False)
            print(f"Model comparison summary saved to: {summary_filename}")
            print("Successful Models Summary:")
            print(summary_df.to_string())
        
        if failed_models:
            error_df = pd.DataFrame(failed_models)
            error_filename = os.path.join(args.output_dir, "model_errors_summary.csv")
            error_df.to_csv(error_filename, index=False)
            print(f"Model errors summary saved to: {error_filename}")
        
        # 최종 모델 정보 저장
        if final_models:
            models_info = {
                name: {
                    'best_params': info['results'].get('best_params', {}),
                    'nested_cv_metrics': info['results'].get('nested_cv', {}),
                    'test_metrics': info['results'].get('test_performance', {})
                } 
                for name, info in final_models.items()
            }
            save_results(models_info, args.output_dir, filename="final_models_info.json")
            print(f"Final models info saved to: {os.path.join(args.output_dir, 'final_models_info.json')}")
    
    else:
        print("No models were successfully processed.")
    
    summary_end_time = time.time()
    print(f"Summary Duration: {summary_end_time - summary_start_time:.2f} seconds")

    overall_end_time = time.time()
    print("\n--- Framework Execution Finished ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
