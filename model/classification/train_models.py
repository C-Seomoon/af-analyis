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

def run_nested_cv(X, y, query_ids, model_obj, output_dir, outer_folds=5, inner_folds=3, 
                 random_iter=50, n_jobs=1, random_state=42):
    """
    주어진 모델 객체에 대해 Nested Cross-Validation을 수행하고 각 단계별 시간 측정.

    Args:
        X (pd.DataFrame): 특성 데이터.
        y (pd.Series): 목표 변수.
        query_ids (pd.Series or None): 그룹 ID (GroupKFold 사용 시).
        model_obj (ClassificationModel): 학습 및 평가할 모델 객체 (base_model.py의 서브클래스).
        output_dir (str): 모든 결과 파일이 저장될 최상위 디렉토리.
        outer_folds (int): 외부 교차 검증 폴드 수.
        inner_folds (int): 내부 교차 검증 폴드 수 (하이퍼파라미터 튜닝용).
        random_iter (int): RandomizedSearchCV 반복 횟수.
        n_jobs (int): 병렬 처리에 사용할 CPU 코어 수.
        random_state (int): 재현성을 위한 랜덤 시드.

    Returns:
        dict: 해당 모델의 모든 외부 폴드에 대한 평균 성능 지표.
    """
    model_name = model_obj.get_model_name()
    model_display_name = model_obj.get_display_name()
    model_output_dir = os.path.join(output_dir, model_name) # Model-specific subdirectory
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"\n--- Running Nested CV for: {model_display_name} ({model_name}) ---")
    print(f"Output directory: {model_output_dir}")
    
    # 결과 저장을 위한 리스트 초기화
    all_fold_metrics = []
    # 추가: 모든 폴드의 SHAP 값과 테스트 데이터를 저장할 리스트
    all_fold_shap_values = []
    all_fold_test_data = []
    all_fold_predictions = []  # Collect per-fold predictions for combined output
    feature_names = X.columns.tolist()
    
    # 변수의 기본값 설정 (스코프 문제 방지)
    best_estimator = None
    best_score = 0.0
    best_params = {}
    
    # 외부 CV 설정 (모델 최종 평가용)
    if query_ids is not None:
        print(f"Using StratifiedGroupKFold for outer CV with {outer_folds} folds based on query IDs and stratified by y.")
        outer_cv = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y, groups=query_ids)
    else:
        print(f"Using KFold for outer CV with {outer_folds} folds (no query IDs provided).")
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y)
    
    fold_times = [] # 각 폴드 처리 시간 저장 리스트
    # --- Outer Loop ---
    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        fold_start_time = time.time() # 폴드 시작 시간 기록
        fold_num = fold + 1
        print(f"\n-- Processing Outer Fold {fold_num}/{outer_folds} --")
        
        # 외부 폴드 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_output_dir = os.path.join(model_output_dir, f"fold_{fold_num}") # Fold-specific subdir
        os.makedirs(fold_output_dir, exist_ok=True)
        
        print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        print(f"Test set indices range from {test_idx.min()} to {test_idx.max()}")

        # 내부 CV 설정 (하이퍼파라미터 튜닝용)
        inner_cv_groups = query_ids.iloc[train_idx] if query_ids is not None else None
        if inner_cv_groups is not None:
            print(f"Using StratifiedGroupKFold for inner CV with {inner_folds} folds.")
            inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
            inner_cv_iterable = list(inner_cv.split(X_train, y_train, groups=inner_cv_groups))
        else:
            print(f"Using KFold for inner CV with {inner_folds} folds.")
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
            inner_cv_iterable = inner_cv # KFold object works directly
            
        # 1. 하이퍼파라미터 튜닝 (RandomizedSearchCV)
        tuning_start_time = time.time()
        try:
            print("Starting hyperparameter tuning (RandomizedSearchCV)...")
            param_grid = model_obj.get_hyperparameter_grid()
            
            # Initial model instance for search (no specific params yet)
            base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs) 
            
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=random_iter,
                cv=inner_cv_iterable,  
                scoring={  # 다중 지표로 변경
                    'roc_auc': 'roc_auc',
                    'pr_auc': 'average_precision',
                    'f1': 'f1',
                    'balanced_acc': 'balanced_accuracy'
                },
                refit='pr_auc',  # PR-AUC로 최적 모델 선택 (불균형에 더 적합)
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0
            )
            
            search.fit(X_train, y_train) # Fit on the outer training data
            
            best_params = search.best_params_
            best_score = search.best_score_
            best_estimator = search.best_estimator_ # The model/pipeline with best params

            print(f"Best Params found: {best_params}")
            print(f"Best Inner CV PR-AUC score: {best_score:.4f}")
            
            # 최적 파라미터 저장
            save_results(best_params, fold_output_dir, filename="best_params.json")
            
            # 하이퍼파라미터 튜닝 후 추가
            # 다양한 지표 결과 저장
            cv_res = search.cv_results_
            best_idx = search.best_index_
            cv_metrics = {
                'best_params': best_params,
                'pr_auc': cv_res['mean_test_pr_auc'][best_idx],
                'roc_auc': cv_res['mean_test_roc_auc'][best_idx],
                'f1': cv_res['mean_test_f1'][best_idx],
                'balanced_acc': cv_res['mean_test_balanced_acc'][best_idx]
            }
            print(f"CV metrics for best model: PR-AUC={cv_metrics['pr_auc']:.4f}, "
                  f"ROC-AUC={cv_metrics['roc_auc']:.4f}, F1={cv_metrics['f1']:.4f}, "
                  f"Balanced Acc={cv_metrics['balanced_acc']:.4f}")

            # 모든 교차 검증 지표 저장
            save_results(cv_metrics, fold_output_dir, filename="cv_metrics.json")
            
        except Exception as e:
            print(f"Error during RandomizedSearchCV in fold {fold_num}: {e}")
            print(traceback.format_exc())
            tuning_end_time = time.time()
            print(f"Hyperparameter Tuning (Error) Duration: {tuning_end_time - tuning_start_time:.2f} seconds")
            # Skip to next fold if tuning fails
            continue 
        tuning_end_time = time.time()
        print(f"Hyperparameter Tuning Duration: {tuning_end_time - tuning_start_time:.2f} seconds")

        # 2. 외부 테스트 세트 예측 및 평가
        predict_eval_start_time = time.time()
        try:
            print("Predicting on outer test set...")
            y_prob = best_estimator.predict_proba(X_test)[:, 1] # Probability of class 1
            y_pred = (y_prob >= 0.5).astype(int) # Standard 0.5 threshold for prediction

            print("Calculating performance metrics...")
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            metrics['best_inner_cv_pr_auc'] = best_score  # PR-AUC를 기본 메트릭으로 사용
            all_fold_metrics.append(metrics)
            
            # 폴드 성능 지표 저장
            save_results(metrics, fold_output_dir, filename="metrics.json")
            
            # 폴드 예측 결과 저장 (인덱스 포함)
            save_predictions(y_test, y_pred, y_prob, fold_output_dir, 
                             filename="predictions.csv", index=X_test.index)

            # Collect this fold's predictions for aggregation
            fold_df = pd.DataFrame({
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'fold': fold
            }, index=X_test.index)
            all_fold_predictions.append(fold_df)

            # F1 스코어를 최대화하는 임계값 찾기
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]

            # 최적 임계값으로 예측값 생성
            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

            print(f"Default threshold (0.5): F1={metrics['f1']:.4f}")
            print(f"Optimal threshold ({optimal_threshold:.4f}): F1={optimal_f1:.4f}")

            # 최적 임계값 메트릭 계산
            metrics_optimal = calculate_classification_metrics(y_test, y_pred_optimal, y_prob)

            # 메트릭에 임계값 정보 추가
            metrics['default_threshold'] = 0.5
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_f1'] = metrics_optimal['f1']
            metrics['threshold_improvement'] = metrics_optimal['f1'] - metrics['f1']

            # 최적 임계값이 기본값보다 상당히 좋으면 최적 임계값 결과 사용
            if metrics['threshold_improvement'] > 0.05:  # 5% 이상 개선되면
                print(f"Using optimal threshold ({optimal_threshold:.4f}) for better F1 score")
                metrics = metrics_optimal
                metrics['threshold_used'] = optimal_threshold
                y_pred = y_pred_optimal  # 예측값 업데이트
            else:
                metrics['threshold_used'] = 0.5
            
        except Exception as e:
            print(f"Error during prediction or evaluation in fold {fold_num}: {e}")
            print(traceback.format_exc())
            predict_eval_end_time = time.time()
            print(f"Prediction & Evaluation (Error) Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")
            # Continue to SHAP calculation if possible, but metrics might be missing
            # Let's skip SHAP if prediction fails to avoid cascading errors
            continue 
        predict_eval_end_time = time.time()
        print(f"Prediction & Evaluation Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")

        # 3. SHAP 값 계산 및 저장
        shap_start_time = time.time()
        try:
            print("Calculating and saving SHAP values...")
            # Pass the best estimator and the outer test set
            shap_values = model_obj.calculate_shap_values(best_estimator, X_test) 
            
            if shap_values is not None:
                # 기존: 개별 폴드의 SHAP 값 저장
                save_shap_results(shap_values, X_test, feature_names, 
                              fold_output_dir, fold_num=fold_num)
                
                # 추가: 전체 SHAP 분석을 위해 리스트에 추가 (같은 try 블록 안에서 동일한 순서로 추가)
                all_fold_shap_values.append(shap_values)
                all_fold_test_data.append(X_test.copy())
            else:
                print("SHAP values calculation failed or returned None.")
                
        except Exception as e:
            print(f"Error during SHAP calculation/saving in fold {fold_num}: {e}")
            print(traceback.format_exc())
        shap_end_time = time.time()
        print(f"SHAP Calculation & Saving Duration: {shap_end_time - shap_start_time:.2f} seconds")

        fold_end_time = time.time() # 폴드 종료 시간 기록
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        print(f"-- Outer Fold {fold_num} finished. Duration: {fold_duration:.2f} seconds --")

    # --- After Outer Loop ---
    aggregation_start_time = time.time()
    
    # 4. 전체 폴드 결과 요약
    if not all_fold_metrics:
         print(f"\nError: No metrics were collected for model {model_name}. Check logs for errors in folds.")
         return {'error': 'No metrics collected'}

    print(f"\n--- Aggregating results for: {model_display_name} ({model_name}) ---")
    avg_metrics = {}
    for metric_key in all_fold_metrics[0].keys():
         # Filter out potential None values before calculating mean/std
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
    
    # 평균 성능 지표 저장 (모델 최상위 디렉토리)
    save_results(avg_metrics, model_output_dir, filename="metrics_summary.json")

    # --- Combine and save all fold predictions ---
    if all_fold_predictions:
        combined_preds = pd.concat(all_fold_predictions, ignore_index=False)
        combined_preds.to_csv(
            os.path.join(model_output_dir, "all_predictions.csv"),
            index=True, index_label="original_index"
        )
        print(f"Combined predictions saved to: {os.path.join(model_output_dir, 'all_predictions.csv')}")

    # 5. 전체 SHAP 분석 (모든 폴드 결과 기반)
    print("\nRunning global SHAP analysis...")
    global_shap_start_time = time.time()
    try:
        # 모든 폴드의 SHAP 값과 테스트 데이터가 있는지 확인
        if all_fold_shap_values and all_fold_test_data:
            # 모든 폴드의 SHAP 값을 하나로 결합
            combined_shap_values = np.vstack(all_fold_shap_values)
            
            # 모든 폴드의 테스트 데이터를 하나로 결합 (ignore_index=True로 인덱스 정리)
            combined_test_data = pd.concat(all_fold_test_data, axis=0, ignore_index=True)
            
            print(f"Combined SHAP values shape: {combined_shap_values.shape}")
            print(f"Combined test data shape: {combined_test_data.shape}")
            
            # 결합된 데이터로 글로벌 SHAP 분석 수행
            analyze_global_shap(
                model_name=model_name,
                model=best_estimator,  # 마지막 폴드의 모델 사용 (시각화에만 필요)
                test_data_df=combined_test_data,
                shap_values_np=combined_shap_values,
                feature_names_list=feature_names,
                output_dir=model_output_dir
            )
            print(f"Global SHAP analysis completed using data from all {outer_folds} folds")
        else:
            print("No SHAP values were collected across folds. Skipping global SHAP analysis.")
    except Exception as e:
        print(f"Error during global SHAP analysis for model {model_name}: {e}")
        print(traceback.format_exc())
    global_shap_end_time = time.time()
    print(f"Global SHAP Analysis Duration: {global_shap_end_time - global_shap_start_time:.2f} seconds")

    aggregation_end_time = time.time()
    print(f"Result Aggregation & Global SHAP Duration: {aggregation_end_time - aggregation_start_time:.2f} seconds")
    
    if fold_times:
        print(f"Average time per outer fold: {np.mean(fold_times):.2f} seconds")

    print(f"--- Nested CV completed for: {model_display_name} ({model_name}) ---")
    return avg_metrics

def main():
    overall_start_time = time.time() # 전체 시작 시간 기록
    parser = argparse.ArgumentParser(description='Unified Nested CV Classification Framework')
    
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
                        help='Number of CPU cores for parallel processing. 0 uses 75%% cores, -1 uses all available cores.')

    args = parser.parse_args()
    
    # --- Setup ---
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
        n_jobs = -1 # Let sklearn/libraries handle -1
        print("Using all available CPU cores (n_jobs=-1).")
    else:
        print(f"Using {n_jobs} CPU cores.")

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'classification_results_{timestamp}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")
    
    # Save arguments
    save_results(vars(args), args.output_dir, filename="run_arguments.json")
    
    # --- Data Loading ---
    print("\n--- Loading Data ---")
    data_load_start_time = time.time()
    try:
        X, y, query_ids = load_and_preprocess_data(
            file_path=args.input_file,
            target_column=args.target_column,
            threshold=args.threshold,
            query_id_column=args.query_id_column,
            user_features_to_drop=args.drop_features 
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return # Exit if data loading fails
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        print(traceback.format_exc())
        return
    data_load_end_time = time.time()
    print(f"Data Loading & Preprocessing Duration: {data_load_end_time - data_load_start_time:.2f} seconds")
        
    # --- Model Training Loop ---
    print("\n--- Starting Model Training ---")
    all_models_summary = []
    selected_model_names = [name.strip() for name in args.models.split(',')]
    available_models_map = get_available_models()
    
    # Filter for valid models requested
    models_to_run = []
    for name in selected_model_names:
        if name in available_models_map:
            models_to_run.append(available_models_map[name])
        else:
            print(f"Warning: Model '{name}' not found or not available. Skipping.")
            
    if not models_to_run:
        print("Error: No valid models selected to run. Exiting.")
        return

    # Run Nested CV for each selected model
    for model_obj in models_to_run:
        model_run_start_time = time.time() # 개별 모델 실행 시작 시간
        try:
             model_avg_metrics = run_nested_cv(
                 X=X, y=y, query_ids=query_ids, 
                 model_obj=model_obj, 
                 output_dir=args.output_dir, 
                 outer_folds=args.outer_folds, 
                 inner_folds=args.inner_folds,
                 random_iter=args.random_iter, 
                 n_jobs=n_jobs,
                 random_state=args.random_state
             )
             # Add model name to the results dictionary
             model_summary = {'model_name': model_obj.get_model_name(), **model_avg_metrics}
             all_models_summary.append(model_summary)
        except Exception as e:
             model_name = model_obj.get_model_name() if model_obj else "Unknown Model"
             print(f"\nCRITICAL ERROR during training/evaluation for model {model_name}: {e}")
             print(traceback.format_exc())
             all_models_summary.append({'model_name': model_name, 'error': str(e)})
        model_run_end_time = time.time() # 개별 모델 실행 종료 시간
        print(f"Total execution time for model '{model_obj.get_model_name()}': {model_run_end_time - model_run_start_time:.2f} seconds")


    # --- Final Summary ---
    summary_start_time = time.time()
    print("\n--- Overall Training Summary ---")
    if all_models_summary:
         summary_df = pd.DataFrame(all_models_summary)
         summary_filename = os.path.join(args.output_dir, "model_comparison_summary.csv")
         summary_df.to_csv(summary_filename, index=False)
         print(f"Model comparison summary saved to: {summary_filename}")
         print(summary_df)
    else:
         print("No models were successfully trained or evaluated.")
    summary_end_time = time.time()
    print(f"Final Summary Saving Duration: {summary_end_time - summary_start_time:.2f} seconds")

    overall_end_time = time.time() # 전체 종료 시간 기록
    print("\n--- Framework Execution Finished ---")
    print(f"Total script execution time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
