import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
import json
import traceback
import time
import warnings
import logging
from sklearn.pipeline import Pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SHAP 임포트
import shap

# --- 내부 모듈 임포트 (상대 경로 사용) ---
# 현재 파일(regression_nested_cv.py)은 model/regression/ 에 위치

# 1. models 디렉토리에서 임포트 (model/regression/models/)
try:
    from .models.tree_models import RandomForestRegressorModel, LGBMRegressorModel, XGBoostRegressorModel
    from .models.linear_models import LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel
    logger.info("Successfully imported model classes using relative paths.")
except ImportError as e:
    logger.error(f"Error importing models using relative paths: {e}")
    raise ImportError("Could not import model classes. Ensure script is run as a module (-m).") from e

# 2. utils 디렉토리에서 임포트 (model/regression/utils/)
try:
    from .utils.data_loader import load_and_preprocess_data
    from .utils.evaluation import calculate_regression_metrics, save_results, save_predictions
    from .utils.shap_analysis import save_shap_results_regression, analyze_global_shap_regression
    from .utils.visualization import (
        plot_actual_vs_predicted,
        plot_residuals_vs_predicted,
        plot_residual_distribution,
        plot_qq
    )
    logger.info("Successfully imported utils using relative paths.")
except ImportError as e:
    logger.error(f"Error importing utils using relative paths: {e}")
    raise ImportError("Could not import utility modules. Ensure script is run as a module (-m).") from e

# --- 경고 필터링 설정 (필요에 따라 조정) ---
warnings.filterwarnings("ignore", category=FutureWarning)  # General future warnings

def get_available_models():
    """사용 가능한 회귀 모델 객체들을 딕셔너리로 반환합니다."""
    # Uses updated model classes with necessary methods
    return {
        'rf': RandomForestRegressorModel(),
        'lgbm': LGBMRegressorModel(),
        'xgb': XGBoostRegressorModel(),
        'lr': LinearRegressionModel(),
        'ridge': RidgeModel(),
        'lasso': LassoModel(),
        'en': ElasticNetModel()
        # Add new regression model classes here
    }

def run_nested_cv_regression(X, y, query_ids, model_obj, output_dir, outer_folds=5, inner_folds=3,
                 random_iter=50, n_jobs=1, random_state=42, scoring_metric='neg_mean_squared_error',
                 shap_background_samples=100):
    """
    주어진 회귀 모델 객체에 대해 Nested Cross-Validation을 수행.

    Args:
        X (pd.DataFrame): 특성 데이터.
        y (pd.Series): 목표 변수 (연속형).
        query_ids (pd.Series or None): 그룹 ID (GroupKFold 사용 시).
        model_obj (BaseModelRegressor): 학습 및 평가할 모델 객체 (업데이트된 베이스 클래스).
        output_dir (str): 모든 결과 파일이 저장될 최상위 디렉토리.
        outer_folds (int): 외부 교차 검증 폴드 수.
        inner_folds (int): 내부 교차 검증 폴드 수 (하이퍼파라미터 튜닝용).
        random_iter (int): RandomizedSearchCV 반복 횟수.
        n_jobs (int): 병렬 처리에 사용할 CPU 코어 수.
        random_state (int): 재현성을 위한 랜덤 시드.
        scoring_metric (str): RandomizedSearchCV에 사용할 평가지표 ('neg_mean_squared_error', 'r2', etc.).
        shap_background_samples (int): KernelExplainer용 백그라운드 데이터 샘플 수.

    Returns:
        dict: 해당 모델의 모든 외부 폴드에 대한 평균 성능 지표.
    """
    model_name = model_obj.get_model_name() # e.g., 'rf'
    model_display_name = model_obj.get_display_name() # e.g., 'Random Forest'
    model_output_dir = os.path.join(output_dir, model_name) # Model-specific subdirectory
    os.makedirs(model_output_dir, exist_ok=True)

    logger.info(f"\n--- Running Nested CV for Regression: {model_display_name} ({model_name}) ---")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Hyperparameter tuning scoring metric: {scoring_metric}")

    # 결과 저장을 위한 리스트 초기화
    all_fold_metrics = []
    all_fold_predictions_list = [] # Store prediction dataframes for optional aggregation
    feature_names = X.columns.tolist() # Get feature names early

    # 외부 CV 설정 (모델 최종 평가용)
    if query_ids is not None:
        logger.info(f"Using GroupKFold for outer CV with {outer_folds} folds based on query IDs.")
        outer_cv = GroupKFold(n_splits=outer_folds)
        cv_splitter = outer_cv.split(X, y, groups=query_ids)
    else:
        logger.info(f"Using KFold for outer CV with {outer_folds} folds (no query IDs provided).")
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y)

    fold_times = [] # 각 폴드 처리 시간 저장 리스트
    # --- Outer Loop ---
    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        fold_start_time = time.time() # 폴드 시작 시간 기록
        fold_num = fold + 1
        logger.info(f"\n-- Processing Outer Fold {fold_num}/{outer_folds} --")

        # 외부 폴드 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_output_dir = os.path.join(model_output_dir, f"fold_{fold_num}") # Fold-specific subdir
        os.makedirs(fold_output_dir, exist_ok=True)

        logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        logger.info(f"Test set indices range from {test_idx.min()} to {test_idx.max()}")

        # 내부 CV 설정 (하이퍼파라미터 튜닝용)
        inner_cv_groups = query_ids.iloc[train_idx] if query_ids is not None else None
        if inner_cv_groups is not None:
            logger.info(f"Using GroupKFold for inner CV with {inner_folds} folds.")
            inner_cv = GroupKFold(n_splits=inner_folds)
        else:
            logger.info(f"Using KFold for inner CV with {inner_folds} folds.")
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

        # --- 1. 하이퍼파라미터 튜닝 (RandomizedSearchCV) ---
        tuning_start_time = time.time()
        best_estimator = None # Initialize best_estimator
        try:
            logger.info("Starting hyperparameter tuning (RandomizedSearchCV)...")
            param_grid = model_obj.get_hyperparameter_grid()
            
            # 모델 이름 가져오기
            model_name = model_obj.get_model_name()

            # Create the base estimator using the model object's method
            base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs)
            
            # 선형 모델인 경우 n_jobs=1로 설정 (병렬화 오버헤드 방지)
            if model_name in ('lr', 'ridge', 'lasso', 'en'):
                cv_n_jobs = 1
                logger.info(f"Linear model '{model_name}' detected. Setting n_jobs=1 for efficient hyperparameter tuning.")
            else:
                # 트리 기반 모델도 n_jobs를 너무 크게 설정하지 않도록 조정
                cv_n_jobs = min(4, n_jobs)  # 최대 4개 코어만 사용 (오버헤드 감소)
                logger.info(f"Tree-based model '{model_name}'. Setting n_jobs={cv_n_jobs} for hyperparameter tuning.")
            
            # 파라미터 공간 크기 추정 (대략적)
            grid_size = 1
            for key, value in param_grid.items():
                if hasattr(value, '__len__'):  # 리스트, 배열 등
                    grid_size *= len(value)
                else:  # 분포 객체는 크기를 정확히 알 수 없으므로 보수적으로 10으로 가정
                    grid_size *= 10

            # n_iter 조정
            adjusted_n_iter = min(random_iter, grid_size)
            if adjusted_n_iter < random_iter:
                logger.info(f"Parameter space size ({grid_size}) is smaller than requested n_iter ({random_iter}). "
                           f"Adjusting to {adjusted_n_iter}.")

            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=adjusted_n_iter,  # 조정된 값 사용
                cv=inner_cv,
                scoring=scoring_metric,
                random_state=random_state,
                n_jobs=cv_n_jobs,
                verbose=1
            )

            # 모델 학습 시 groups 인자 전달
            if inner_cv_groups is not None:
                logger.info("Fitting with groups parameter for GroupKFold")
                search.fit(X_train, y_train, groups=inner_cv_groups)
            else:
                logger.info("Fitting without groups parameter (using KFold)")
                search.fit(X_train, y_train)

            best_params = search.best_params_
            # Note: best_score_ is the score based on 'scoring_metric' (e.g., negative MSE)
            best_score = search.best_score_
            best_estimator = search.best_estimator_ # The model/pipeline with best params

            logger.info(f"Best Params found: {best_params}")
            logger.info(f"Best Inner CV Score ({scoring_metric}): {best_score:.4f}")

            # 선형 모델 디버깅 출력 추가
            if isinstance(model_obj, (LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel)):
                logger.info(f"\n[DEBUG] Linear model details:")
                logger.info(f"  - Model type: {model_name} ({model_display_name})")
                logger.info(f"  - Best estimator type: {type(best_estimator)}")
                all_params = best_estimator.get_params()
                logger.info(f"  - All params: {all_params}")
                
                # Pipeline 구조 확인 및 스케일러 유무 검증
                if isinstance(best_estimator, Pipeline):
                    logger.info(f"  - Pipeline steps: {[s[0] for s in best_estimator.steps]}")
                    if 'scaler' in best_estimator.named_steps:
                        logger.info(f"  - Scaler found: {type(best_estimator.named_steps['scaler'])}")
                    else:
                        logger.warning(f"  - WARNING: No scaler found in pipeline!")
                
                # 중요 파라미터(특히 alpha) 확인
                alpha_param = next((v for k, v in all_params.items() if 'alpha' in k.lower()), None)
                if alpha_param is not None:
                    logger.info(f"  - Alpha value: {alpha_param}")
                    if alpha_param > 1e4:
                        logger.warning(f"  - WARNING: Alpha value is extremely high!")

            # 최적 파라미터 저장
            save_results(best_params, fold_output_dir, filename="best_params.json")

        except Exception as e:
            logger.error(f"Error during RandomizedSearchCV in fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
            tuning_end_time = time.time()
            logger.error(f"Hyperparameter Tuning (Error) Duration: {tuning_end_time - tuning_start_time:.2f} seconds")
            logger.warning(f"Fold {fold_num}/{outer_folds} will be skipped in final metrics calculation.")
            continue  # Skip to next fold if tuning fails
        tuning_end_time = time.time()
        logger.info(f"Hyperparameter Tuning Duration: {tuning_end_time - tuning_start_time:.2f} seconds")


        if best_estimator is None:
             logger.error(f"Error: best_estimator is None after tuning in fold {fold_num}. Skipping evaluation and SHAP.")
             continue

        # --- 2. 외부 테스트 세트 예측 및 평가 ---
        predict_eval_start_time = time.time()
        try:
            logger.info("Predicting on outer test set...")
            y_pred = best_estimator.predict(X_test)

            logger.info("Calculating performance metrics...")
            # Use the dedicated regression metrics function
            metrics = calculate_regression_metrics(y_test, y_pred)
            metrics['best_inner_cv_score'] = best_score  # Add inner CV score for reference
            metrics['scoring_metric_used'] = scoring_metric  # Record which score was optimized
            
            # 폴드 단위로 비정상 체크 추가
            model_name = model_obj.get_model_name()
            if model_name in ('lr', 'ridge', 'lasso', 'en'):
                if metrics['r2'] < -1000 or metrics['mse'] > 1000000:
                    logger.warning(
                        f"Fold {fold_num}/{outer_folds}: 비정상적 평가 지표 detected "
                        f"(r2={metrics['r2']:.2e}, mse={metrics['mse']:.2e}), "
                        "이 폴드의 추가 처리는 건너뜁니다."
                    )
                    # 비정상 지표임을 표시하는 필드 추가
                    metrics['abnormal_performance'] = True
                    
                    # 폴드 성능 지표는 저장 (디버깅 및 분석용)
                    all_fold_metrics.append(metrics)
                    save_results(metrics, fold_output_dir, filename="metrics.json")
                    
                    # 현재 폴드의 추가 처리(진단 플롯, SHAP 등)는 건너뛰고 다음 폴드로 진행
                    logger.info(f"-- Outer Fold {fold_num} partially processed (abnormal metrics). --")
                    continue
            
            # 정상 메트릭인 경우 기존 처리 계속
            all_fold_metrics.append(metrics)

            # 폴드 성능 지표 저장
            save_results(metrics, fold_output_dir, filename="metrics.json")

            # 폴드 예측 결과 저장 (인덱스 포함)
            save_predictions(y_test, y_pred, fold_output_dir,
                             filename="predictions.csv", index=X_test.index)
            # Store DF for potential later aggregation
            preds_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
            all_fold_predictions_list.append(preds_df)
            
            # --- 2.a Diagnostic Plots 추가 ---
            try:
                logger.info("Generating diagnostic plots...")
                y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
                y_pred_arr = np.array(y_pred)
                plot_actual_vs_predicted(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                plot_residuals_vs_predicted(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                plot_residual_distribution(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                residuals = y_test_arr - y_pred_arr
                plot_qq(residuals, fold_output_dir, fold_num)
                logger.info("Diagnostic plots saved.")
            except Exception as e:
                logger.error(f"Error generating diagnostic plots for fold {fold_num}: {e}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error during prediction or evaluation in fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
            predict_eval_end_time = time.time()
            logger.info(f"Prediction & Evaluation (Error) Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")
            # Let's skip SHAP if prediction fails
            continue
        predict_eval_end_time = time.time()
        logger.info(f"Prediction & Evaluation Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")

        # --- 3. SHAP 값 계산 및 저장 ---
        shap_start_time = time.time()
        try:
            logger.info("Calculating and saving SHAP values...")

            # Prepare background data
            n_samples = min(shap_background_samples, X_train.shape[0])
            logger.info(f"Sampling {n_samples} instances from X_train for SHAP background data.")
            background_data = shap.sample(X_train, n_samples, random_state=random_state)
            
            # 배경 데이터가 X_test와 동일한 컬럼을 가지도록 확인
            if isinstance(background_data, np.ndarray):
                background_data = pd.DataFrame(background_data, columns=X_test.columns)

            # SHAP 계산 시 배경 데이터 전달
            shap_df, base_vals = model_obj.calculate_shap_values(
                trained_model=best_estimator,
                X_explain=X_test,
                X_train_sample=background_data  # 배경 데이터 명시적 전달
            )

            # Check if the DataFrame part of the tuple is not None
            if shap_df is not None:
                 # Pass the DataFrame to shap_values and base_vals to expected_value
                 save_shap_results_regression(
                     shap_values=shap_df,       # Pass the DataFrame
                     X_explain=X_test,
                     feature_names=feature_names,
                     output_dir=fold_output_dir,
                     fold_num=fold_num,
                     expected_value=base_vals  # Pass the base values
                 )
            else:
                 logger.warning("SHAP values calculation failed or returned None.")

        except Exception as e:
            logger.error(f"Error during SHAP calculation/saving in fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
        shap_end_time = time.time()
        logger.info(f"SHAP Calculation & Saving Duration: {shap_end_time - shap_start_time:.2f} seconds")


        fold_end_time = time.time() # 폴드 종료 시간 기록
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        logger.info(f"-- Outer Fold {fold_num} finished. Duration: {fold_duration:.2f} seconds --")

    # --- After Outer Loop ---
    aggregation_start_time = time.time()

    # --- 4. 전체 폴드 결과 요약 ---
    if not all_fold_metrics:
        logger.error(f"\nError: No metrics were collected for model {model_name}. Check logs for errors in folds.")
        return {'error': 'No metrics collected'}

    logger.info(f"\n--- Aggregating results for: {model_display_name} ({model_name}) ---")
    # Calculate mean/std, handling potential None values if a fold failed evaluation
    avg_metrics = {}
    metric_keys = list(all_fold_metrics[0].keys()) if all_fold_metrics else []

    for key in metric_keys:
        # 문자열 필드는 집계에서 제외
        if key == 'scoring_metric_used':
            # 문자열 필드는 첫 번째 값 그대로 사용
            valid_values = [m[key] for m in all_fold_metrics if m and key in m and m[key] is not None]
            if valid_values:
                avg_metrics[key] = valid_values[0]  # 첫 번째 값 사용
            continue
        
        valid_values = [m[key] for m in all_fold_metrics if m and key in m and m[key] is not None]
        if valid_values:
            try:
                avg_metrics[f"{key}_mean"] = np.mean(valid_values)
                avg_metrics[f"{key}_std"] = np.std(valid_values)
            except TypeError: # Handle non-numeric types if they sneak in
                logger.warning(f"Warning: Could not compute mean/std for metric '{key}'. Values: {valid_values}")
        else:
            avg_metrics[f"{key}_mean"] = None
            avg_metrics[f"{key}_std"] = None

    logger.info("Average Metrics across folds:")
    # Pretty print the metrics dictionary
    logger.info(json.dumps(avg_metrics, indent=4))

    # 선형 회귀 모델에 대한 특별한 검증 추가
    if model_name in ('lr', 'ridge', 'lasso', 'en'):
        # R2, MSE 등 주요 메트릭 확인
        r2_mean = avg_metrics.get('r2_mean', -float('inf'))
        mse_mean = avg_metrics.get('mse_mean', float('inf'))
        
        # 극단적인 값인지 확인 (임계값은 데이터 스케일에 맞게 조정)
        if r2_mean < -1000 or mse_mean > 1000000:
            logger.warning(f"Linear model '{model_name}' has extremely poor metrics: R2={r2_mean:.2e}, MSE={mse_mean:.2e}")
            logger.warning("Skipping further processing for this model due to poor performance")
            
            # 현재까지 계산된 메트릭만 반환하고 SHAP 등 추가 계산 건너뛰기
            avg_metrics['abnormal_performance'] = True
            avg_metrics['processing_skipped'] = True
            return avg_metrics
    
    # 정상적인 경우 기존 처리 계속 진행
    
    # 평균 성능 지표 저장 (모델 최상위 디렉토리)
    save_results(avg_metrics, model_output_dir, filename="metrics_summary.json")

    # (Optional) 모든 폴드의 예측 결과 합치기
    if all_fold_predictions_list:
         try:
             all_preds_df = pd.concat(all_fold_predictions_list) # Index should align if split correctly
             all_preds_df.sort_index(inplace=True)
             preds_save_path = os.path.join(model_output_dir, "all_folds_predictions.csv")
             all_preds_df.to_csv(preds_save_path, index=True, index_label='original_index')
             logger.info(f"Combined predictions saved to: {preds_save_path}")
         except Exception as e:
             logger.error(f"Error saving combined predictions: {e}")


    # --- 5. 전체 SHAP 분석 (폴드 결과 기반) ---
    logger.info("\nRunning global SHAP analysis...")
    global_shap_start_time = time.time()
    try:
        # This function reads from the fold directories within model_output_dir
        analyze_global_shap_regression(model_results_dir=model_output_dir)
    except Exception as e:
        logger.error(f"Error during global SHAP analysis for model {model_name}: {e}")
        logger.error(traceback.format_exc())
    global_shap_end_time = time.time()
    logger.info(f"Global SHAP Analysis Duration: {global_shap_end_time - global_shap_start_time:.2f} seconds")


    aggregation_end_time = time.time()
    logger.info(f"Result Aggregation & Global SHAP Duration: {aggregation_end_time - aggregation_start_time:.2f} seconds")

    if fold_times:
        logger.info(f"Average time per outer fold: {np.mean(fold_times):.2f} seconds")

    logger.info(f"--- Nested CV completed for: {model_display_name} ({model_name}) ---")
    return avg_metrics


def main():
    overall_start_time = time.time() # 전체 시작 시간 기록
    parser = argparse.ArgumentParser(description='Nested CV Regression Framework with SHAP')

    # --- Input/Output Arguments ---
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input CSV data file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save all results. If None, creates a timestamped directory.')
    parser.add_argument('--target_column', type=str, default='DockQ',
                        help='Name of the continuous target variable column.')
    # NO --threshold argument needed for regression
    parser.add_argument('--query_id_column', type=str, default='query',
                        help='Name of the column containing group IDs for GroupKFold. Set to "" or omit to disable.')
    parser.add_argument('--drop_features', nargs='*', default=None,
                        help='List of additional feature columns to drop from X.')

    # --- Model Selection ---
    available_model_names = list(get_available_models().keys())
    parser.add_argument('--models', type=str, default=','.join(available_model_names),
                        help=f'Comma-separated list of models to train. Available: {", ".join(available_model_names)}')

    # --- CV, Tuning, and Scoring Arguments ---
    parser.add_argument('--outer_folds', type=int, default=5, help='Number of outer CV folds.')
    parser.add_argument('--inner_folds', type=int, default=3, help='Number of inner CV folds.')
    parser.add_argument('--random_iter', type=int, default=50,
                        help='Number of iterations for RandomizedSearchCV.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--scoring', type=str, default='neg_mean_squared_error',
                        help='Scoring metric for hyperparameter tuning (e.g., neg_mean_squared_error, r2, neg_mean_absolute_error).')

    # --- Computation Arguments ---
    parser.add_argument('--n_jobs', type=int, default=0,
                        help='Number of CPU cores for parallel processing. 0 uses 75%% cores, -1 uses all available cores.')
    parser.add_argument('--shap_background_samples', type=int, default=100,
                        help='Number of samples for background data in SHAP KernelExplainer.')


    args = parser.parse_args()

    # --- Setup ---
    # Handle n_jobs
    n_jobs = args.n_jobs
    if n_jobs == 0:
        import multiprocessing
        try:
            n_cores = multiprocessing.cpu_count()
            n_jobs = max(1, int(n_cores * 0.75))
            logger.info(f"Using {n_jobs} CPU cores (75% of available {n_cores}).")
        except NotImplementedError:
            n_jobs = 1
            logger.info("Could not detect number of CPU cores. Using 1 core.")
    elif n_jobs < 0:
        n_jobs = -1  # 단순화: 음수는 모두 -1로 처리
        logger.info("Using all available CPU cores (n_jobs=-1).")
    else:
        logger.info(f"Using {n_jobs} CPU cores.")

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'regression_nested_cv_results_{timestamp}'
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output_dir}")

    # Save arguments
    save_results(vars(args), args.output_dir, filename="run_arguments.json")

    # Handle empty string for query_id_column
    query_id_col = args.query_id_column if args.query_id_column else None


    # --- Data Loading ---
    logger.info("\n--- Loading Data ---")
    data_load_start_time = time.time()
    try:
        X, y, query_ids, _ = load_and_preprocess_data(
            file_path=args.input_file,
            target_column=args.target_column,
            query_id_column=query_id_col,
            user_features_to_drop=args.drop_features
        )
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Error loading data: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        logger.error(traceback.format_exc())
        return
    data_load_end_time = time.time()
    logger.info(f"Data Loading & Preprocessing Duration: {data_load_end_time - data_load_start_time:.2f} seconds")


    # --- Model Training Loop ---
    logger.info("\n--- Starting Model Training ---")
    all_models_summary = []
    selected_model_names = [name.strip() for name in args.models.split(',')]
    available_models_map = get_available_models()

    # Filter for valid models requested
    models_to_run = []
    for name in selected_model_names:
        if name in available_models_map:
            models_to_run.append(available_models_map[name])
        else:
            logger.warning(f"Warning: Model '{name}' not found or not available. Skipping.")

    if not models_to_run:
        logger.error("Error: No valid models selected to run. Exiting.")
        return

    # Run Nested CV for each selected model
    for model_obj in models_to_run:
        model_run_start_time = time.time()
        try:
            logger.info(f"\n=== Processing model: {model_obj.get_display_name()} ({model_obj.get_model_name()}) ===")
            
            model_avg_metrics = run_nested_cv_regression(
                X=X,
                y=y,
                query_ids=query_ids,
                model_obj=model_obj,
                output_dir=args.output_dir,
                outer_folds=args.outer_folds,
                inner_folds=args.inner_folds,
                random_iter=args.random_iter,
                n_jobs=n_jobs,
                random_state=args.random_state,
                scoring_metric=args.scoring,
                shap_background_samples=args.shap_background_samples
            )
            
            # 비정상적인 성능의 모델 결과 확인
            if model_avg_metrics.get('abnormal_performance', False):
                logger.warning(f"Model {model_obj.get_model_name()} showed abnormal performance and was partially processed.")
                # 비정상 모델도 결과는 저장
                model_summary = {'model_name': model_obj.get_model_name(), **model_avg_metrics}
                all_models_summary.append(model_summary)
                continue  # 다음 모델로 넘어감
            
            # 정상 모델 처리
            # Add model name to the results dictionary
            model_summary = {'model_name': model_obj.get_model_name(), **model_avg_metrics}
            all_models_summary.append(model_summary)
            
        except Exception as e:
            model_name = model_obj.get_model_name() if model_obj else "Unknown Model"
            logger.error(f"\nCRITICAL ERROR during training/evaluation for model {model_name}: {e}")
            logger.error(traceback.format_exc())
            all_models_summary.append({'model_name': model_name, 'error': str(e)})
            continue  # 예외가 발생해도 다음 모델로 계속 진행
            
        model_run_end_time = time.time()
        logger.info(f"Total execution time for model '{model_obj.get_model_name()}': {model_run_end_time - model_run_start_time:.2f} seconds")


    # --- Final Summary ---
    summary_start_time = time.time()
    logger.info("\n--- Overall Training Summary ---")
    if all_models_summary:
         # Convert list of dicts to DataFrame, handling potential errors
         summary_df = pd.DataFrame([m for m in all_models_summary if 'error' not in m])
         error_df = pd.DataFrame([m for m in all_models_summary if 'error' in m])

         if not summary_df.empty:
             summary_filename = os.path.join(args.output_dir, "model_comparison_summary.csv")
             summary_df.to_csv(summary_filename, index=False)
             logger.info(f"Model comparison summary saved to: {summary_filename}")
             logger.info("Successful Model Summary:")
             logger.info(summary_df)
         else:
              logger.info("No models completed successfully.")

         if not error_df.empty:
             error_filename = os.path.join(args.output_dir, "model_errors_summary.csv")
             error_df.to_csv(error_filename, index=False)
             logger.info(f"\nModel errors summary saved to: {error_filename}")
             logger.info("Models with Errors:")
             logger.info(error_df)

    else:
         logger.info("No models were run or evaluated.")
    summary_end_time = time.time()
    logger.info(f"Final Summary Saving Duration: {summary_end_time - summary_start_time:.2f} seconds")


    overall_end_time = time.time() # 전체 종료 시간 기록
    logger.info("\n--- Framework Execution Finished ---")
    logger.info(f"Total script execution time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
