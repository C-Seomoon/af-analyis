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
from collections import Counter

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

def run_nested_cv_regression_evaluation_only(X, y, query_ids, model_obj, output_dir, outer_folds=5, inner_folds=3,
                 random_iter=50, n_jobs=1, random_state=42, scoring_metric='neg_mean_squared_error'):
    """
    Nested Cross-Validation을 통한 성능 평가만 수행 (SHAP 계산 제외).
    
    Returns:
        dict: 평균 성능 지표
    """
    model_name = model_obj.get_model_name()
    model_display_name = model_obj.get_display_name()
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    logger.info(f"\n--- Running Nested CV Evaluation for: {model_display_name} ({model_name}) ---")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Hyperparameter tuning scoring metric: {scoring_metric}")

    # 결과 저장을 위한 리스트 초기화
    all_fold_metrics = []
    all_fold_predictions_list = []
    feature_names = X.columns.tolist()

    # 외부 CV 설정
    if query_ids is not None:
        logger.info(f"Using GroupKFold for outer CV with {outer_folds} folds based on query IDs.")
        outer_cv = GroupKFold(n_splits=outer_folds)
        cv_splitter = outer_cv.split(X, y, groups=query_ids)
    else:
        logger.info(f"Using KFold for outer CV with {outer_folds} folds (no query IDs provided).")
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        cv_splitter = outer_cv.split(X, y)

    fold_times = []
    
    # --- Outer Loop ---
    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        fold_start_time = time.time()
        fold_num = fold + 1
        logger.info(f"\n-- Processing Outer Fold {fold_num}/{outer_folds} --")

        # 외부 폴드 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_output_dir = os.path.join(model_output_dir, f"fold_{fold_num}")
        os.makedirs(fold_output_dir, exist_ok=True)

        logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

        # 내부 CV 설정
        inner_cv_groups = query_ids.iloc[train_idx] if query_ids is not None else None
        if inner_cv_groups is not None:
            logger.info(f"Using GroupKFold for inner CV with {inner_folds} folds.")
            inner_cv = GroupKFold(n_splits=inner_folds)
        else:
            logger.info(f"Using KFold for inner CV with {inner_folds} folds.")
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

        # --- 하이퍼파라미터 튜닝 ---
        tuning_start_time = time.time()
        best_estimator = None
        try:
            logger.info("Starting hyperparameter tuning (RandomizedSearchCV)...")
            param_grid = model_obj.get_hyperparameter_grid()
            base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs)
            
            # 모델별 n_jobs 조정
            if model_name in ('lr', 'ridge', 'lasso', 'en'):
                cv_n_jobs = 1
            else:
                cv_n_jobs = min(4, n_jobs)
            
            # 파라미터 공간 크기 추정
            grid_size = 1
            for key, value in param_grid.items():
                if hasattr(value, '__len__'):
                    grid_size *= len(value)
                else:
                    grid_size *= 10
            
            adjusted_n_iter = min(random_iter, grid_size)

            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=adjusted_n_iter,
                cv=inner_cv,
                scoring=scoring_metric,
                random_state=random_state,
                n_jobs=cv_n_jobs,
                verbose=1
            )

            # 모델 학습
            if inner_cv_groups is not None:
                search.fit(X_train, y_train, groups=inner_cv_groups)
            else:
                search.fit(X_train, y_train)

            best_params = search.best_params_
            best_score = search.best_score_
            best_estimator = search.best_estimator_

            logger.info(f"Best Params found: {best_params}")
            logger.info(f"Best Inner CV Score ({scoring_metric}): {best_score:.4f}")

            # 최적 파라미터 저장
            save_results(best_params, fold_output_dir, filename="best_params.json")

        except Exception as e:
            logger.error(f"Error during RandomizedSearchCV in fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
            continue

        tuning_end_time = time.time()
        logger.info(f"Hyperparameter Tuning Duration: {tuning_end_time - tuning_start_time:.2f} seconds")

        if best_estimator is None:
            logger.error(f"Error: best_estimator is None after tuning in fold {fold_num}. Skipping evaluation.")
            continue

        # --- 예측 및 평가 ---
        predict_eval_start_time = time.time()
        try:
            logger.info("Predicting on outer test set...")
            y_pred = best_estimator.predict(X_test)

            logger.info("Calculating performance metrics...")
            metrics = calculate_regression_metrics(y_test, y_pred)
            metrics['best_inner_cv_score'] = best_score
            metrics['scoring_metric_used'] = scoring_metric
            
            # 비정상 성능 체크
            if model_name in ('lr', 'ridge', 'lasso', 'en'):
                if metrics['r2'] < -1000 or metrics['mse'] > 1000000:
                    logger.warning(f"Fold {fold_num}: Abnormal performance detected. Skipping this fold.")
                    metrics['abnormal_performance'] = True
                    all_fold_metrics.append(metrics)
                    save_results(metrics, fold_output_dir, filename="metrics.json")
                    continue
            
            all_fold_metrics.append(metrics)

            # 결과 저장
            save_results(metrics, fold_output_dir, filename="metrics.json")
            save_predictions(y_test, y_pred, fold_output_dir, filename="predictions.csv", index=X_test.index)
            
            preds_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
            all_fold_predictions_list.append(preds_df)
            
            # 진단 플롯 생성
            try:
                logger.info("Generating diagnostic plots...")
                y_test_arr = y_test.values if isinstance(y_test, pd.Series) else y_test
                y_pred_arr = np.array(y_pred)
                plot_actual_vs_predicted(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                plot_residuals_vs_predicted(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                plot_residual_distribution(y_test_arr, y_pred_arr, fold_output_dir, fold_num)
                residuals = y_test_arr - y_pred_arr
                plot_qq(residuals, fold_output_dir, fold_num)
            except Exception as e:
                logger.error(f"Error generating diagnostic plots for fold {fold_num}: {e}")

        except Exception as e:
            logger.error(f"Error during prediction or evaluation in fold {fold_num}: {e}")
            logger.error(traceback.format_exc())
            continue

        predict_eval_end_time = time.time()
        logger.info(f"Prediction & Evaluation Duration: {predict_eval_end_time - predict_eval_start_time:.2f} seconds")

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)
        logger.info(f"-- Outer Fold {fold_num} finished. Duration: {fold_duration:.2f} seconds --")

    # --- 결과 집계 ---
    if not all_fold_metrics:
        logger.error(f"Error: No metrics were collected for model {model_name}.")
        return {'error': 'No metrics collected'}

    logger.info(f"\n--- Aggregating results for: {model_display_name} ({model_name}) ---")
    
    avg_metrics = {}
    metric_keys = list(all_fold_metrics[0].keys()) if all_fold_metrics else []

    for key in metric_keys:
        if key == 'scoring_metric_used':
            valid_values = [m[key] for m in all_fold_metrics if m and key in m and m[key] is not None]
            if valid_values:
                avg_metrics[key] = valid_values[0]
            continue
        
        valid_values = [m[key] for m in all_fold_metrics if m and key in m and m[key] is not None]
        if valid_values:
            try:
                avg_metrics[f"{key}_mean"] = np.mean(valid_values)
                avg_metrics[f"{key}_std"] = np.std(valid_values)
            except TypeError:
                logger.warning(f"Warning: Could not compute mean/std for metric '{key}'.")
        else:
            avg_metrics[f"{key}_mean"] = None
            avg_metrics[f"{key}_std"] = None

    logger.info("Average Metrics across folds:")
    logger.info(json.dumps(avg_metrics, indent=4))

    # 선형 모델 성능 체크
    if model_name in ('lr', 'ridge', 'lasso', 'en'):
        r2_mean = avg_metrics.get('r2_mean', -float('inf'))
        mse_mean = avg_metrics.get('mse_mean', float('inf'))
        
        if r2_mean < -1000 or mse_mean > 1000000:
            logger.warning(f"Linear model '{model_name}' has extremely poor metrics.")
            avg_metrics['abnormal_performance'] = True
            avg_metrics['processing_skipped'] = True
            return avg_metrics

    # 평균 성능 지표 저장
    save_results(avg_metrics, model_output_dir, filename="metrics_summary.json")

    # 모든 폴드 예측 결과 합치기
    if all_fold_predictions_list:
        try:
            all_preds_df = pd.concat(all_fold_predictions_list)
            all_preds_df.sort_index(inplace=True)
            preds_save_path = os.path.join(model_output_dir, "all_folds_predictions.csv")
            all_preds_df.to_csv(preds_save_path, index=True, index_label='original_index')
            logger.info(f"Combined predictions saved to: {preds_save_path}")
        except Exception as e:
            logger.error(f"Error saving combined predictions: {e}")

    if fold_times:
        logger.info(f"Average time per outer fold: {np.mean(fold_times):.2f} seconds")

    logger.info(f"--- Nested CV completed for: {model_display_name} ({model_name}) ---")
    return avg_metrics

def get_best_hyperparameters_separate_cv(X, y, query_ids, model_obj, inner_folds=3, 
                                        random_iter=50, n_jobs=1, random_state=42,
                                        scoring_metric='neg_mean_squared_error'):
    """
    전체 학습 데이터에서 별도의 CV로 최적 하이퍼파라미터를 찾습니다.
    """
    logger.info("Finding optimal hyperparameters using separate cross-validation...")
    
    # 내부 CV 설정
    if query_ids is not None:
        inner_cv = GroupKFold(n_splits=inner_folds)
    else:
        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    
    # 하이퍼파라미터 탐색
    param_grid = model_obj.get_hyperparameter_grid()
    base_estimator = model_obj.create_model(params=None, n_jobs=n_jobs)
    
    # 모델별 n_jobs 조정
    model_name = model_obj.get_model_name()
    if model_name in ('lr', 'ridge', 'lasso', 'en'):
        cv_n_jobs = 1
    else:
        cv_n_jobs = min(4, n_jobs)
    
    # 파라미터 공간 크기 추정
    grid_size = 1
    for key, value in param_grid.items():
        if hasattr(value, '__len__'):
            grid_size *= len(value)
        else:
            grid_size *= 10
    
    adjusted_n_iter = min(random_iter, grid_size)
    
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=adjusted_n_iter,
        cv=inner_cv,
        scoring=scoring_metric,
        random_state=random_state,
        n_jobs=cv_n_jobs,
        verbose=1
    )
    
    # 전체 데이터로 튜닝
    if query_ids is not None:
        search.fit(X, y, groups=query_ids)
    else:
        search.fit(X, y)
    
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_params_, search.best_score_

def train_final_regression_model_with_shap(X, y, model_obj, best_params, output_dir, 
                                         n_jobs=1, shap_background_samples=100, random_state=42):
    """
    최종 회귀 모델을 전체 데이터로 학습하고 SHAP 값을 계산합니다.
    """
    model_name = model_obj.get_model_name()
    logger.info(f"\n--- Training Final Regression Model: {model_obj.get_display_name()} ---")
    logger.info(f"Using parameters: {best_params}")
    
    # 최종 모델 생성 및 학습
    final_model = model_obj.create_model(params=best_params, n_jobs=n_jobs)
    
    logger.info("Training final model on entire dataset...")
    start_time = time.time()
    final_model.fit(X, y)
    training_time = time.time() - start_time
    logger.info(f"Final model training completed in {training_time:.2f} seconds")
    
    # 전체 데이터에 대한 예측 및 메트릭 계산
    logger.info("Evaluating final model on training data...")
    y_pred_final = final_model.predict(X)
    final_metrics = calculate_regression_metrics(y, y_pred_final)
    
    # 최종 모델 메트릭 저장
    save_results(final_metrics, output_dir, filename="final_model_metrics.json")
    logger.info(f"Final model R²: {final_metrics['r2']:.4f}, MSE: {final_metrics['mse']:.4f}")
    
    # 최종 예측 결과 저장
    save_predictions(y, y_pred_final, output_dir, filename="final_model_predictions.csv", index=X.index)
    
    # 진단 플롯 생성
    try:
        logger.info("Generating final model diagnostic plots...")
        y_arr = y.values if isinstance(y, pd.Series) else y
        y_pred_arr = np.array(y_pred_final)
        plot_actual_vs_predicted(y_arr, y_pred_arr, output_dir, fold_num="final")
        plot_residuals_vs_predicted(y_arr, y_pred_arr, output_dir, fold_num="final")
        plot_residual_distribution(y_arr, y_pred_arr, output_dir, fold_num="final")
        residuals = y_arr - y_pred_arr
        plot_qq(residuals, output_dir, fold_num="final")
        logger.info("Final model diagnostic plots saved.")
    except Exception as e:
        logger.error(f"Error generating final model diagnostic plots: {e}")
    
    # SHAP 값 계산
    logger.info("Calculating SHAP values for final model...")
    shap_start_time = time.time()
    try:
        # 배경 데이터 준비
        n_samples = min(shap_background_samples, X.shape[0])
        logger.info(f"Sampling {n_samples} instances from X for SHAP background data.")
        background_data = shap.sample(X, n_samples, random_state=random_state)
        
        # 배경 데이터가 X와 동일한 컬럼을 가지도록 확인
        if isinstance(background_data, np.ndarray):
            background_data = pd.DataFrame(background_data, columns=X.columns)

        # SHAP 계산
        shap_df, base_vals = model_obj.calculate_shap_values(
            trained_model=final_model,
            X_explain=X,
            X_train_sample=background_data
        )

        if shap_df is not None:
            feature_names = X.columns.tolist()
            
            # SHAP 결과 저장 (fold_num 없이)
            save_shap_results_regression(
                shap_values=shap_df,
                X_explain=X,
                feature_names=feature_names,
                output_dir=output_dir,
                fold_num=None,  # 최종 모델이므로 fold_num 없음
                expected_value=base_vals
            )
            
            # 글로벌 SHAP 분석
            analyze_global_shap_regression(model_results_dir=output_dir)
            logger.info("SHAP analysis completed successfully")
        else:
            logger.warning("SHAP values calculation failed or returned None")
            
    except Exception as e:
        logger.error(f"Error during SHAP calculation: {e}")
        logger.error(traceback.format_exc())
    
    shap_end_time = time.time()
    logger.info(f"SHAP calculation completed in {shap_end_time - shap_start_time:.2f} seconds")
    
    return final_model

def compare_performances(nested_cv_metrics, test_metrics):
    """
    Nested CV와 Test set 성능을 비교하고 해석합니다.
    """
    comparison = {}
    
    # 주요 메트릭들 비교
    for metric in ['r2', 'mse', 'mae']:
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
            
            logger.info(f"{metric.upper()} Comparison:")
            logger.info(f"  Nested CV: {cv_mean:.4f} ± {cv_std:.4f} (95% CI: [{cv_lower:.4f}, {cv_upper:.4f}])")
            logger.info(f"  Test Set:  {test_val:.4f}")
            logger.info(f"  {interpretation}")
    
    return comparison

def comprehensive_model_evaluation(X_train, y_train, X_test, y_test, query_ids_train, 
                                 model_obj, args, output_dir):
    """포괄적 모델 평가 함수"""
    model_name = model_obj.get_model_name()
    model_display_name = model_obj.get_display_name()
    
    # n_jobs 처리 - args에서 올바른 n_jobs 값 사용
    n_jobs = getattr(args, 'n_jobs', 1)
    if n_jobs == 0:
        import multiprocessing
        try:
            n_cores = multiprocessing.cpu_count()
            n_jobs = max(1, int(n_cores * 0.75))
        except NotImplementedError:
            n_jobs = 1
    
    results = {}
    
    # Step 1: Nested CV Performance Estimation
    logger.info("=== Step 1: Nested CV Performance Estimation ===")
    nested_cv_results = run_nested_cv_regression_evaluation_only(
        X=X_train, y=y_train, query_ids=query_ids_train,
        model_obj=model_obj,
        output_dir=output_dir,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        random_iter=args.random_iter,
        n_jobs=n_jobs,  # 올바른 n_jobs 값 전달
        random_state=args.random_state,
        scoring_metric=args.scoring
    )
    
    results['nested_cv'] = nested_cv_results
    
    # 비정상 성능 체크
    if nested_cv_results.get('abnormal_performance', False):
        logger.warning(f"Model {model_name} showed abnormal performance. Skipping further processing.")
        return results, None
    
    # Step 2: Hyperparameter Tuning
    logger.info("=== Step 2: Hyperparameter Tuning ===")
    best_params, best_cv_score = get_best_hyperparameters_separate_cv(
        X=X_train, y=y_train, query_ids=query_ids_train,
        model_obj=model_obj,
        inner_folds=args.inner_folds,
        random_iter=args.random_iter,
        n_jobs=n_jobs,  # 올바른 n_jobs 값 전달
        random_state=args.random_state,
        scoring_metric=args.scoring
    )
    
    results['best_params'] = best_params
    results['hyperparameter_tuning_cv_score'] = best_cv_score
    
    # Step 3: Final Model Training
    logger.info("=== Step 3: Final Model Training ===")
    model_output_dir = os.path.join(output_dir, model_name)
    final_model = train_final_regression_model_with_shap(
        X=X_train, y=y_train,
        model_obj=model_obj,
        best_params=best_params,
        output_dir=model_output_dir,
        n_jobs=n_jobs,
        shap_background_samples=args.shap_background_samples,
        random_state=args.random_state
    )
    
    # Step 4: Test Set Evaluation
    if X_test is not None and y_test is not None:
        logger.info("=== Step 4: Test Set Evaluation ===")
        y_test_pred = final_model.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred)
        results['test_performance'] = test_metrics
        
        # Test set 예측 결과 저장
        save_predictions(y_test, y_test_pred, model_output_dir, 
                        filename="test_predictions.csv", index=X_test.index)
        
        # Step 5: Performance Comparison
        logger.info("=== Step 5: Performance Comparison ===")
        comparison = compare_performances(nested_cv_results, test_metrics)
        results['performance_comparison'] = comparison
        
        # 비교 결과 저장
        save_results(comparison, model_output_dir, filename="performance_comparison.json")
    
    return results, final_model

def main():
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(description='Comprehensive Regression Framework with Nested CV')

    # --- Input/Output Arguments ---
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input CSV data file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save all results. If None, creates a timestamped directory.')
    parser.add_argument('--target_column', type=str, default='DockQ',
                        help='Name of the continuous target variable column.')
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

    # --- CV, Tuning, and Scoring Arguments ---
    parser.add_argument('--outer_folds', type=int, default=5, help='Number of outer CV folds.')
    parser.add_argument('--inner_folds', type=int, default=3, help='Number of inner CV folds.')
    parser.add_argument('--random_iter', type=int, default=50,
                        help='Number of iterations for RandomizedSearchCV.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--scoring', type=str, default='neg_mean_squared_error',
                        help='Scoring metric for hyperparameter tuning.')

    # --- Optuna Arguments ---
    parser.add_argument('--use_optuna_final', action='store_true',
                       help='Use Optuna for final hyperparameter tuning on full train data (Strategy A)')
    parser.add_argument('--optuna_trials', type=int, default=200,
                       help='Number of Optuna trials for final tuning')
    parser.add_argument('--use_nested_cv_consensus', action='store_true',
                       help='Use consensus of nested CV best params instead of retuning (Strategy B)')

    # --- Computation Arguments ---
    parser.add_argument('--n_jobs', type=int, default=0,
                        help='Number of CPU cores for parallel processing.')
    parser.add_argument('--shap_background_samples', type=int, default=100,
                        help='Number of samples for background data in SHAP KernelExplainer.')

    args = parser.parse_args()

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
        n_jobs = -1
        logger.info("Using all available CPU cores (n_jobs=-1).")
    else:
        logger.info(f"Using {n_jobs} CPU cores.")

    # 중요: args.n_jobs를 업데이트해야 함
    args.n_jobs = n_jobs

    # Optuna 사용 가능성 체크
    if args.use_optuna_final and not OPTUNA_AVAILABLE:
        logger.error("Optuna is not installed but --use_optuna_final was specified. "
                    "Please install optuna: pip install optuna")
        return

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'comprehensive_regression_results_{timestamp}'
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output_dir}")

    # Save arguments
    save_results(vars(args), args.output_dir, filename="run_arguments.json")

    # Handle query_id_column
    query_id_col = args.query_id_column if args.query_id_column else None

    # --- Data Loading ---
    logger.info("\n--- Loading Data ---")
    data_load_start_time = time.time()
    try:
        X_train, y_train, query_ids_train, _ = load_and_preprocess_data(
            file_path=args.input_file,
            target_column=args.target_column,
            query_id_column=query_id_col,
            user_features_to_drop=args.drop_features
        )
        
        # 별도 테스트 세트 로딩 (선택적)
        X_test, y_test, query_ids_test = None, None, None
        if args.test_file:
            logger.info(f"Loading separate test set from: {args.test_file}")
            X_test, y_test, query_ids_test, _ = load_and_preprocess_data(
                file_path=args.test_file,
                target_column=args.target_column,
                query_id_column=query_id_col,
                user_features_to_drop=args.drop_features
            )
            logger.info(f"Test set loaded: {X_test.shape}")
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"Error loading data: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        logger.error(traceback.format_exc())
        return
    
    data_load_end_time = time.time()
    logger.info(f"Data Loading Duration: {data_load_end_time - data_load_start_time:.2f} seconds")

    # --- Model Training Loop ---
    logger.info("\n--- Starting Comprehensive Model Evaluation ---")
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
            logger.warning(f"Model '{name}' not found. Skipping.")

    if not models_to_run:
        logger.error("No valid models selected. Exiting.")
        return

    # Run comprehensive evaluation for each model
    for model_obj in models_to_run:
        model_run_start_time = time.time()
        model_name = model_obj.get_model_name()
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_obj.get_display_name()} ({model_name})")
            logger.info(f"{'='*60}")
            
            # 포괄적 모델 평가
            results, final_model = comprehensive_model_evaluation(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                query_ids_train=query_ids_train,
                model_obj=model_obj,
                args=args,
                output_dir=args.output_dir
            )
            
            # 비정상 성능 체크
            if results.get('nested_cv', {}).get('abnormal_performance', False):
                logger.warning(f"Model {model_name} showed abnormal performance.")
                model_summary = {'model_name': model_name, 'status': 'abnormal_performance', **results['nested_cv']}
                all_models_summary.append(model_summary)
                continue
            
            # 최종 모델 저장
            if final_model is not None:
                final_models[model_name] = {
                    'model': final_model,
                    'results': results
                }
            
            # 결과 요약
            nested_cv_results = results.get('nested_cv', {})
            model_summary = {'model_name': model_name, 'status': 'completed', **nested_cv_results}
            
            # 하이퍼파라미터 전략 정보 추가
            model_summary['hyperparameter_strategy'] = results.get('hyperparameter_strategy', 'unknown')
            
            # 테스트 성능 추가 (있는 경우)
            if 'test_performance' in results:
                for key, value in results['test_performance'].items():
                    model_summary[f'test_{key}'] = value
            
            all_models_summary.append(model_summary)
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR for model {model_name}: {e}")
            logger.error(traceback.format_exc())
            all_models_summary.append({'model_name': model_name, 'status': 'error', 'error': str(e)})
            continue
            
        model_run_end_time = time.time()
        logger.info(f"Total time for model '{model_name}': {model_run_end_time - model_run_start_time:.2f} seconds")

    # --- Final Summary ---
    summary_start_time = time.time()
    logger.info("\n--- Overall Summary ---")
    
    if all_models_summary:
        # 성공한 모델들
        successful_models = [m for m in all_models_summary if m.get('status') == 'completed']
        failed_models = [m for m in all_models_summary if m.get('status') in ['error', 'abnormal_performance']]
        
        if successful_models:
            summary_df = pd.DataFrame(successful_models)
            summary_filename = os.path.join(args.output_dir, "model_comparison_summary.csv")
            summary_df.to_csv(summary_filename, index=False)
            logger.info(f"Model comparison summary saved to: {summary_filename}")
            logger.info("Successful Models Summary:")
            logger.info(summary_df.to_string())
            
            # 논문용 표 생성
            nested_cv_table = generate_nested_cv_table(all_models_summary, args.output_dir)
            if nested_cv_table is not None:
                logger.info("\nNested CV Results Table for Publication:")
                logger.info(nested_cv_table.to_string())
        
        if failed_models:
            error_df = pd.DataFrame(failed_models)
            error_filename = os.path.join(args.output_dir, "model_errors_summary.csv")
            error_df.to_csv(error_filename, index=False)
            logger.info(f"Model errors summary saved to: {error_filename}")
        
        # 최종 모델 정보 저장
        if final_models:
            models_info = {
                name: {
                    'best_params': info['results'].get('best_params', {}),
                    'hyperparameter_strategy': info['results'].get('hyperparameter_strategy', 'unknown'),
                    'nested_cv_metrics': info['results'].get('nested_cv', {}),
                    'test_metrics': info['results'].get('test_performance', {})
                } 
                for name, info in final_models.items()
            }
            save_results(models_info, args.output_dir, filename="final_models_info.json")
            logger.info(f"Final models info saved to: {os.path.join(args.output_dir, 'final_models_info.json')}")
            
            # 하이퍼파라미터 전략 요약
            strategy_summary = {}
            for name, info in final_models.items():
                strategy = info['results'].get('hyperparameter_strategy', 'unknown')
                if strategy not in strategy_summary:
                    strategy_summary[strategy] = []
                strategy_summary[strategy].append(name)
            
            logger.info("\nHyperparameter Strategy Summary:")
            for strategy, models in strategy_summary.items():
                logger.info(f"  {strategy}: {', '.join(models)}")
    
    else:
        logger.info("No models were successfully processed.")
    
    summary_end_time = time.time()
    logger.info(f"Summary Duration: {summary_end_time - summary_start_time:.2f} seconds")

    overall_end_time = time.time()
    logger.info("\n--- Framework Execution Finished ---")
    logger.info(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    
    # 실행 요약 로깅
    logger.info(f"\n--- Execution Summary ---")
    logger.info(f"Data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Models processed: {len(models_to_run)}")
    logger.info(f"Successful models: {len([m for m in all_models_summary if m.get('status') == 'completed'])}")
    logger.info(f"Failed models: {len([m for m in all_models_summary if m.get('status') in ['error', 'abnormal_performance']])}")
    
    if args.use_optuna_final:
        logger.info(f"Hyperparameter optimization: Optuna (Strategy A) with {args.optuna_trials} trials")
    elif args.use_nested_cv_consensus:
        logger.info(f"Hyperparameter optimization: Nested CV consensus (Strategy B)")
    else:
        logger.info(f"Hyperparameter optimization: Separate CV (default)")
    
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
