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
import statistics
import collections
import scipy.stats as stats

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SHAP 임포트
import shap

# Optuna 임포트 (선택적)
try:
    import optuna
    from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
    logger.info("Optuna is available for hyperparameter optimization.")
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not found. Using RandomizedSearchCV only.")

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
    return {
        'rf': RandomForestRegressorModel(),
        'lgbm': LGBMRegressorModel(),
        'xgb': XGBoostRegressorModel(),
        'lr': LinearRegressionModel(),
        'ridge': RidgeModel(),
        'lasso': LassoModel(),
        'en': ElasticNetModel()
    }

def calculate_confidence_intervals(all_fold_metrics, confidence=0.95):
    """Nested CV 결과에 대한 95% 신뢰구간 계산"""
    ci_results = {}
    alpha = 1 - confidence
    
    for key in ['r2', 'mse', 'mae', 'rmse']:
        values = [m[key] for m in all_fold_metrics if m and key in m and m[key] is not None]
        if len(values) > 1:
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            
            # t-분포 기반 신뢰구간
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * std / np.sqrt(n)
            
            ci_results[key] = {
                'mean': mean,
                'std': std,
                'ci_lower': mean - margin_error,
                'ci_upper': mean + margin_error,
                'n_folds': n
            }
    
    return ci_results

def aggregate_params(best_params_list):
    """파라미터 리스트를 합의하여 최종 파라미터 결정 (전략 B)"""
    if not best_params_list:
        return {}
    
    agg = {}
    for key in best_params_list[0]:
        vals = [d[key] for d in best_params_list if key in d and d[key] is not None]
        if not vals:
            continue
            
        if isinstance(vals[0], (int, float)):
            # 로그 스케일 파라미터는 기하평균 사용
            if key in ["learning_rate", "reg_lambda", "reg_alpha", "alpha", "l1_ratio"]:
                try:
                    # 0보다 큰 값들만 사용
                    positive_vals = [v for v in vals if v > 0]
                    if positive_vals:
                        agg[key] = float(np.exp(np.mean(np.log(positive_vals))))
                    else:
                        agg[key] = type(vals[0])(round(statistics.mean(vals)))
                except (ValueError, OverflowError):
                    agg[key] = type(vals[0])(round(statistics.mean(vals)))
            else:
                # 일반적인 파라미터는 산술평균
                agg[key] = type(vals[0])(round(statistics.mean(vals)))
        else:  # str, bool 등
            agg[key] = collections.Counter(vals).most_common(1)[0][0]
    
    return agg

def aggregate_nested_cv_params(model_output_dir, outer_folds):
    """전략 B: Nested CV 결과에서 파라미터 합의"""
    best_params_list = []
    
    for fold in range(1, outer_folds + 1):
        params_path = os.path.join(model_output_dir, f"fold_{fold}", "best_params.json")
        if os.path.exists(params_path):
            try:
                with open(params_path, 'r') as f:
                    best_params_list.append(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load params from fold {fold}: {e}")
    
    if not best_params_list:
        logger.error("No valid parameter files found for aggregation")
        return {}
    
    return aggregate_params(best_params_list)

def convert_grid_to_optuna_space(param_grid):
    """기존 하이퍼파라미터 grid를 Optuna distributions으로 변환"""
    if not OPTUNA_AVAILABLE:
        return param_grid
    
    optuna_space = {}
    
    for key, values in param_grid.items():
        if isinstance(values, list):
            if all(isinstance(v, int) for v in values):
                optuna_space[key] = optuna.distributions.IntUniformDistribution(
                    min(values), max(values))
            elif all(isinstance(v, float) for v in values):
                if key in ["learning_rate", "reg_lambda", "reg_alpha", "alpha"]:
                    # 로그 스케일 파라미터
                    optuna_space[key] = optuna.distributions.LogUniformDistribution(
                        min(values), max(values))
                else:
                    optuna_space[key] = optuna.distributions.UniformDistribution(
                        min(values), max(values))
            elif all(isinstance(v, str) for v in values):
                optuna_space[key] = optuna.distributions.CategoricalDistribution(values)
            else:
                # 혼합 타입의 경우 categorical로 처리
                optuna_space[key] = optuna.distributions.CategoricalDistribution(values)
        else:
            # 단일 값인 경우 그대로 유지
            optuna_space[key] = values
    
    return optuna_space

def get_best_hyperparameters_optuna(X, y, query_ids, model_obj, inner_folds=3, 
                                   n_trials=200, n_jobs=1, random_state=42,
                                   scoring_metric='neg_mean_squared_error'):
    """전략 A: Optuna로 전체 학습 데이터에서 최적 하이퍼파라미터 찾기"""
    
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available. Falling back to RandomizedSearchCV.")
        return get_best_hyperparameters_separate_cv(
            X, y, query_ids, model_obj, inner_folds, n_trials, n_jobs, random_state, scoring_metric
        )
    
    logger.info("Finding optimal hyperparameters using Optuna...")
    
    # 내부 CV 설정
    if query_ids is not None:
        inner_cv = GroupKFold(n_splits=inner_folds)
    else:
        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    
    # 모델별 n_jobs 조정
    model_name = model_obj.get_model_name()
    if model_name in ('lr', 'ridge', 'lasso', 'en'):
        cv_n_jobs = 1
    else:
        cv_n_jobs = min(4, n_jobs)
    
    try:
        # Optuna 파라미터 공간 시도 (있으면 사용, 없으면 기존 방식)
        if hasattr(model_obj, 'get_optuna_space'):
            param_distributions = model_obj.get_optuna_space()
        else:
            # 기존 grid를 optuna distributions으로 변환
            param_distributions = convert_grid_to_optuna_space(model_obj.get_hyperparameter_grid())
        
        # Optuna Study 설정
        direction = "maximize" if scoring_metric in ['r2', 'explained_variance'] else "minimize"
        sampler = optuna.samplers.TPESampler(seed=random_state, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        search = OptunaSearchCV(
            estimator=model_obj.create_model(params=None, n_jobs=n_jobs),
            param_distributions=param_distributions,
            n_trials=n_trials,
            cv=inner_cv,
            scoring=scoring_metric,
            study=study,
            refit=True,
            n_jobs=cv_n_jobs,
            verbose=1
        )
        
        # 전체 데이터로 튜닝
        if query_ids is not None:
            search.fit(X, y, groups=query_ids)
        else:
            search.fit(X, y)
        
        logger.info(f"Best parameters (Optuna): {search.best_params_}")
        logger.info(f"Best CV score (Optuna): {search.best_score_:.4f}")
        
        return search.best_params_, search.best_score_
        
    except Exception as e:
        logger.error(f"Error during Optuna optimization: {e}")
        logger.warning("Falling back to RandomizedSearchCV.")
        return get_best_hyperparameters_separate_cv(
            X, y, query_ids, model_obj, inner_folds, n_trials, n_jobs, random_state, scoring_metric
        )

def generate_nested_cv_table(all_models_summary, output_dir):
    """논문용 Nested CV 결과 표 생성"""
    
    table_data = []
    for model_summary in all_models_summary:
        if model_summary.get('status') == 'completed':
            model_name = model_summary['model_name']
            
            # 각 메트릭별 결과 정리
            row = {'Model': model_name}
            for metric in ['r2', 'mse', 'mae', 'rmse']:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                
                if mean_key in model_summary and std_key in model_summary:
                    mean_val = model_summary[mean_key]
                    std_val = model_summary[std_key]
                    
                    if mean_val is not None and std_val is not None:
                        # 논문 형식: "mean ± std"
                        row[metric.upper()] = f"{mean_val:.3f} ± {std_val:.3f}"
                        
                        # 95% CI 계산 (간단 근사)
                        ci_lower = mean_val - 1.96 * std_val
                        ci_upper = mean_val + 1.96 * std_val
                        row[f'{metric.upper()}_CI'] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            
            table_data.append(row)
    
    if not table_data:
        logger.warning("No completed models found for table generation")
        return None
    
    # DataFrame으로 변환 후 저장
    df = pd.DataFrame(table_data)
    csv_path = os.path.join(output_dir, "nested_cv_results_table.csv")
    df.to_csv(csv_path, index=False)
    
    # LaTeX 형식으로도 저장
    try:
        latex_table = df.to_latex(index=False, float_format="%.3f")
        latex_path = os.path.join(output_dir, "nested_cv_results_table.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to: {latex_path}")
    except Exception as e:
        logger.warning(f"Could not save LaTeX table: {e}")
    
    logger.info(f"Nested CV results table saved to: {csv_path}")
    return df

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

    # 95% 신뢰구간 계산 및 저장
    try:
        ci_results = calculate_confidence_intervals(all_fold_metrics)
        avg_metrics['confidence_intervals'] = ci_results
        save_results(ci_results, model_output_dir, filename="confidence_intervals.json")
        
        # 신뢰구간 로깅
        logger.info("95% Confidence Intervals:")
        for metric, ci_data in ci_results.items():
            logger.info(f"  {metric.upper()}: {ci_data['mean']:.4f} ± {ci_data['std']:.4f} "
                       f"(95% CI: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}])")
    except Exception as e:
        logger.error(f"Error calculating confidence intervals: {e}")

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
    for metric in ['r2', 'mse', 'mae', 'rmse']:
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
        n_jobs=n_jobs,
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
    
    model_output_dir = os.path.join(output_dir, model_name)
    
    # 최종 하이퍼파라미터 확정 전략 선택
    if getattr(args, 'use_optuna_final', False):
        # 전략 A: Optuna로 전체 데이터 재튜닝
        logger.info("Using Strategy A: Optuna optimization on full training data")
        best_params, best_cv_score = get_best_hyperparameters_optuna(
            X=X_train, y=y_train, query_ids=query_ids_train,
            model_obj=model_obj,
            inner_folds=args.inner_folds,
            n_trials=getattr(args, 'optuna_trials', 200),
            n_jobs=n_jobs,
            random_state=args.random_state,
            scoring_metric=args.scoring
        )
        results['hyperparameter_strategy'] = 'optuna_full_retune'
        
    elif getattr(args, 'use_nested_cv_consensus', False):
        # 전략 B: Nested CV 결과 합의
        logger.info("Using Strategy B: Consensus of nested CV best params")
        best_params = aggregate_nested_cv_params(model_output_dir, args.outer_folds)
        best_cv_score = None  # 별도 계산하지 않음
        results['hyperparameter_strategy'] = 'nested_cv_consensus'
        
        if not best_params:
            logger.error("Could not aggregate parameters from nested CV. Falling back to separate CV.")
            best_params, best_cv_score = get_best_hyperparameters_separate_cv(
                X=X_train, y=y_train, query_ids=query_ids_train,
                model_obj=model_obj,
                inner_folds=args.inner_folds,
                random_iter=args.random_iter,
                n_jobs=n_jobs,
                random_state=args.random_state,
                scoring_metric=args.scoring
            )
            results['hyperparameter_strategy'] = 'separate_cv_fallback'
    else:
        # 기본 전략: 별도 CV
        logger.info("Using default strategy: Separate CV on full training data")
        best_params, best_cv_score = get_best_hyperparameters_separate_cv(
            X=X_train, y=y_train, query_ids=query_ids_train,
            model_obj=model_obj,
            inner_folds=args.inner_folds,
            random_iter=args.random_iter,
            n_jobs=n_jobs,
            random_state=args.random_state,
            scoring_metric=args.scoring
        )
        results['hyperparameter_strategy'] = 'separate_cv'
    
    results['best_params'] = best_params
    results['hyperparameter_tuning_cv_score'] = best_cv_score
    
    # 최종 파라미터 저장
    save_results({
        'strategy': results['hyperparameter_strategy'],
        'best_params': best_params,
        'cv_score': best_cv_score
    }, model_output_dir, filename="final_hyperparameters.json")
    
    # Step 3: Final Model Training
    logger.info("=== Step 3: Final Model Training ===")
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
    parser = argparse.ArgumentParser(description="Advanced Regression Analysis with Nested CV and Optuna")
    
    # 기본 인자들
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--target_column", type=str, required=True, help="Target column name for regression")
    parser.add_argument("--models", type=str, nargs='+', 
                       choices=['rf', 'lgbm', 'xgb', 'lr', 'ridge', 'lasso', 'en'], 
                       default=['rf', 'lgbm'], help="List of models to run")
    
    # CV 설정
    parser.add_argument("--outer_folds", type=int, default=5, help="Number of outer CV folds")
    parser.add_argument("--inner_folds", type=int, default=3, help="Number of inner CV folds")
    parser.add_argument("--random_iter", type=int, default=50, help="Number of random search iterations")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--scoring", type=str, default="neg_mean_squared_error", 
                       choices=['neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2', 
                               'neg_mean_absolute_error'], help="Scoring metric for CV")
    
    # 연산 관련
    parser.add_argument("--n_jobs", type=int, default=0, help="Number of parallel jobs (0 = auto)")
    parser.add_argument("--shap_background_samples", type=int, default=100, 
                       help="Number of background samples for SHAP")
    
    # Optuna 옵션들
    parser.add_argument("--use_optuna_final", action="store_true", 
                       help="Use Optuna for final hyperparameter tuning (Strategy A)")
    parser.add_argument("--use_nested_cv_consensus", action="store_true", 
                       help="Use consensus of nested CV params (Strategy B)")
    parser.add_argument("--optuna_trials", type=int, default=200, 
                       help="Number of Optuna trials for final tuning")
    
    # 데이터 관련
    parser.add_argument("--query_id_column", type=str, default=None, 
                       help="Column name for query/group IDs (for GroupKFold)")
    parser.add_argument("--test_file", type=str, default=None, 
                       help="Path to separate test file (optional)")
    parser.add_argument("--train_test_split", type=float, default=0.8, 
                       help="Train/test split ratio if no separate test file")
    
    # 추가 옵션들
    parser.add_argument("--skip_shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--generate_paper_table", action="store_true", 
                       help="Generate paper-ready result tables")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일 로깅 설정
    log_file = os.path.join(args.output_dir, "regression_analysis.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("=== Advanced Regression Analysis Started ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Optuna 옵션 체크
    if args.use_optuna_final and not OPTUNA_AVAILABLE:
        logger.error("Optuna was requested but is not available. Please install optuna.")
        return
    
    # 상호 배타적 옵션 체크
    if args.use_optuna_final and args.use_nested_cv_consensus:
        logger.warning("Both Optuna and consensus strategies specified. Using Optuna (Strategy A).")
        args.use_nested_cv_consensus = False
    
    # 64코어 환경 최적화
    setup_cpu_environment()
    
    # 권장 설정 적용
    if args.n_jobs == 0:
        args.n_jobs = 64  # 전체 코어 활용
        logger.info("64-core optimized settings applied")
    
    # 권장 CV 설정
    if not hasattr(args, 'outer_folds') or args.outer_folds != 5:
        args.outer_folds = 5
        logger.info("Applied recommended outer_folds=5")
    
    if not hasattr(args, 'inner_folds') or args.inner_folds != 3:
        args.inner_folds = 3
        logger.info("Applied recommended inner_folds=3")
    
    # SHAP 샘플 수 최적화
    if args.shap_background_samples < 300:
        args.shap_background_samples = 300
        logger.info("Increased SHAP background samples to 300")
    
    try:
        # 데이터 로딩
        logger.info("Loading and preprocessing data...")
        if args.test_file:
            # 별도의 테스트 파일이 있는 경우
            X_train, y_train, query_ids_train = load_and_preprocess_data(
                args.input_file, args.target_column, args.query_id_column
            )
            X_test, y_test, query_ids_test = load_and_preprocess_data(
                args.test_file, args.target_column, args.query_id_column
            )
            logger.info(f"Loaded separate train ({X_train.shape}) and test ({X_test.shape}) sets")
        else:
            # 단일 파일에서 분할
            from sklearn.model_selection import train_test_split
            X_full, y_full, query_ids_full = load_and_preprocess_data(
                args.input_file, args.target_column, args.query_id_column
            )
            
            if query_ids_full is not None:
                # GroupKFold 방식 분할
                from sklearn.model_selection import GroupShuffleSplit
                splitter = GroupShuffleSplit(n_splits=1, test_size=1-args.train_test_split, 
                                           random_state=args.random_state)
                train_idx, test_idx = next(splitter.split(X_full, y_full, groups=query_ids_full))
                
                X_train = X_full.iloc[train_idx]
                y_train = y_full.iloc[train_idx]
                query_ids_train = query_ids_full.iloc[train_idx]
                
                X_test = X_full.iloc[test_idx]
                y_test = y_full.iloc[test_idx]
            else:
                # 일반 분할
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=1-args.train_test_split, 
                    random_state=args.random_state
                )
                query_ids_train = None
            
            logger.info(f"Split data into train ({X_train.shape}) and test ({X_test.shape}) sets")
        
        # 모델 객체 획득
        available_models = get_available_models()
        
        # 모든 모델 결과 저장
        all_models_summary = []
        
        # 예상 실행시간 알림
        n_tree_models = len([m for m in args.models if m in ['rf', 'lgbm', 'xgb']])
        n_linear_models = len([m for m in args.models if m in ['lr', 'ridge', 'lasso', 'en']])
        
        estimated_time = (n_tree_models * 20 + n_linear_models * 3 + 
                         n_tree_models * 15 + 3 * len(args.models))
        
        logger.info(f"Estimated total runtime: ~{estimated_time} minutes")
        logger.info(f"Tree models: {n_tree_models} (Optuna), Linear models: {n_linear_models} (Random)")
        
        # 각 모델에 대해 평가 수행
        for model_name in args.models:
            if model_name not in available_models:
                logger.error(f"Model '{model_name}' not available. Skipping.")
                continue
            
            model_obj = available_models[model_name]
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting evaluation for: {model_obj.get_display_name()}")
            logger.info(f"{'='*60}")
            
            model_start_time = time.time()
            
            try:
                # 포괄적 모델 평가 수행
                results, final_model = comprehensive_model_evaluation(
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    query_ids_train=query_ids_train,
                    model_obj=model_obj,
                    args=args,
                    output_dir=args.output_dir
                )
                
                # 결과 요약
                summary = {
                    'model_name': model_name,
                    'model_display_name': model_obj.get_display_name(),
                    'status': 'completed',
                    'processing_time': time.time() - model_start_time,
                    **results.get('nested_cv', {}),
                    'hyperparameter_strategy': results.get('hyperparameter_strategy', 'unknown'),
                    'best_params': results.get('best_params', {}),
                }
                
                if 'test_performance' in results:
                    summary['test_performance'] = results['test_performance']
                
                all_models_summary.append(summary)
                
                logger.info(f"Model {model_name} completed successfully in "
                           f"{summary['processing_time']:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing model {model_name}: {e}")
                logger.error(traceback.format_exc())
                
                # 실패한 모델도 요약에 포함
                summary = {
                    'model_name': model_name,
                    'model_display_name': model_obj.get_display_name(),
                    'status': 'failed',
                    'error': str(e),
                    'processing_time': time.time() - model_start_time
                }
                all_models_summary.append(summary)
        
        # 전체 결과 요약 저장
        save_results(all_models_summary, args.output_dir, filename="all_models_summary.json")
        
        # 논문용 표 생성 (옵션)
        if args.generate_paper_table:
            logger.info("Generating paper-ready result tables...")
            try:
                table_df = generate_nested_cv_table(all_models_summary, args.output_dir)
                if table_df is not None:
                    logger.info("Paper-ready tables generated successfully")
                else:
                    logger.warning("No valid results for table generation")
            except Exception as e:
                logger.error(f"Error generating paper tables: {e}")
        
        # 최종 요약 로깅
        logger.info("\n" + "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        
        completed_models = [s for s in all_models_summary if s['status'] == 'completed']
        failed_models = [s for s in all_models_summary if s['status'] == 'failed']
        
        logger.info(f"Total models processed: {len(all_models_summary)}")
        logger.info(f"Successfully completed: {len(completed_models)}")
        logger.info(f"Failed: {len(failed_models)}")
        
        if completed_models:
            logger.info("\nCompleted models performance (Nested CV R²):")
            for model in completed_models:
                r2_mean = model.get('r2_mean', 'N/A')
                r2_std = model.get('r2_std', 'N/A')
                if r2_mean != 'N/A' and r2_std != 'N/A':
                    logger.info(f"  {model['model_display_name']}: {r2_mean:.4f} ± {r2_std:.4f}")
                else:
                    logger.info(f"  {model['model_display_name']}: Performance data not available")
        
        if failed_models:
            logger.info("\nFailed models:")
            for model in failed_models:
                logger.info(f"  {model['model_display_name']}: {model.get('error', 'Unknown error')}")
        
        total_time = time.time() - overall_start_time
        logger.info(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info("=== Advanced Regression Analysis Completed ===")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()