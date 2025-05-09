import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import argparse

def load_model_results(model_dirs):
    """
    각 모델 디렉토리에서 결과 데이터 로드
    """
    models_data = {}
    
    for model_dir in model_dirs:
        # 디렉토리 이름에서 모델 유형 추출 (RF, LightGBM, XGBoost)
        dir_name = os.path.basename(model_dir)
        
        if 'RF_regressor' in dir_name:
            model_name = 'RandomForest'
            color = 'blue'
        elif 'lightgbm' in dir_name:
            model_name = 'LightGBM'
            color = 'green'
        elif 'xgboost' in dir_name:
            model_name = 'XGBoost'
            color = 'red'
        else:
            model_name = dir_name.split('_')[0]
            color = 'purple'
        
        # 비교 데이터 디렉토리 경로
        comparison_dir = os.path.join(model_dir, 'comparison_data')
        
        if not os.path.exists(comparison_dir):
            print(f"경고: {model_dir}에 비교 데이터가 없습니다.")
            continue
        
        try:
            # 1. 예측값 로드
            pred_file = os.path.join(comparison_dir, f'{model_name}_predictions.csv')
            if os.path.exists(pred_file):
                predictions = pd.read_csv(pred_file)
                y_test = predictions['actual']
                y_pred = predictions['predicted']
            else:
                print(f"경고: {pred_file} 파일이 없습니다.")
                continue
            
            # 2. 성능 지표 로드
            metrics_file = os.path.join(comparison_dir, f'{model_name}_metrics.csv')
            if os.path.exists(metrics_file):
                metrics = pd.read_csv(metrics_file)
                if len(metrics) > 0:
                    metrics_dict = metrics.iloc[0].to_dict()
                    training_time = metrics['training_time'].values[0] if 'training_time' in metrics.columns else None
                else:
                    print(f"경고: {metrics_file}에 데이터가 없습니다.")
                    continue
            else:
                print(f"경고: {metrics_file} 파일이 없습니다.")
                continue
            
            # 3. 특성 중요도 로드
            importance_file = os.path.join(comparison_dir, f'{model_name}_feature_importance.csv')
            if os.path.exists(importance_file):
                importance_df = pd.read_csv(importance_file)
            else:
                print(f"경고: {importance_file} 파일이 없습니다.")
                importance_df = pd.DataFrame()
            
            # 4. 교차 검증 점수 로드
            cv_file = os.path.join(comparison_dir, f'{model_name}_cv_scores.csv')
            if os.path.exists(cv_file):
                cv_df = pd.read_csv(cv_file)
                cv_scores = cv_df['cv_score'].values
            else:
                print(f"경고: {cv_file} 파일이 없습니다.")
                cv_scores = np.array([])
            
            # 5. 하이퍼파라미터 로드
            hyperparams_file = os.path.join(comparison_dir, f'{model_name}_hyperparams.json')
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    hyperparams = json.load(f)
            else:
                print(f"경고: {hyperparams_file} 파일이 없습니다.")
                hyperparams = {}
            
            # 모델 데이터 저장
            models_data[model_name] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'metrics': metrics_dict,
                'feature_importance': importance_df,
                'cv_scores': cv_scores,
                'hyperparams': hyperparams,
                'training_time': training_time,
                'color': color,
                'model_dir': model_dir
            }
            
            print(f"{model_name} 모델 데이터 로드 완료")
            
        except Exception as e:
            print(f"{model_name} 모델 데이터 로드 중 오류: {str(e)}")
    
    return models_data

def create_metrics_comparison(models_data, output_dir, timestamp):
    """
    모델 성능 지표 비교 테이블 생성
    """
    metrics_rows = []
    
    for model_name, data in models_data.items():
        metrics = data['metrics']
        metrics_row = {
            'Model': model_name,
            'MSE': metrics.get('MSE', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAE': metrics.get('MAE', np.nan),
            'R²': metrics.get('R2', np.nan),
            'Training Time (s)': data.get('training_time', np.nan)
        }
        metrics_rows.append(metrics_row)
    
    # DataFrame 생성
    metrics_table = pd.DataFrame(metrics_rows)
    
    # 결과 저장
    metrics_file = os.path.join(output_dir, f'metrics_comparison_{timestamp}.csv')
    metrics_table.to_csv(metrics_file, index=False)
    print(f"성능 지표 비교 테이블 저장: {metrics_file}")
    
    return metrics_table

def plot_predictions_comparison(models_data, output_dir, timestamp, point_size=20, point_alpha=0.5):
    """
    모델 예측값 비교 시각화
    """
    plt.figure(figsize=(12, 10))
    
    # 각 모델별로 자체 테스트 세트와 예측값 사용
    for model_name, data in models_data.items():
        r2 = data['metrics'].get('R2', 0)
        plt.scatter(data['y_test'], data['y_pred'], 
                   alpha=point_alpha, s=point_size, 
                   label=f"{model_name} (R²={r2:.4f})",
                   color=data['color'])
    
    # 모든 데이터의 최소/최대값 찾기
    min_vals = [min(data['y_test'].min(), data['y_pred'].min()) for data in models_data.values()]
    max_vals = [max(data['y_test'].max(), data['y_pred'].max()) for data in models_data.values()]
    global_min = min(min_vals)
    global_max = max(max_vals)
    
    # 이상적인 예측선(y=x) 추가
    plt.plot([global_min, global_max], [global_min, global_max], 'k--')
    
    # 그래프 설정
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Model Comparison: Predictions vs Actual Values', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    pred_file = os.path.join(output_dir, f'prediction_comparison_{timestamp}.png')
    plt.savefig(pred_file, dpi=300)
    plt.close()
    print(f"예측 비교 그래프 저장: {pred_file}")

def plot_residuals_comparison(models_data, output_dir, timestamp, point_size=20, point_alpha=0.5):
    """
    모델 잔차 비교 시각화
    """
    plt.figure(figsize=(12, 10))
    
    # 각 모델의 잔차 플롯
    for model_name, data in models_data.items():
        residuals = data['y_test'] - data['y_pred']
        plt.scatter(data['y_pred'], residuals, 
                   alpha=point_alpha, s=point_size, 
                   label=model_name,
                   color=data['color'])
    
    # 기준선(y=0) 추가
    plt.axhline(y=0, color='k', linestyle='-')
    
    # 그래프 설정
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Model Comparison: Residual Plots', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 저장
    residual_file = os.path.join(output_dir, f'residual_comparison_{timestamp}.png')
    plt.savefig(residual_file, dpi=300)
    plt.close()
    print(f"잔차 비교 그래프 저장: {residual_file}")

def plot_cv_scores_comparison(models_data, output_dir, timestamp):
    """
    교차 검증 점수 비교 시각화
    """
    # 교차 검증 점수가 있는 모델만 필터링
    cv_data = []
    
    for model_name, data in models_data.items():
        if len(data['cv_scores']) > 0:
            for score in data['cv_scores']:
                cv_data.append({
                    'Model': model_name,
                    'MSE': score
                })
    
    # CV 점수가 없으면 함수 종료
    if not cv_data:
        print("교차 검증 점수가 없어 그래프를 생성하지 않습니다.")
        return
    
    # 데이터프레임 생성 및 그래프 플롯
    cv_df = pd.DataFrame(cv_data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='MSE', data=cv_df)
    plt.title('Cross-Validation MSE Comparison')
    plt.tight_layout()
    
    # 저장
    cv_file = os.path.join(output_dir, f'cv_comparison_{timestamp}.png')
    plt.savefig(cv_file)
    plt.close()
    print(f"교차 검증 비교 그래프 저장: {cv_file}")

def plot_feature_importance_comparison(models_data, output_dir, timestamp, top_n=15):
    """
    특성 중요도 비교 시각화
    """
    # 특성 중요도가 있는 모델만 필터링
    models_with_importance = {k: v for k, v in models_data.items() 
                              if not v['feature_importance'].empty}
    
    if not models_with_importance:
        print("특성 중요도 데이터가 없어 그래프를 생성하지 않습니다.")
        return
    
    # 모든 모델의 상위 N개 특성 추출
    top_features = set()
    for model_name, data in models_with_importance.items():
        top_n_features = min(top_n, len(data['feature_importance']))
        model_top_features = data['feature_importance'].head(top_n_features)['feature'].tolist()
        top_features.update(model_top_features)
    
    top_features = list(top_features)
    
    # 비교용 데이터프레임 생성
    importance_df = pd.DataFrame(index=top_features)
    
    for model_name, data in models_with_importance.items():
        # 각 모델의 특성 중요도를 딕셔너리로 변환
        feature_imp_dict = dict(zip(
            data['feature_importance']['feature'], 
            data['feature_importance']['importance']
        ))
        
        # 상위 특성에 대한 중요도 할당 (없는 경우 0)
        importance_df[model_name] = [feature_imp_dict.get(feature, 0) for feature in top_features]
    
    # 중요도 정규화 (각 모델별로 최대값을 1로)
    for model_name in importance_df.columns:
        max_val = importance_df[model_name].max()
        if max_val > 0:
            importance_df[model_name] = importance_df[model_name] / max_val
    
    # 합계 기준으로 정렬
    importance_df['total'] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values('total', ascending=True).drop(columns=['total'])
    
    # 그래프 플롯
    plt.figure(figsize=(12, max(8, len(top_features) * 0.4)))
    importance_df.plot(kind='barh', figsize=(12, max(8, len(top_features) * 0.4)))
    plt.title('Normalized Feature Importance Comparison')
    plt.xlabel('Normalized Importance')
    plt.legend(title='Model')
    plt.tight_layout()
    
    # 저장
    importance_file = os.path.join(output_dir, f'feature_importance_comparison_{timestamp}.png')
    plt.savefig(importance_file)
    plt.close()
    print(f"특성 중요도 비교 그래프 저장: {importance_file}")

def create_summary_report(models_data, metrics_table, output_dir, timestamp):
    """
    모델 비교 결과 요약 리포트 생성
    """
    best_mse_model = metrics_table.loc[metrics_table['MSE'].idxmin()]['Model']
    best_r2_model = metrics_table.loc[metrics_table['R²'].idxmax()]['Model']
    
    # 훈련 시간이 있는 모델만 필터링
    time_df = metrics_table.dropna(subset=['Training Time (s)'])
    fastest_model = time_df.loc[time_df['Training Time (s)'].idxmin()]['Model'] if not time_df.empty else "N/A"
    
    report_file = os.path.join(output_dir, f'model_comparison_summary_{timestamp}.txt')
    with open(report_file, 'w') as f:
        f.write(f"=========== 모델 비교 분석 요약 ({timestamp}) ===========\n\n")
        
        f.write("1. 성능 지표 비교\n")
        f.write("-" * 50 + "\n")
        f.write(metrics_table.to_string(index=False) + "\n\n")
        
        f.write("2. 모델 순위\n")
        f.write("-" * 50 + "\n")
        f.write(f"MSE 기준 최고 모델: {best_mse_model}\n")
        f.write(f"R² 기준 최고 모델: {best_r2_model}\n")
        f.write(f"훈련 속도 최고 모델: {fastest_model}\n\n")
        
        f.write("3. 모델별 하이퍼파라미터\n")
        f.write("-" * 50 + "\n")
        for model_name, data in models_data.items():
            f.write(f"{model_name}:\n")
            for param_name, param_value in data['hyperparams'].items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write("\n")
        
        f.write("4. 결론 및 추천\n")
        f.write("-" * 50 + "\n")
        f.write(f"- 전반적인 성능이 가장 우수한 모델: {best_r2_model} (R² 기준)\n")
        f.write(f"- 예측 오차가 가장 작은 모델: {best_mse_model} (MSE 기준)\n")
        f.write(f"- 훈련 속도가 가장 빠른 모델: {fastest_model}\n\n")
        
        f.write("* 참고: 최종 모델 선택 시 성능뿐만 아니라 해석 가능성, 배포 용이성, 추론 속도 등도 고려하세요.\n")
    
    print(f"모델 비교 요약 리포트 저장: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='모델 비교 분석')
    parser.add_argument('--model_dirs', nargs='+', required=True,
                        help='모델 결과 디렉토리 경로들 (공백으로 구분)')
    parser.add_argument('--output_dir', default='regression_model_comparison_results',
                        help='비교 결과 저장 디렉토리 (기본값: model_comparison_results)')
    parser.add_argument('--point_size', type=float, default=20.0,
                        help='산점도 포인트 크기 (기본값: 20.0)')
    parser.add_argument('--point_alpha', type=float, default=0.5,
                        help='산점도 포인트 투명도 (기본값: 0.5)')
    parser.add_argument('--top_n_features', type=int, default=15,
                        help='특성 중요도 비교 시 포함할 상위 특성 수 (기본값: 15)')
    
    args = parser.parse_args()
    
    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과 저장 디렉토리 생성: {output_dir}")
    
    # 모델 결과 로드
    models_data = load_model_results(args.model_dirs)
    
    if len(models_data) < 2:
        print(f"비교할 모델이 충분하지 않습니다. 최소 2개 이상의 모델이 필요합니다.")
        return
    
    # 성능 지표 비교 테이블 생성
    metrics_table = create_metrics_comparison(models_data, output_dir, timestamp)
    
    # 예측값 비교 시각화
    plot_predictions_comparison(models_data, output_dir, timestamp, 
                                point_size=args.point_size, point_alpha=args.point_alpha)
    
    # 잔차 비교 시각화
    plot_residuals_comparison(models_data, output_dir, timestamp, 
                             point_size=args.point_size, point_alpha=args.point_alpha)
    
    # 교차 검증 점수 비교 시각화
    plot_cv_scores_comparison(models_data, output_dir, timestamp)
    
    # 특성 중요도 비교 시각화
    plot_feature_importance_comparison(models_data, output_dir, timestamp, top_n=args.top_n_features)
    
    # 요약 리포트 생성
    create_summary_report(models_data, metrics_table, output_dir, timestamp)
    
    print(f"\n모델 비교 분석 완료! 결과는 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
