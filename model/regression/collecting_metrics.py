import os
import json
import pandas as pd
import glob

def collect_metrics_data(results_dir):
    """
    results_dir 내의 모든 모델의 모든 폴드에서 metrics.json 파일을 수집하고 DataFrame으로 반환합니다.
    
    Args:
        results_dir (str): 결과 디렉토리 경로 
                          (예: "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/results/rg_final_20250425_215156")
    
    Returns:
        pd.DataFrame: 모델과 폴드별 메트릭이 포함된 DataFrame
    """
    # 결과를 저장할 리스트
    results = []
    
    # 결과 디렉토리에서 모든 모델 폴더 찾기
    model_dirs = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) 
                 and d not in ('__pycache__', '.ipynb_checkpoints')]
    
    # 각 모델 폴더 처리
    for model_name in model_dirs:
        model_dir = os.path.join(results_dir, model_name)
        
        # 모델 폴더에서 fold_* 디렉토리 찾기
        fold_dirs = glob.glob(os.path.join(model_dir, "fold_*"))
        
        # 각 fold 폴더 처리
        for fold_dir in fold_dirs:
            # fold 번호 추출
            fold_num = os.path.basename(fold_dir).replace("fold_", "")
            
            # metrics.json 파일 경로
            metrics_file = os.path.join(fold_dir, "metrics.json")
            
            if os.path.exists(metrics_file):
                # metrics.json 파일 읽기
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # metrics 딕셔너리에 모델 이름과 폴드 번호 추가
                metrics["model"] = model_name
                metrics["fold"] = fold_num
                
                # 결과 리스트에 추가
                results.append(metrics)
    
    # 결과를 DataFrame으로 변환
    if results:
        df = pd.DataFrame(results)
        
        # 열 순서 조정 (모델과 폴드를 맨 앞으로)
        cols = ["model", "fold"] + [col for col in df.columns if col not in ["model", "fold"]]
        df = df[cols]
        
        return df
    else:
        return pd.DataFrame()

def save_metrics_table(results_dir, output_file=None):
    """
    메트릭 데이터를 수집하고 CSV 파일로 저장합니다.
    
    Args:
        results_dir (str): 결과 디렉토리 경로
        output_file (str, optional): 출력 CSV 파일 경로. None이면 results_dir에 'metrics_by_fold.csv' 저장
    
    Returns:
        pd.DataFrame: 수집된 메트릭 데이터
    """
    # 메트릭 데이터 수집
    df = collect_metrics_data(results_dir)
    
    if df.empty:
        print(f"No metrics data found in {results_dir}")
        return df
    
    # 출력 파일 경로 설정
    if output_file is None:
        output_file = os.path.join(results_dir, "metrics_by_fold.csv")
    
    # CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"Metrics data saved to {output_file}")
    
    # 문자열 필드와 수치 필드 구분
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["model", "fold"]]
    
    # 수치 필드만 평균 및 표준편차 계산
    if numeric_cols:
        summary_df = df.groupby("model")[numeric_cols].agg(['mean', 'std'])
        summary_file = os.path.join(os.path.dirname(output_file), "metrics_summary_by_model.csv")
        summary_df.to_csv(summary_file)
        print(f"Summary statistics saved to {summary_file}")
    else:
        print("No numeric columns found for summary statistics")
    
    return df

# 사용 예시
if __name__ == "__main__":
    # 결과 디렉토리 경로 설정
    results_dir = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/results/rg_renewal_test_20250429_033939"
    
    # 메트릭 데이터 수집 및 저장
    metrics_df = save_metrics_table(results_dir)
    
    # 데이터 미리보기 출력
    if not metrics_df.empty:
        print("\nMetrics Data Preview:")
        print(metrics_df.head())
        
        # 성능 지표별 모델 순위 (수치 데이터만)
        if "r2" in metrics_df.columns:
            print("\nModel Rankings by R2:")
            r2_ranking = metrics_df.groupby("model")["r2"].mean().sort_values(ascending=False)
            print(r2_ranking)
        
        if "rmse" in metrics_df.columns:
            print("\nModel Rankings by RMSE:")
            rmse_ranking = metrics_df.groupby("model")["rmse"].mean().sort_values()
            print(rmse_ranking)
