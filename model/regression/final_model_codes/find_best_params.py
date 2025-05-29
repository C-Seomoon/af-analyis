import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def collect_best_params(results_dir):
    """
    각 모델별, fold별 best parameter를 수집하고 정리하는 함수
    
    Args:
        results_dir (str): 결과 디렉토리 경로
    """
    results_dir = Path(results_dir)
    
    # 모델 디렉토리 찾기 (rf, en)
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and 
                 not d.name.startswith('fold_') and 
                 not d.name.endswith('_shap_analysis') and
                 d.name != 'comparison_results']
    
    # 결과를 저장할 딕셔너리
    all_params = defaultdict(dict)
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n=== {model_name.upper()} Model Parameters ===")
        
        # fold 디렉토리 찾기
        fold_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')]
        
        # 각 fold의 best parameters 수집
        fold_params = []
        for fold_dir in sorted(fold_dirs):
            fold_num = fold_dir.name.split('_')[1]
            params_file = fold_dir / "best_params.json"
            
            if params_file.exists():
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    fold_params.append(params)
                    all_params[model_name][f'fold_{fold_num}'] = params
                    
                    # 파라미터 출력
                    print(f"\nFold {fold_num}:")
                    for param, value in params.items():
                        print(f"  {param}: {value}")
        
        # 파라미터별 평균값과 최빈값 계산
        if fold_params:
            print("\nParameter Statistics:")
            
            # 모든 파라미터 이름 수집
            all_param_names = set()
            for params in fold_params:
                all_param_names.update(params.keys())
            
            for param in sorted(all_param_names):
                values = [p.get(param) for p in fold_params if param in p]
                
                # None이 아닌 값만 필터링
                valid_values = [v for v in values if v is not None]
                
                if valid_values:
                    # 숫자형 파라미터인 경우 평균 계산
                    if all(isinstance(v, (int, float)) for v in valid_values):
                        mean_value = sum(valid_values) / len(valid_values)
                        print(f"  {param}:")
                        print(f"    - Mean: {mean_value:.3f}")
                        print(f"    - Values: {valid_values}")
                    # 범주형 파라미터인 경우 최빈값 계산
                    else:
                        from collections import Counter
                        value_counts = Counter(valid_values)
                        most_common = value_counts.most_common(1)[0]
                        print(f"  {param}:")
                        print(f"    - Most common: {most_common[0]} (count: {most_common[1]})")
                        print(f"    - Values: {valid_values}")
    
    return all_params

def main():
    # 결과 디렉토리 경로
    results_dir = "/home/cseomoon/appl/af_analysis-0.1.4/model/regression/results/rg_renewal_test_20250429_033939-ppt"
    
    # 파라미터 수집
    all_params = collect_best_params(results_dir)
    
    # 결과를 DataFrame으로 변환하여 CSV로 저장
    for model_name, fold_params in all_params.items():
        # DataFrame 생성
        df = pd.DataFrame.from_dict(fold_params, orient='index')
        
        # CSV 파일로 저장
        output_file = os.path.join(results_dir, f"{model_name}_best_params_summary.csv")
        df.to_csv(output_file)
        print(f"\nSaved parameter summary to: {output_file}")

if __name__ == "__main__":
    main()
