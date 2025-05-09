import af_analysis
from af_analysis import analysis
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_folder(folder_path):
    """개별 폴더 처리 함수"""
    try:
        data = af_analysis.Data(directory=folder_path)
        data.add_ranking_scores(verbose=False)  # 출력 끄기
        analysis.add_interface_metrics(data, verbose=False)  # 출력 끄기
        return data
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return None

# 메인 코드
base_dir = "/home/cseomoon/project/TMEM120A/inference_data"
folders = [os.path.join(base_dir, file) for file in os.listdir(base_dir) 
           if os.path.isdir(os.path.join(base_dir, file))]

# 사용할 CPU 코어 수 설정 (전체 코어의 75% 사용)
num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
print(f"Using {num_workers} workers for parallel processing")

# 병렬 처리 실행
results = []
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_folder, folders))

# None 값 제거
results = [r for r in results if r is not None]

# 결과 병합 및 저장
if results:
    data_list = af_analysis.data.concat_data(results)
    data_list.df.to_csv("/home/cseomoon/project/TMEM120A/new_TMEM120A_analysis_results.csv", index=False)
    print(f"Successfully processed {len(results)} folders")
else:
    print("No valid results to save")