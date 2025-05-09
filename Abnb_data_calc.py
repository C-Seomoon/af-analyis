import af_analysis
import os
from af_analysis import analysis
from af_analysis import data
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def calculate_metrics(my_data):
    analysis.pdockq(my_data, verbose=False)
    analysis.pdockq2(my_data, verbose=False)
    analysis.mpdockq(my_data, verbose=False)
    analysis.LIS_matrix(my_data, verbose=False)
    analysis.add_interface_metrics(my_data, verbose=False)
    return my_data

def process_pdb(pdb_id):
    try:
        pdb_path = os.path.join(base_path, f"{pdb_id}_{pdb_id}")
        if os.path.exists(pdb_path):
            my_data = af_analysis.Data(pdb_path, format="AF3")
            calculate_metrics(my_data)
            return my_data
        return None
    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")
        return None

# 메인 코드
if __name__ == "__main__":
    df = pd.read_csv("/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/input_datastructure.tsv", sep="\t")
    pdb_list = df['PDB_ID'].unique()
    base_path = "/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results"
    
    # 사용할 CPU 코어 수 결정 (전체 코어의 80%를 사용하거나 사용자 지정)
    num_cores = int(cpu_count() * 0.8)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # 병렬 처리 실행
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_pdb, pdb_list),
            total=len(pdb_list),
            desc="Processing PDB structures"
        ))
    
    # None 값 제거
    results = [r for r in results if r is not None]
    
    # 결과 병합
    final_data = data.concat_data(results)
    
    # 결과 저장
    final_data.df.to_csv("/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/pdockq_results.tsv", sep="\t")
    print("Analysis complete and results saved!")