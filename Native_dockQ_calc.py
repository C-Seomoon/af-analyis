import af_analysis
from af_analysis import analysis
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_pdb_id(pdb_id):
    """개별 PDB ID 처리 함수"""
    try:
        print(f"Processing {pdb_id}")
        base_dir = f"/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results/{pdb_id}_{pdb_id}"
        my_data = af_analysis.data.Data(directory=base_dir)
        my_data.prep_dockq(verbose=False)
        
        for i, row in my_data.df.iterrows():
            if row['native_path'] is not None:
                af_analysis.analysis.calculate_dockQ(
                    my_data, model_idx=i, native_path=row['native_path'], 
                    rec_chains='A', lig_chains='H', 
                    native_rec_chains='A', native_lig_chains='H', 
                    verbose=False  # 병렬 처리 시 출력 끄는 것이 좋음
                )
        return my_data
    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")
        return None

# 메인 코드
if __name__ == "__main__":
    df = pd.read_csv('/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/input_datastructure.tsv', sep='\t')
    pdb_ids = list(np.unique(df['PDB_ID']))
    
    # 사용할 CPU 코어 수 설정 (전체 코어의 75% 사용)
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_workers} workers for parallel processing")
    
    # 병렬 처리 실행
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_pdb_id, pdb_id): pdb_id for pdb_id in pdb_ids}
        
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
                print(f"Completed {futures[future]}")
    
    # 결과 병합 및 저장
    if results:
        data_list = af_analysis.data.concat_data(results)
        data_list.df.to_csv("/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/new_dockq/native_dockQ_results.csv", index=False)
        print(f"Successfully processed {len(results)} folders")
    else:
        print("No valid results to save")