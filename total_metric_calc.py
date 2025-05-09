import af_analysis
from af_analysis import analysis
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from datetime import datetime
import sys

def process_pdb_id(pdb_id):
    """개별 PDB ID 처리 함수"""
    try:
        print(f"Processing {pdb_id}")
        base_dir = f"/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results/{pdb_id}_{pdb_id}"
        if not os.path.exists(base_dir):
            return None, f"Directory not found for {pdb_id}"
            
        my_data = af_analysis.data.Data(directory=base_dir)
        #my_data.prep_dockq(verbose=False)
        # analysis.calculate_dockQ(my_data,
        #             rec_chains='A', lig_chains='H', 
        #             native_rec_chains='A', native_lig_chains='H', 
        #             verbose=False)
        
        #analysis.pdockq(my_data, verbose=False)
        #analysis.pdockq2(my_data, verbose=False)
        #analysis.mpdockq(my_data, verbose=False)
        #analysis.LIS_matrix(my_data, verbose=False)
        #analysis.add_interface_metrics(my_data, verbose=False)
        my_data.analyze_chains()   
        #my_data.add_chain_rmsd(align_chain='A', rmsd_chain='H')
        #my_data.add_rmsd_scale() 
        return my_data, None
    except Exception as e:
        return None, f"Error processing {pdb_id}: {str(e)}"

# 메인 코드
if __name__ == "__main__":
    start_time = time.time()
    
    # 입력 파일 확인
    input_file = '/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/input_datastructure.tsv'
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
        
    # 출력 디렉토리 확인
    output_dir = "/home/cseomoon/project/ABAG/AbNb_benchmark/datastructure/new_dockq"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 타임스탬프가 있는 출력 파일 이름 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/chain_metrics_results_{timestamp}.csv"
    
    try:
        df = pd.read_csv(input_file, sep='\t')
        pdb_ids = list(np.unique(df['PDB_ID']))
        total_pdb_ids = len(pdb_ids)
        
        print(f"Found {total_pdb_ids} unique PDB IDs to process")
        
        # 사용할 CPU 코어 수 설정 (전체 코어의 75% 사용)
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        print(f"Using {num_workers} workers for parallel processing")
        
        # 병렬 처리 실행
        results = []
        errors = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_pdb_id, pdb_id): pdb_id for pdb_id in pdb_ids}
            
            for future in as_completed(futures):
                pdb_id = futures[future]
                result, error = future.result()
                
                completed += 1
                progress = (completed / total_pdb_ids) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_pdb_ids - completed) if completed > 0 else 0
                
                print(f"Progress: {progress:.1f}% ({completed}/{total_pdb_ids}) - ETA: {eta:.1f} seconds")
                
                if result is not None:
                    results.append(result)
                    print(f"Completed {pdb_id}")
                else:
                    errors.append(error)
                    print(error)
        
        # 결과 병합 및 저장
        if results:
            data_list = af_analysis.data.concat_data(results)
            data_list.df.to_csv(output_file, index=False)
            print(f"Successfully processed {len(results)} out of {total_pdb_ids} folders")
            print(f"Results saved to {output_file}")
        else:
            print("No valid results to save")
            
        # 오류 요약 출력
        if errors:
            print(f"Encountered {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
                
        # 총 실행 시간 출력
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        sys.exit(1)