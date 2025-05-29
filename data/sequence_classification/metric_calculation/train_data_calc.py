#!/usr/bin/env python3

import af_analysis
from af_analysis import analysis
import pandas as pd
import numpy as np
from datetime import datetime



from concurrent.futures import ProcessPoolExecutor, as_completed
import os, sys, time, multiprocessing

def process_pdb_id(antibody_id,antigen_id):
    """개별 PDB ID 처리 함수"""
    try:
        pair = f"{antibody_id}_{antigen_id}"
        print(f"Processing {pair}")
        base_dir = f"/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results/{pair}"
        if not os.path.exists(base_dir):
            return None, f"Directory not found for {pair}"

        my_data = af_analysis.data.Data(directory=base_dir)
        # my_data.prep_dockq(verbose=False)
        # analysis.calculate_dockQ(my_data,
        #             rec_chains='A', lig_chains='H',
        #             native_rec_chains='A', native_lig_chains='H',
        #             verbose=False)

        # analysis.pdockq(my_data, verbose=False)
        # analysis.pdockq2(my_data, verbose=False)
        # analysis.mpdockq(my_data, verbose=False)
        # analysis.LIS_matrix(my_data, verbose=False)
        # my_data.analyze_interfaces(verbose=False)
        # my_data.analyze_chains()
        # my_data.add_chain_rmsd(align_chain='A', rmsd_chain='H')
        # my_data.add_rmsd_scale()
        my_data.extract_chain_columns(verbose=True)
        return my_data, None
    except Exception as e:
        return None, f"Error processing {pair}: {e}"

if __name__ == "__main__":
    start_time = time.time()

    # 입력 파일 확인
    input_file = '/home/cseomoon/appl/af_analysis-0.1.4/data/sequence_classification/metric_calculation/negative_sorted_decoy_set_train.csv'
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    # 출력 디렉토리 준비
    output_dir = "/home/cseomoon/appl/af_analysis-0.1.4/data/sequence_classification/metric_calculation"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/chain_metric_{timestamp}.csv"

    # negative_pairs 생성
    df = pd.read_csv(input_file)
    negative_pairs = [(row['antibody'], row['antigen']) for _, row in df.iterrows()]
    total_pairs = len(negative_pairs)
    print(f"Found {total_pairs} negative pairs to process")


   # 병렬 처리 설정
    num_workers = min(48, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_workers} workers for parallel processing")

    results = []
    errors = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # futures: { Future: (antibody, antigen) }
        futures = {
            executor.submit(process_pdb_id, ab, ag): (ab, ag)
            for ab, ag in negative_pairs
        }

        for future in as_completed(futures):
            ab, ag = futures[future]
            pair = f"{ab}_{ag}"
            result, error = future.result()

            completed += 1
            progress = (completed / total_pairs) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_pairs - completed) if completed else 0
            print(f"Progress: {progress:.1f}% ({completed}/{total_pairs}) - ETA: {eta:.1f}s")

            if result is not None:
                results.append(result)
                print(f"Completed {pair}")
            else:
                errors.append(error)
                print(error)

    # 결과 저장
    if results:
        merged = af_analysis.data.concat_data(results)
        merged.df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(results)}/{total_pairs} folders")
        print(f"Results saved to {output_file}")
    else:
        print("No valid results to save")

    if errors:
        print(f"Encountered {len(errors)} errors:")
        for err in errors:
            print("  -", err)

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f}min)")
