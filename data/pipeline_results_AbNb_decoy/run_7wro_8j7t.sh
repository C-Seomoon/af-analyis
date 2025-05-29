#!/bin/bash
#SBATCH --job-name=af_7wro_8j7t
#SBATCH --output=pipeline_results_AbNb_decoy/slurm_7wro_8j7t_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --partition=skylake_short,rome_short,milan_short
#SBATCH --exclude=gpu[1-10]

# 환경 설정
source /home/cseomoon/miniconda3/etc/profile.d/conda.sh
conda activate Abnb

# Python 스크립트 실행
python3 -c '
import sys
import os
import time
import pandas as pd
from contextlib import contextmanager
from af_analysis.data import Data
from af_analysis import analysis

@contextmanager
def time_tracker(description):
    start = time.time()
    print(f"🔄 Starting: {description}")
    yield
    elapsed = time.time() - start
    print(f"✅ Completed: {description} ({elapsed:.2f}s)")

def calculate_all_metrics_per_query_optimized(query_group_data, **kwargs):
    query_name, query_df = query_group_data
    
    print(f"\n🚀 Processing query: {query_name} ({len(query_df)} models)")
    total_start = time.time()
    
    try:
        # Data 객체 생성
        with time_tracker("Data object creation"):
            data_dict = {}
            for col in query_df.columns:
                data_dict[col] = query_df[col].tolist()
            
            temp_data = Data(data_dict=data_dict)
        
        # 1. 기본 AF3 metrics
        with time_tracker("Basic AF3 metrics"):
            temp_data = (temp_data
                        .extract_chain_columns(verbose=False)
                        .analyze_chains(verbose=False)
                        .add_h3_l3_plddt(verbose=False))
                
        # 2. Interface metrics
        with time_tracker("Interface metrics"):
            analysis.add_interface_metrics(temp_data, verbose=False)
        
        # 3. PPI metrics
        with time_tracker("pDockQ calculations"):
            analysis.pdockq(temp_data, verbose=False)
            analysis.pdockq2(temp_data, verbose=False)
        
        with time_tracker("LIS matrix"):
            analysis.LIS_matrix(temp_data, verbose=False)
        
        # 4. piTM/pIS
        with time_tracker("piTM/pIS calculation"):
            temp_data.add_pitm_pis(cutoff=8.0, verbose=False)
        
        # 5. RMSD metrics
        with time_tracker("RMSD calculations"):
            temp_data = (temp_data
                        .add_chain_rmsd(align_chain="A", rmsd_chain="H")
                        .add_rmsd_scale())
        
        # 6. Rosetta metrics
        with time_tracker("Rosetta interface metrics"):
            temp_data.add_rosetta_metrics(n_jobs=1, verbose=False)
        
        # # 7. DockQ
        # if kwargs.get("calculate_dockq", False):
        #     with time_tracker("DockQ calculation"):
        #         temp_data.prep_dockq(native_dir=kwargs.get("native_dir"), verbose=False)
        #         analysis.calculate_dockq(temp_data, 
        #                                rec_chains="A", lig_chains="H",
        #                                native_rec_chains="A", native_lig_chains="H",
        #                                verbose=False)
        
        total_time = time.time() - total_start
        print(f"✨ Query {query_name} completed in {total_time:.2f}s")
        
        return query_name, temp_data.df
        
    except Exception as e:
        total_time = time.time() - total_start
        print(f"❌ Error processing query {query_name} after {total_time:.2f}s: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return query_name, None

# 쿼리 처리 함수
def process_query(query_name):
    base_path = "/home/cseomoon/project/ABAG/DB/AbNb_structure/AF3/decoy"
    target = os.path.join(base_path, query_name)
    
    try:
        # Data 객체 생성 및 처리
        my_data = Data(directory=target)
        df = my_data.df
        
        # 메트릭 계산
        for _, group_df in df.groupby("query"):
            name, result_df = calculate_all_metrics_per_query_optimized(
                (query_name, group_df),
                calculate_dockq=True,
                native_dir="/home/cseomoon/project/ABAG/DB/AbNb_structure/original_pdb"
            )
            
            if result_df is not None:
                output_file = os.path.join("pipeline_results_AbNb_decoy", f"metrics_{name}.csv")
                result_df.to_csv(output_file, index=False)
                print(f"Successfully processed {query_name}")
            else:
                print(f"Failed to process {query_name}")
                
    except Exception as e:
        print(f"Error processing {query_name}: {str(e)}")
        return False
    
    return True

# 메인 실행
if __name__ == "__main__":
    query_name = "7wro_8j7t"
    process_query(query_name)
'
