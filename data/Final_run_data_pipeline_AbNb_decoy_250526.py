#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import af_analysis
from af_analysis import analysis
from af_analysis.data import Data

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 로깅 레벨 설정
logging.getLogger('pdb_numpy').setLevel(logging.WARNING)
logging.getLogger('pdb_numpy.coor').setLevel(logging.WARNING)
logging.getLogger('pdb_numpy.analysis').setLevel(logging.WARNING)

# 기본 경로 설정
BASE_PATH = '/home/cseomoon/project/ABAG/DB/AbNb_structure/AF3/decoy'
NATIVE_DIR = '/home/cseomoon/project/ABAG/DB/AbNb_structure/original_pdb'
OUTPUT_DIR = 'pipeline_results_AbNb_decoy'

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_slurm_script(query_name, output_dir):
    """Slurm 작업 스크립트 생성"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=af_{query_name}
#SBATCH --output={output_dir}/slurm_{query_name}_%j.log
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
    print(f"🔄 Starting: {{description}}")
    yield
    elapsed = time.time() - start
    print(f"✅ Completed: {{description}} ({{elapsed:.2f}}s)")

def calculate_all_metrics_per_query_optimized(query_group_data, **kwargs):
    query_name, query_df = query_group_data
    
    print(f"\\n🚀 Processing query: {{query_name}} ({{len(query_df)}} models)")
    total_start = time.time()
    
    try:
        # Data 객체 생성
        with time_tracker("Data object creation"):
            data_dict = {{}}
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
        print(f"✨ Query {{query_name}} completed in {{total_time:.2f}}s")
        
        return query_name, temp_data.df
        
    except Exception as e:
        total_time = time.time() - total_start
        print(f"❌ Error processing query {{query_name}} after {{total_time:.2f}}s: {{str(e)}}")
        import traceback
        print(traceback.format_exc())
        return query_name, None

# 쿼리 처리 함수
def process_query(query_name):
    base_path = "{BASE_PATH}"
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
                native_dir="{NATIVE_DIR}"
            )
            
            if result_df is not None:
                output_file = os.path.join("{output_dir}", f"metrics_{{name}}.csv")
                result_df.to_csv(output_file, index=False)
                print(f"Successfully processed {{query_name}}")
            else:
                print(f"Failed to process {{query_name}}")
                
    except Exception as e:
        print(f"Error processing {{query_name}}: {{str(e)}}")
        return False
    
    return True

# 메인 실행
if __name__ == "__main__":
    query_name = "{query_name}"
    process_query(query_name)
'
"""
    script_path = os.path.join(output_dir, f"run_{query_name}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def submit_jobs(queries, max_concurrent=50):
    """Slurm 작업 제출 및 모니터링"""
    def get_available_cpu_nodes():
        """사용 가능한 CPU 노드 수 확인"""
        result = subprocess.run(['sinfo', '-p', 'skylake_normal,rome_normal', '-h', '-o', '%C'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            # 사용 가능한 CPU 수 파싱
            available = result.stdout.strip()
            return int(available.split('/')[1])  # 전체 CPU 수
        return 0

    running_jobs = set()
    completed_jobs = set()
    failed_jobs = set()
    
    for query in queries:
        # 사용 가능한 CPU 노드 확인
        available_cpus = get_available_cpu_nodes()
        if available_cpus < 4:  # 각 작업이 4개의 CPU를 사용
            logger.warning(f"Not enough CPU resources available. Waiting...")
            time.sleep(60)  # 1분 대기
            continue

        # 실행 중인 작업이 최대 개수에 도달하면 대기
        while len(running_jobs) >= max_concurrent:
            # 작업 상태 확인
            for job_id in list(running_jobs):
                result = subprocess.run(['squeue', '-j', str(job_id)], 
                                     capture_output=True, text=True)
                if str(job_id) not in result.stdout:
                    running_jobs.remove(job_id)
                    # 결과 파일 확인
                    output_file = os.path.join(OUTPUT_DIR, f"metrics_{query}.csv")
                    if os.path.exists(output_file):
                        completed_jobs.add(query)
                    else:
                        failed_jobs.add(query)
            time.sleep(10)
        
        # 새 작업 제출
        script_path = create_slurm_script(query, OUTPUT_DIR)
        try:
            # 스크립트 실행 권한 부여
            os.chmod(script_path, 0o755)
            
            # 작업 제출
            result = subprocess.run(['sbatch', script_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = result.stdout.split()[-1]
                running_jobs.add(job_id)
                logger.info(f"Submitted job {job_id} for query {query}")
            else:
                logger.error(f"Failed to submit job for query {query}")
                logger.error(f"Error output: {result.stderr}")
                logger.error(f"Command output: {result.stdout}")
                failed_jobs.add(query)
                
        except Exception as e:
            logger.error(f"Exception while submitting job for query {query}: {str(e)}")
            failed_jobs.add(query)
    
    return completed_jobs, failed_jobs

def main():
    """메인 실행 함수"""
    # 쿼리 목록 가져오기
    queries = [d for d in os.listdir(BASE_PATH) 
              if os.path.isdir(os.path.join(BASE_PATH, d))]
    
    logger.info(f"Found {len(queries)} queries to process")
    
    # 작업 제출 및 모니터링
    completed, failed = submit_jobs(queries)
    
    # 결과 요약
    logger.info(f"Processing completed:")
    logger.info(f"Successfully processed: {len(completed)} queries")
    logger.info(f"Failed to process: {len(failed)} queries")
    
    if failed:
        logger.info("Failed queries:")
        for query in failed:
            logger.info(f"- {query}")

if __name__ == "__main__":
    main()
