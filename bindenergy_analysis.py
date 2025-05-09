#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from /home/cseomoon/appl/AF3_AbNb_Benchmark/scripts/bindenergy_with_datastruc.py
for use in af_analysis with parallelization for SLURM
"""

from biopandas.pdb import PandasPdb
import scipy as sp
import pandas as pd
import os
import numpy as np
import Bio
try:
    from abnumber import Chain
except ImportError:
    print("abnumber module not found. Some functionality might be limited.")
import math
import enum
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities
try:
    from Bio.SeqIO import PdbIO
except ImportError:
    print("Bio.SeqIO.PdbIO not found. Some functionality might be limited.")
import warnings
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Optional, Union
from Levenshtein import distance, ratio
try:
    from Bio.Align.Applications import ClustalwCommandline
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import AlignIO
except ImportError:
    print("Some Bio modules not found. Alignment functionality might be limited.")
import argparse
import pyrosetta
from pyrosetta import init, pose_from_pdb, create_score_function
from pyrosetta.rosetta.core.scoring import get_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from Bio import BiopythonWarning
import multiprocessing as mp
from functools import partial
import time
import sys

warnings.simplefilter('ignore', BiopythonWarning)

init('-ignore_unrecognized_res \
     -ignore_zero_occupancy false -load_PDB_components false \
     -no_fconfig -check_cdr_chainbreaks false \
     -mute all')

# PDB 파일에서 체인 정보 추출 함수
def get_atmseq(pdb_file):
    """Extract chain IDs from a PDB file"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", pdb_file)
        chains = [chain.id for chain in structure.get_chains()]
        return chains
    except Exception as e:
        print(f"Error reading PDB file {pdb_file}: {e}")
        return []

def get_interface_analyzer(partner_chain_str, scorefxn, pack_sep=False):
    interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
    interface_analyzer.fresh_instance()
    interface_analyzer.set_pack_input(True)
    interface_analyzer.set_interface(partner_chain_str)
    interface_analyzer.set_scorefunction(scorefxn)
    interface_analyzer.set_compute_interface_energy(True)
    interface_analyzer.set_compute_interface_sc(True)
    interface_analyzer.set_calc_dSASA(True)
    interface_analyzer.set_pack_separated(pack_sep)

    return interface_analyzer

def interface_energy_calc(pred_pdb, pack_scorefxn):
    pred_pdb_Ag = [x for x in list(get_atmseq(pred_pdb)) if x not in ['H','L']]
    if not pred_pdb_Ag:
        return None, f"No antigen chains found in {pred_pdb}"
    
    try:
        pred_pose = pose_from_pdb(pred_pdb)
        pred_interface = f"HL_{pred_pdb_Ag}"
    
        interface_analyzer = get_interface_analyzer(pred_interface, pack_scorefxn)
        interface_analyzer.apply(pred_pose)
        interface_analyzer_packsep = get_interface_analyzer(pred_interface, pack_scorefxn, pack_sep=True)
        interface_analyzer_packsep.apply(pred_pose)
        binding_energy_dgsep = interface_analyzer_packsep.get_interface_dG()
        return binding_energy_dgsep, None
    except Exception as e:
        return None, str(e)

def process_model(path):
    """병렬 처리를 위한 단일 모델 처리 함수"""
    if not os.path.exists(path):
        return path, None, f"File not found: {path}"
    
    try:
        # 각 프로세스 내에서 ScoreFunction 생성
        pack_scorefxn = create_score_function("ref2015")
        b_E, error = interface_energy_calc(path, pack_scorefxn)
        return path, b_E, error
    except Exception as e:
        return path, None, str(e)

def chunk_list(lst, n):
    """리스트를 n개의 청크로 분할"""
    chunk_size = max(1, len(lst) // n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datafilepath", help="datastructure filepath")
    parser.add_argument("resultsfilepath", help="filepath to dump results")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of parallel workers. 0 for auto-detection (default: 0)")
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Number of models to process per worker (default: 1)")
    parser.add_argument("--slurm_array_task_id", type=int, default=None,
                        help="SLURM array task ID for processing a subset of data")
    parser.add_argument("--slurm_array_task_count", type=int, default=None,
                        help="Total number of SLURM array tasks")
    args = parser.parse_args()

    # 결과 디렉토리 확인 및 생성
    results_dir = os.path.dirname(args.resultsfilepath)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created output directory: {results_dir}")

    # 데이터 로드
    if not os.path.exists(args.datafilepath):
        print(f"Error: Input file {args.datafilepath} does not exist.")
        return
    
    print(f"Loading data from {args.datafilepath}")
    datastructure = pd.read_csv(args.datafilepath)
    print(f"Loaded {datastructure.shape[0]} entries")

    # SLURM 배열 작업에 대한 데이터 분할
    if args.slurm_array_task_id is not None and args.slurm_array_task_count is not None:
        task_id = args.slurm_array_task_id
        task_count = args.slurm_array_task_count
        
        print(f"SLURM Array Task: {task_id+1}/{task_count}")
        
        # 데이터 분할
        n_per_task = len(datastructure) // task_count
        start_idx = task_id * n_per_task
        end_idx = start_idx + n_per_task if task_id < task_count - 1 else len(datastructure)
        
        datastructure = datastructure.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"Processing {len(datastructure)} entries (indices {start_idx} to {end_idx-1})")

    # 'model_path' 열 확인
    if 'model_path' not in datastructure.columns:
        print(f"Error: 'model_path' column not found in datafile.")
        print(f"Available columns: {datastructure.columns.tolist()}")
        return

    # 에너지 계산 초기화
    model_paths = datastructure['model_path'].tolist()
    
    # 워커 수 설정
    if args.num_workers <= 0:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(args.num_workers, mp.cpu_count())
    
    print(f"Using {num_workers} workers for parallel processing")
    
    # 병렬 처리 설정
    results = []
    
    if num_workers > 1:
        # 병렬 처리
        print(f"Starting parallel processing with {num_workers} workers...")
        
        # 작업자 풀 생성
        with mp.Pool(processes=num_workers) as pool:
            # ScoreFunction을 프로세스 내부로 이동했으므로 process_func는 수정 필요 없음
            
            # 청크 크기 조정을 통한 오버헤드 감소
            chunk_size = max(1, min(args.chunk_size, len(model_paths) // (num_workers * 2)))
            
            # 병렬 처리 실행 및 결과 수집
            for result in tqdm(pool.imap(process_model, model_paths, chunksize=chunk_size), 
                             total=len(model_paths), desc="Processing models"):
                results.append(result)
    else:
        # 순차 처리
        print("Running in sequential mode...")
        # 단일 프로세스에서는 ScoreFunction을 한 번만 생성
        pack_scorefxn = create_score_function("ref2015")
        for path in tqdm(model_paths, desc="Processing models"):
            try:
                b_E, error = interface_energy_calc(path, pack_scorefxn)
                results.append((path, b_E, error))
            except Exception as e:
                results.append((path, None, str(e)))

    # 결과 처리
    valid_results = []
    errors = []
    
    for path, b_E, error in results:
        if b_E is not None:
            valid_results.append((path, b_E))
        if error:
            errors.append((path, error))
    
    # 결과 저장
    pdbs, bind_es = zip(*valid_results) if valid_results else ([], [])
    
    binding_es = pd.DataFrame({"model_path": pdbs, "del_G_B": bind_es})
    
    # SLURM 배열 작업에 대한 결과 파일 이름 조정
    if args.slurm_array_task_id is not None:
        base, ext = os.path.splitext(args.resultsfilepath)
        output_path = f"{base}_task{args.slurm_array_task_id}{ext}"
    else:
        output_path = args.resultsfilepath
    
    binding_es.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # 오류 보고서 저장
    if errors:
        error_path = f"{os.path.splitext(output_path)[0]}_errors.csv"
        pd.DataFrame({"model_path": [e[0] for e in errors], "error": [e[1] for e in errors]}).to_csv(error_path, index=False)
        print(f"Errors saved to {error_path}")
    
    total_time = time.time() - start_time
    print(f"Processed {len(valid_results)} models successfully out of {len(model_paths)} total.")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per model: {total_time/len(model_paths):.2f} seconds")

if __name__ == "__main__":
    main() 