#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_rosetta_energy.py
====================
AlphaFold3 모델(또는 기타 PDB/mmCIF 구조)에 대해
PyRosetta **InterfaceAnalyzerMover** 를 실행하고, 항체‑항원 인터페이스품질지표를
DataFrame 형태로 추출·저장하는 모듈.

❚ Why this file? ──────────────────────────────────────────────────────────────
‣ 연구 목표는"서열 쌍‑단위(native vs decoy) 분류"이므로,
  ipTM/pTM 같은 AF3 confidence 외에도 **물리·기하학적 인터페이스 metric**을
  추가해 모델 성능을 높이고 해석력을 확보한다.

❚ 주요 Metric 설명 (InterfaceAnalyzerMover nomenclature) ────────────────
• dG_separated       : 결합ΔG (kcal/mol). 0보다 작을수록 결합 안정.
• dSASA_int          : 인터페이스에서 매몰된 SASA (Å²). 면적이 클수록 접촉이 넓다.
• nres_int           : 인터페이스에 관여하는 잔기 수. 
• delta_unsatHbonds  : 매몰되지만 불만족된(H-bond 수용/공여 불충족) H-bond 수.
• packstat           : 포장 치밀도(0-1). 0.6 ↓ 이면 내부 빈틈이 많음.
• dG_dSASA_norm       : ΔG / dSASA_int. 면적 편차 보정된 에너지.
"""

from __future__ import annotations

import os
import tempfile
import logging
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import PDBIO
from Bio import BiopythonWarning
import warnings
from logging import INFO, DEBUG, WARNING, ERROR

# ───────────────────────────── suppress BioPython warnings
warnings.simplefilter("ignore", BiopythonWarning)

# ───────────────────────────── PyRosetta lazy import
_pyrosetta_initialized = False
try:
    import pyrosetta
    from pyrosetta import init, pose_from_pdb, create_score_function
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
except ImportError as e:
    raise ImportError("PyRosetta가 설치되어 있지 않습니다. 'pip install pyrosetta' 후 재시도하세요.") from e

# ───────────────────────────── Logging setup
def setup_logger(verbosity: int = 0):
    """로깅 레벨 설정: 0=INFO, 1=DEBUG"""
    level = DEBUG if verbosity > 0 else INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # 외부 라이브러리 로그 제한
    logging.getLogger("pyrosetta").setLevel(WARNING)
    logging.getLogger("Bio").setLevel(WARNING)


def initialize_rosetta(silent: bool = True) -> None:
    """글로벌 1회 PyRosetta 초기화."""
    global _pyrosetta_initialized
    if _pyrosetta_initialized:
        return

    opts = "-ignore_unrecognized_res -ignore_zero_occupancy false -load_PDB_components false " \
           "-no_fconfig -check_cdr_chainbreaks false"
    if silent:
        opts += " -mute all"

    init(opts)
    _pyrosetta_initialized = True


# ───────────────────────────── chain helper

def get_chain_ids(file_path: str) -> List[str]:
    """PDB/mmCIF 파일에서 체인 ID 리스트 반환."""
    parser = MMCIFParser(QUIET=True) if file_path.lower().endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure("structure", file_path)
    chains = [c.id.strip() for c in structure.get_chains() if c.id.strip()] or ["A"]
    return chains


# ───────────────────────────── core metric extractor

def _run_interface_analyzer(
    pose, 
    interface_str: str, 
    *, 
    scorefxn_tag: str = "ref2015", 
    pack_sep: bool = True,
    extract_energy_terms: bool = False,
    extract_per_residue: bool = False
) -> Dict[str, float]:
    """InterfaceAnalyzerMover를 적용하고 metric dict 반환."""
    print("[DEBUG] Creating score function...")
    scorefxn = create_score_function(scorefxn_tag)

    print("[DEBUG] Setting up InterfaceAnalyzerMover...")
    iam = InterfaceAnalyzerMover()
    iam.set_scorefunction(scorefxn)
    iam.set_interface(interface_str)
    iam.set_pack_input(True)
    iam.set_pack_separated(pack_sep)
    iam.set_compute_interface_sc(True)
    iam.set_calc_dSASA(True)
    iam.set_compute_packstat(True)

    print("[DEBUG] Applying InterfaceAnalyzerMover...")
    iam.apply(pose)

    print("[DEBUG] Extracting basic metrics...")
    
    # 확인된 기본 메트릭들
    dG = iam.get_interface_dG()
    dSASA = iam.get_interface_delta_sasa()
    packstat = iam.get_interface_packstat()
    nres_int = iam.get_num_interface_residues()
    unsat_hbonds = iam.get_interface_delta_hbond_unsat()
    
    # 안전한 계산
    dSASA_safe = max(dSASA, 1e-3) if dSASA > 0 else np.nan
    
    print("[DEBUG] Creating metrics dictionary...")
    metrics = {
        "dG_separated": dG,
        "dSASA_int": dSASA,
        "nres_int": nres_int,
        "delta_unsatHbonds": unsat_hbonds,
        "packstat": packstat,
        "dG_dSASA_norm": dG / dSASA_safe if not np.isnan(dSASA_safe) else np.nan,
    }
    
    # 선택적 메트릭들
    if extract_energy_terms:
        print("[DEBUG] Extracting energy terms...")
        try:
            from pyrosetta.rosetta.core.scoring import fa_elec, fa_atr, fa_rep, fa_sol
            total_energies = pose.energies().total_energies()
            
            metrics.update({
                "fa_elec": total_energies[fa_elec],
                "fa_atr": total_energies[fa_atr],
                "fa_rep": total_energies[fa_rep], 
                "fa_sol": total_energies[fa_sol],
            })
        except Exception as e:
            print(f"[WARNING] Could not extract energy terms: {e}")
    
    if extract_per_residue:
        print("[DEBUG] Extracting per-residue energies...")
        try:
            per_res_energies = iam.get_per_residue_energy()
            if per_res_energies:
                energies_list = list(per_res_energies.values())
                metrics.update({
                    "per_res_energy_mean": np.mean(energies_list),
                    "per_res_energy_std": np.std(energies_list),
                    "per_res_energy_min": min(energies_list),
                    "per_res_energy_max": max(energies_list),
                })
        except Exception as e:
            print(f"[WARNING] Could not extract per-residue energies: {e}")
    
    # 데이터 품질 검증
    critical_metrics = [dG, dSASA, nres_int]
    if any(np.isnan(m) or m is None for m in critical_metrics):
        print(f"[WARNING] Some critical metrics are invalid: dG={dG}, dSASA={dSASA}, nres_int={nres_int}")
    
    print("[DEBUG] Metrics extraction completed")
    return metrics


def extract_metrics_from_file(
        pdb_path: str,
        *,
        antibody_mode: bool = True,
        receptor_chains: List[str] | None = None,
        ligand_chains: List[str] | None = None,
) -> Tuple[Dict[str, float] | None, str | None]:
    """단일 PDB/mmCIF → metric dict.
    반환: (metric_dict, error_msg). 실패 시 metric_dict = None.
    """
    print(f"\n[DEBUG] Processing file: {pdb_path}")
    initialize_rosetta(silent=True)

    temp_pdb = None  # mmCIF → 임시 PDB 변환 경로
    try:
        # ─── mmCIF는 임시 PDB로 변환 (PyRosetta CIF parser 사용 안함)
        working = pdb_path
        if pdb_path.lower().endswith(".cif"):
            print(f"[DEBUG] Converting mmCIF to PDB...")
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("tmp", pdb_path)
            fd, temp_pdb = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_pdb)
            working = temp_pdb
            print(f"[DEBUG] Temporary PDB created at: {temp_pdb}")

        print(f"[DEBUG] Reading chains from: {working}")
        chains = get_chain_ids(working)
        print(f"[DEBUG] Found chains: {chains}")
        
        if len(chains) < 2:
            return None, f"Less than two chains in {pdb_path}"

        # ─── 인터페이스 체인 정의
        if antibody_mode:
            ab = [c for c in chains if c in {"H", "L"}]
            ag = [c for c in chains if c not in {"H", "L"}]
            print(f"[DEBUG] Antibody chains: {ab}")
            print(f"[DEBUG] Antigen chains: {ag}")
            
            if not (ab and ag):
                return None, "Antibody or antigen chains missing"
            interface_str = f"{''.join(ab)}_{''.join(ag)}"
        else:
            receptor_chains = receptor_chains or [chains[0]]
            ligand_chains   = ligand_chains   or [chains[1]]
            interface_str   = f"{''.join(receptor_chains)}_{''.join(ligand_chains)}"
        
        print(f"[DEBUG] Interface string: {interface_str}")

        print("[DEBUG] Loading pose from PDB...")
        pose = pose_from_pdb(working)
        print("[DEBUG] Running interface analyzer...")
        metrics = _run_interface_analyzer(pose, interface_str)
        print(f"[DEBUG] Extracted metrics: {metrics}")

        # 추가 metadata (파일·체인 정보)
        metrics.update({
            "pdb": os.path.abspath(pdb_path),
        })
        print("[DEBUG] Processing completed successfully")
        return metrics, None

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return None, str(e)

    finally:
        if temp_pdb and os.path.exists(temp_pdb):
            print(f"[DEBUG] Cleaning up temporary file: {temp_pdb}")
            os.remove(temp_pdb)


# ───────────────────────────── batch API

def batch_extract(
        pdb_files: List[str],
        *,
        n_jobs: int = 4,
        verbose: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """다수 구조 파일 → metric DataFrame"""
    pdb_files = list(pdb_files)
    if not pdb_files:
        raise ValueError("No PDB/mmCIF files supplied")

    worker = partial(extract_metrics_from_file, **kwargs)

    if n_jobs > 1 and len(pdb_files) > 1:
        n_jobs = min(n_jobs, mp.cpu_count(), len(pdb_files))
        ctx = mp.get_context("fork")  # linux
        with ctx.Pool(n_jobs) as pool:
            it = pool.imap(worker, pdb_files, chunksize= max(1, len(pdb_files)//(n_jobs*2)))
            if verbose:
                it = tqdm(it, total=len(pdb_files), desc="Interface metrics")
            rows = [r for r, _ in it if r]
    else:
        it = pdb_files
        if verbose:
            it = tqdm(it, desc="Interface metrics")
        rows = []
        for f in it:
            r, _ = worker(f)
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes("number").columns
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


# ───────────────────────────── script entry
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract interface metrics with PyRosetta")
    parser.add_argument("inputs", nargs="+", help="PDB/mmCIF files or directories")
    parser.add_argument("--out", required=True, help="Output CSV/Parquet path (auto by ext)")
    parser.add_argument("--jobs", type=int, default=4, help="CPU processes")
    parser.add_argument("--no-antibody-mode", action="store_true", help="Disable automatic H/L detection")
    args = parser.parse_args()

    # expand dir → file list
    pdb_files: List[str] = []
    for path in args.inputs:
        if os.path.isdir(path):
            pdb_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith((".pdb", ".cif"))])
        else:
            pdb_files.append(path)

    df = batch_extract(
        pdb_files,
        n_jobs=args.jobs,
        verbose=True,
        antibody_mode= not args.no_antibody_mode,
        
    )

    if args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"[✓] Saved {len(df)} rows → {args.out}")
