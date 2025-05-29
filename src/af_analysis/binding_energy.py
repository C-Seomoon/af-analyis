#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binding energy calculation module for AF-Analysis
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser
import warnings
from Bio import BiopythonWarning
from tqdm.auto import tqdm
import logging
import tempfile

# PyRosetta 임포트
try:
    import pyrosetta
    from pyrosetta import init, pose_from_pdb, create_score_function
    from pyrosetta.rosetta.core.scoring import get_score_function
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    
    # PyRosetta 초기화
    _pyrosetta_initialized = False
    
    def initialize_pyrosetta(silent=True):
        """PyRosetta 초기화 (필요시에만 한 번 실행)"""
        global _pyrosetta_initialized
        if not _pyrosetta_initialized:
            init_options = '-ignore_unrecognized_res -ignore_zero_occupancy false -load_PDB_components false -no_fconfig -check_cdr_chainbreaks false'
            if silent:
                init_options += ' -mute all'
            init(init_options)
            _pyrosetta_initialized = True
            return True
        return False
        
except ImportError:
    def initialize_pyrosetta(silent=True):
        """PyRosetta가 설치되지 않은 경우 경고 출력"""
        logging.error("PyRosetta is not installed. Binding energy calculations will not be available.")
        return False

warnings.simplefilter('ignore', BiopythonWarning)

# PDB 또는 mmCIF 파일에서 체인 정보 추출 함수
def get_chain_ids(file_path):
    """Extract chain IDs from a PDB or mmCIF file"""
    # 파일 확장자에 따라 적절한 파서 선택
    if file_path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure("structure", file_path)
        chains = [chain.id for chain in structure.get_chains()]
        
        # 체인이 없는 경우 체인 ID가 빈 문자열일 수 있음
        chains = [c for c in chains if c.strip()]
        
        if not chains:
            # 체인 ID가 없는 경우 기본 체인 'A'를 제안
            chains = ['A']
            print(f"Warning: No explicit chain IDs found in {file_path}. Assuming default chain 'A'.")
            
        return chains
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return []

def get_interface_analyzer(partner_chain_str, scorefxn, pack_sep=False):
    """InterfaceAnalyzerMover 설정"""
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

def compute_single_binding_energy(pdb_file, receptor_chains=None, ligand_chains=None, antibody_mode=True):
    """
    단일 PDB/CIF 파일의 결합 에너지 계산
    
    Parameters
    ----------
    pdb_file : str
        PDB 또는 mmCIF 파일 경로
    receptor_chains : list, optional
        수용체 체인 ID 목록 (지정하지 않으면 자동 감지)
    ligand_chains : list, optional
        리간드 체인 ID 목록 (지정하지 않으면 자동 감지)
    antibody_mode : bool, optional
        항체 모드 사용 여부 (True면 H,L 체인을 항체로 간주)
        
    Returns
    -------
    float or None
        결합 에너지 (kcal/mol) 또는 계산 실패 시 None
    str or None
        오류 메시지 (성공 시 None)
    """
    # PyRosetta 초기화 확인
    if not _pyrosetta_initialized:
        if not initialize_pyrosetta():
            return None, "PyRosetta is not installed"
    
    temp_pdb = None
    
    try:
        # mmCIF 파일을 PDB로 변환 (필요한 경우) - tempfile 모듈 사용으로 통일
        working_file = pdb_file
        if pdb_file.lower().endswith('.cif'):
            try:
                from Bio.PDB import PDBIO
                
                # tempfile 모듈을 사용한 안전한 임시 파일 생성
                temp_fd, temp_pdb = tempfile.mkstemp(suffix='.pdb', prefix='af_analysis_')
                os.close(temp_fd)
                
                # mmCIF 파일 파싱
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("temp", pdb_file)
                
                # PDB 파일로 저장
                io = PDBIO()
                io.set_structure(structure)
                io.save(temp_pdb)
                
                # 실제 처리에 사용할 파일을 임시 PDB 파일로 변경
                working_file = temp_pdb
                logging.info(f"Converted {pdb_file} to temporary PDB format at {temp_pdb}")
            except Exception as e:
                return None, f"Error converting mmCIF to PDB: {str(e)}"
        
        # 자동 체인 감지
        all_chains = get_chain_ids(working_file)
        if not all_chains:
            return None, f"No chains found in {pdb_file}"
            
        if antibody_mode:
            # 항체 모드: H,L은 항체, 나머지는 항원으로 처리
            antibody_chains = [c for c in all_chains if c in ['H', 'L']]
            antigen_chains = [c for c in all_chains if c not in ['H', 'L']]
            
            if not antibody_chains:
                return None, f"No antibody chains (H,L) found in {pdb_file}"
            if not antigen_chains:
                return None, f"No antigen chains found in {pdb_file}"
                
            # 인터페이스 문자열 생성 ("HL_A" 또는 "HL_ABC" 형식)
            interface_str = f"{''.join(antibody_chains)}_{''.join(antigen_chains)}"
        else:
            # 일반 모드: 사용자 지정 체인 사용
            if not receptor_chains or not ligand_chains:
                if len(all_chains) < 2:
                    return None, f"Not enough chains in {pdb_file} for interface analysis"
                
                # 체인이 지정되지 않은 경우 자동 할당
                if not receptor_chains:
                    receptor_chains = [all_chains[0]]
                if not ligand_chains:
                    ligand_chains = [all_chains[1]]
            
            # 인터페이스 문자열 생성
            interface_str = f"{''.join(receptor_chains)}_{''.join(ligand_chains)}"
        
        # PDB 파일 로드
        pose = pose_from_pdb(working_file)
        
        # 점수 함수 생성
        scorefxn = create_score_function("ref2015")
        
        # 인터페이스 분석
        interface_analyzer = get_interface_analyzer(interface_str, scorefxn, pack_sep=True)
        interface_analyzer.apply(pose)
        
        # 결합 에너지 추출
        binding_energy = interface_analyzer.get_interface_dG()
        
        return binding_energy, None
        
    except Exception as e:
        return None, str(e)
    finally:
        # tempfile을 사용한 안전한 임시 파일 정리
        if temp_pdb and os.path.exists(temp_pdb):
            try:
                os.remove(temp_pdb)
                logging.debug(f"Cleaned up temporary file: {temp_pdb}")
            except Exception as cleanup_error:
                logging.warning(f"Failed to cleanup temporary file {temp_pdb}: {cleanup_error}")

def calculate_binding_energies(pdb_files, verbose=False, **kwargs):
    """
    여러 PDB 파일의 결합 에너지 순차 계산
    """
    # PyRosetta 초기화
    initialize_pyrosetta(silent=True)
    
    # 결과 저장용 변수
    results = {}
    errors = {}
    
    # 순차 처리
    iterator = pdb_files
    if verbose:
        iterator = tqdm(iterator, desc="Calculating binding energies")
        
    for pdb_file in iterator:
        energy, error = compute_single_binding_energy(pdb_file, **kwargs)
        if energy is not None:
            results[pdb_file] = energy
        if error:
            errors[pdb_file] = error
    
    return results, errors
