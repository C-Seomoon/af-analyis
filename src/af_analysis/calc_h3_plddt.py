#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import argparse
import pandas as pd
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO
import pyrosetta
from pyrosetta import init, pose_from_pdb
import multiprocessing as mp

# =========================
# 1) PyRosetta 한 번만 초기화
# =========================
if not hasattr(pyrosetta, "_already_init"):
    init("-mute all")
    pyrosetta._already_init = True

# =========================
# 2) CDR-H3 / CDR-L3 pLDDT 추출 함수
# =========================
def CDRH3_Bfactors(pose):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    start = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(
        pose_i1, pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3, pose
    )
    end = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(
        pose_i1, pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3, pose
    )
    return [pose.pdb_info().temperature(i,1) for i in range(start+1, end+1)]

def CDRL3_Bfactors(pose):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    start = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(
        pose_i1, pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l3, pose
    )
    end = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(
        pose_i1, pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l3, pose
    )
    return [pose.pdb_info().temperature(i,1) for i in range(start+1, end+1)]

# =========================
# 3) mmCIF → PDB 변환 함수
# =========================
def convert_cif_to_pdb(cif_path: str, pdb_out: str):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("X", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_out)

# =========================
# 4) 단일 PDB/mmCIF 경로 처리 함수
# =========================
def compute_h3_l3(path: str):
    # .cif 파일이면 임시 PDB로 변환
    is_cif = path.lower().endswith(".cif")
    if is_cif:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.close()
        convert_cif_to_pdb(path, tmp.name)
        pdb_path = tmp.name
    else:
        pdb_path = path

    # PyRosetta로 구조 로드
    pose = pose_from_pdb(pdb_path)

    # pLDDT 값 추출
    h3 = CDRH3_Bfactors(pose)
    l3 = CDRL3_Bfactors(pose)

    # 임시 파일 정리
    if is_cif:
        os.remove(pdb_path)

    # 평균값 반환 (없으면 NaN)
    return (float(np.mean(h3)) if h3 else np.nan,
            float(np.mean(l3)) if l3 else np.nan)

# =========================
# 5) 워커 함수 (Pool.map 용)
# =========================
def worker(args):
    idx, row = args
    # model_path 우선, 없으면 pdb 경로 사용
    path = row.model_path if "model_path" in row.index and pd.notnull(row.model_path) else row.pdb
    return (idx, *compute_h3_l3(path))

# =========================
# 6) 메인 함수
# =========================
def main():
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description="CDR-H3/L3 pLDDT 계산")
    parser.add_argument("-i", "--input",  required=True, help="입력 CSV 파일 경로")
    parser.add_argument("-o", "--output", required=True, help="출력 CSV 파일 경로")
    parser.add_argument("-c", "--cpus",   type=int, default=mp.cpu_count(),
                        help="사용할 CPU 코어 개수 (기본: 모든 코어)")
    args = parser.parse_args()

    # 데이터 불러오기 및 결과 컬럼 초기화
    df = pd.read_csv(args.input)
    df["plddt_H3"] = np.nan
    df["plddt_L3"] = np.nan

    # (index, row) 리스트 생성
    inputs = list(df.iterrows())

    # 멀티프로세싱 실행
    with mp.Pool(processes=args.cpus) as pool:
        results = pool.map(worker, inputs)

    # 결과 기록
    for idx, h3, l3 in results:
        df.loc[idx, "plddt_H3"] = h3
        df.loc[idx, "plddt_L3"] = l3

    # CSV 저장
    df.to_csv(args.output, index=False)
    print(f"완료: {args.output} 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()