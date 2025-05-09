import pdb_numpy
from pdb_numpy.analysis import dockQ
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 예측 모델과 native 구조 파일 로드
model_pdb = "/home/cseomoon/project/ABAG/AbNb_benchmark/AF3_results/7wog_7wog/seed-1_sample-0/PDBs/model.pdb"  # 예측된 도킹 모델
native_pdb = "/home/cseomoon/project/ABAG/AbNb_benchmark/Native_PDB/renum_fv_7wog_0.pdb"  # 참조 구조(실험적으로 확인된 구조)

# Coor 객체 생성
model_coor = pdb_numpy.Coor(model_pdb)
native_coor = pdb_numpy.Coor(native_pdb)

# 기본 방식으로 호출 (체인 자동 감지)
results = dockQ(model_coor, native_coor)

# 결과 출력
print(f"DockQ 점수: {results['DockQ'][0]:.3f}")
print(f"Fnat: {results['Fnat'][0]:.3f}")
print(f"Fnonnat: {results['Fnonnat'][0]:.3f}")
print(f"리간드 RMSD: {results['LRMS'][0]:.3f} Å")
print(f"인터페이스 RMSD: {results['iRMS'][0]:.3f} Å")
print(f"수용체 RMSD: {results['rRMS'][0]:.3f} Å")

# 체인 ID를 명시적으로 지정하는 방식
# 예: 모델에서 A 체인은 수용체, B 체인은 리간드
results_explicit = dockQ(
    model_coor, 
    native_coor,
    rec_chains=["A"], 
    lig_chains=["H"],
    native_rec_chains=["A"], 
    native_lig_chains=["H"],
    back_atom=["CA", "N", "C", "O"]  # 백본 원자 지정
)

print(f"DockQ 점수: {results_explicit['DockQ'][0]:.3f}")
print(f"Fnat: {results_explicit['Fnat'][0]:.3f}")
print(f"Fnonnat: {results_explicit['Fnonnat'][0]:.3f}")
print(f"리간드 RMSD: {results_explicit['LRMS'][0]:.3f} Å")
print(f"인터페이스 RMSD: {results_explicit['iRMS'][0]:.3f} Å")
print(f"수용체 RMSD: {results_explicit['rRMS'][0]:.3f} Å")

# DockQ 점수 해석
dockq_score = results["DockQ"][0]
if dockq_score > 0.8:
    quality = "높음 (High)"
elif dockq_score > 0.5:
    quality = "중간 (Medium)"
elif dockq_score > 0.23:
    quality = "허용 (Acceptable)"
else:
    quality = "낮음 (Incorrect)"

print(f"도킹 모델 품질: {quality} (DockQ = {dockq_score:.3f})")
