import af_analysis
import multiprocessing as mp

if __name__ == '__main__':
    # 데이터 로드
    my_data = af_analysis.Data("/home/cseomoon/project/ABAG/2025_H_L_A/20250504_seeds_10/af3_results/6x97")
    
    # 결합 에너지 계산 (기본 설정: 항체-항원 모드)
    my_data.calculate_binding_energy(n_jobs=24)  # 먼저 단일 프로세스로 테스트
    
    # 결과 확인
    print(my_data.df[['pdb', 'del_G_B']])
    
    # 결과 저장
    my_data.export_file("results_with_binding_energy.csv")

# # 또는 사용자 지정 체인으로 계산
# my_data.calculate_binding_energy(
#     receptor_chains=['A'], 
#     ligand_chains=['B'], 
#     antibody_mode=False
# )