# 인터페이스 에너지 계산 도구

이 도구는 단백질 복합체 구조의 인터페이스 에너지를 계산합니다. 원본 코드는 `/home/cseomoon/appl/AF3_AbNb_Benchmark/scripts/bindenergy_with_datastruc.py`에서 가져와 수정되었습니다.

## 설치 방법

필요한 패키지를 설치합니다:

```bash
pip install -r requirements_bindenergy.txt
```

### PyRosetta 설치

이 도구는 PyRosetta가 필요합니다. PyRosetta는 학술 라이센스가 필요하며, [PyRosetta 웹사이트](https://www.pyrosetta.org/downloads)에서 다운로드할 수 있습니다.

### abnumber 패키지 설치

항체 번호 매기기를 위한 abnumber 패키지가 필요한 경우:

```bash
pip install abnumber
```

## 사용 방법

1. 입력 CSV 파일 준비 - CSV 파일에는 다음 열이 포함되어야 합니다:
   - `model_path`: 분석할 PDB 파일의 경로

2. 스크립트 실행:

```bash
python bindenergy_analysis.py <입력_CSV_파일> <결과_CSV_파일>
```

예시:
```bash
python bindenergy_analysis.py data/input_models.csv results/binding_energies.csv
```

## 출력 결과

결과 CSV 파일에는 다음 열이 포함됩니다:
- `model_path`: 입력 PDB 파일 경로
- `del_G_B`: 계산된 결합 에너지 값 (kcal/mol)

## 주의사항

- PDB 파일에는 항체 체인과 항원 체인이 포함되어야 합니다.
- 항체 체인은 'H'와 'L' 식별자를 사용해야 합니다.
- 출력 디렉토리가 없을 경우 자동으로 생성됩니다.

## 오류 해결

원래 오류 메시지: `FileNotFoundError: [Errno 2] No such file or directory: '/home/cseomoon/appl/chai-1/output_1'`

이 스크립트는 결과 파일의 디렉토리를 자동으로 생성하도록 수정되었습니다. 