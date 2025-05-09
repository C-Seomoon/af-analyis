import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path, 
                             target_column='DockQ', 
                             threshold=0.23, 
                             query_id_column='query', # Legacy code used 'query'
                             default_features_to_drop=None, # Added default drop list
                             user_features_to_drop=None):   # User can specify additional drops
    """
    데이터 파일을 로드하고 기본적인 전처리를 수행합니다.
    NaN 값을 포함하는 행을 특성 컬럼 기준으로 제거합니다.

    Args:
        file_path (str): 입력 데이터 CSV 파일 경로.
        target_column (str, optional): 목표 변수를 생성할 기준 컬럼 이름. Defaults to 'DockQ'.
        threshold (float, optional): 목표 변수 생성을 위한 임계값. Defaults to 0.23.
        query_id_column (str, optional): 그룹 교차 검증에 사용할 그룹 ID 컬럼 이름. 
                                           Defaults to 'query' based on legacy code.
        default_features_to_drop (list, optional): 기본적으로 제외할 특성 목록. 
                                                   Defaults to list from legacy code.
        user_features_to_drop (list, optional): 사용자가 추가로 제외할 특성 목록. 
                                                Defaults to None.

    Returns:
        tuple: 다음을 포함하는 튜플:
               - X (pd.DataFrame): 모델 학습에 사용될 특성 데이터 (NaN 제거됨, 스케일링 전).
               - y (pd.Series): 생성된 이진 목표 변수.
               - query_ids (pd.Series or None): 그룹 ID. 해당 컬럼이 없으면 None.
               
    Raises:
        FileNotFoundError: file_path에 해당하는 파일이 없을 경우.
        ValueError: target_column이 데이터에 존재하지 않거나, 처리 후 X가 비어있을 경우.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Input file not found at {file_path}")

    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")

    # 목표 변수 생성 (NaN 처리 전에 수행하여 대상 행 유지)
    if target_column not in data.columns:
        raise ValueError(f"Error: Target column '{target_column}' not found in the data.")
    target_temp_name = f"__target__{target_column}" 
    data[target_temp_name] = (data[target_column] >= threshold).astype(int)
    print(f"Class distribution before NaN drop ({target_column} >= {threshold}): "
          f"0 (Negative) = {(data[target_temp_name]==0).sum()}, "
          f"1 (Positive) = {(data[target_temp_name]==1).sum()}")
    
    # 그룹 ID 추출 (NaN 처리 전에 수행하여 대상 행 유지)
    if query_id_column in data.columns:
        query_ids_temp = data[query_id_column].copy() # 임시 저장
    else:
        query_ids_temp = None
        print(f"Warning: Query ID column '{query_id_column}' not found. GroupKFold cannot be used.")

    # 제외할 컬럼 목록 구성
    cols_to_drop = set([target_column]) # Start with the original target column
    if query_id_column in data.columns:
        cols_to_drop.add(query_id_column)
        
    if default_features_to_drop is None:
        default_features_to_drop = [
            'pdb', 'seed', 'sample', 'data_file', 'chain_iptm', 'chain_pair_iptm', 
            'chain_pair_pae_min', 'chain_ptm', 'format', 'model_path', 'native_path',
            'Fnat', 'Fnonnat', 'rRMS', 'iRMS', 'LRMS'
        ]
        
    valid_default_drops = [col for col in default_features_to_drop if col in data.columns]
    cols_to_drop.update(valid_default_drops)
    
    if user_features_to_drop:
        valid_user_drops = [col for col in user_features_to_drop if col in data.columns]
        cols_to_drop.update(valid_user_drops)

    # 최종 특성으로 사용될 컬럼 식별 (임시 타겟 컬럼 제외)
    potential_feature_cols = [col for col in data.columns if col not in cols_to_drop and col != target_temp_name]
    print(f"Identified {len(potential_feature_cols)} potential feature columns.")
    if not potential_feature_cols:
         raise ValueError("Error: No potential feature columns identified after configuring drops.")

    # --- NaN 값 포함 행 제거 (특성 컬럼 기준) ---
    print(f"Checking for NaN values in potential feature columns...")
    original_rows = len(data)
    # subset 인자에 실제 특성으로 사용될 컬럼 목록을 전달
    data.dropna(subset=potential_feature_cols, inplace=True) 
    rows_dropped = original_rows - len(data)
    
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NaN values in one or more feature columns.")
    else:
        print("No rows dropped due to NaN values in features.")

    # NaN 제거 후 최종 X, y, query_ids 생성
    final_cols_to_drop = list(cols_to_drop) + [target_temp_name]
    X = data.drop(columns=final_cols_to_drop, errors='ignore') 
    y = data[target_temp_name]
    
    # query_ids도 NaN 제거된 데이터의 인덱스에 맞춰 정렬
    if query_ids_temp is not None:
        query_ids = query_ids_temp.loc[X.index] 
    else:
        query_ids = None

    print(f"Processed Features (X) shape after NaN drop: {X.shape}")
    print(f"Processed Target (y) shape after NaN drop: {y.shape}")
    if query_ids is not None:
        print(f"Processed Query IDs shape after NaN drop: {query_ids.shape}")
        # 최종 query_ids NaN 확인 (이론상 없어야 함)
        if query_ids.isnull().any():
             print(f"Warning: Query ID column still contains NaN after row dropping. Filling with placeholder.")
             query_ids = query_ids.fillna('NaN_Placeholder_Group')
             
    # 최종 클래스 분포 확인
    print(f"Class distribution after NaN drop ({target_column} >= {threshold}): "
          f"0 (Negative) = {(y==0).sum()}, "
          f"1 (Positive) = {(y==1).sum()}")

    # Check for empty features dataframe
    if X.empty or X.shape[1] == 0:
        raise ValueError("Error: Feature set (X) is empty after dropping columns and NaN rows. Check configuration.")
        
    # 최종 NaN 확인 (X에 NaN이 없어야 함)
    if X.isnull().sum().sum() > 0:
        print("Error: Final feature set X still contains NaN values after dropping rows!")
        print(X.isnull().sum()[X.isnull().sum() > 0])

    return X, y, query_ids
