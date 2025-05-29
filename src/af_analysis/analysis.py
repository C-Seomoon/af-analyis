import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

import pdb_numpy

# Autorship information
__author__ = "Alaa Reguei"
__copyright__ = "Copyright 2023, RPBS"
__credits__ = ["Samuel Murail", "Alaa Reguei"]
__license__ = "GNU General Public License version 2"
__version__ = "0.1.4"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"


def get_pae(data_file):
    """Get the PAE matrix from a json/npz file.

    Parameters
    ----------
    data_file : str
        Path to the json/npz file.

    Returns
    -------
    np.array
        PAE matrix.
    """

    if data_file is None:
        return None

    if data_file.endswith(".json"):
        return extract_pae_json(data_file)
    elif data_file.endswith(".npz"):
        return extract_pae_npz(data_file)
    else:
        raise ValueError("Unknown file format.")


def extract_pae_json(json_file):
    """Get the PAE matrix from a json file.

    Parameters
    ----------
    json_file : str
        Path to the json file.

    Returns
    -------
    np.array
        PAE matrix.
    """

    with open(json_file) as f:
        local_json = json.load(f)

    if "pae" in local_json:
        pae_array = np.array(local_json["pae"])
    elif "predicted_aligned_error" in local_json[0]:
        pae_array = np.array(local_json[0]["predicted_aligned_error"])
    else:
        raise ValueError("No PAE found in the json file.")

    return pae_array


def extract_pae_npz(npz_file):
    """Get the PAE matrix from a json file.

    Parameters
    ----------
    npz_file : str
        Path to the npz file.

    Returns
    -------
    np.array
        PAE matrix.
    """

    data_npz = np.load(npz_file)
    pae_array = data_npz["pae"]

    return pae_array


def extract_fields_file(data_file, fields):
    """Get the PAE matrix from a json/pickle file.

    Parameters
    ----------
    file : str
        Path to the json file.
    fields : list
        List of fields to extract.

    Returns
    -------
    value
    """

    if data_file is None:
        return None

    if data_file.endswith(".json"):
        with open(data_file) as f:
            local_data = json.load(f)
    elif data_file.endswith(".npz"):
        local_data = np.load(data_file)

    values = []
    for field in fields:
        if field in local_data:
            values.append(local_data[field])
        else:
            raise ValueError(f"No field {field} found in the json/npz file.")

    return values


def pdockq(data, verbose=True, compute_pairs=True):
    r"""Compute the pdockQ [1]_ from the pdb file, optionally for each chain pair.

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where:

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.724`, :math:`x0 = 152.611`, :math:`k = 0.052` and :math:`b = 0.018`.

    Parameters
    ----------
    data : AFData
        object containing the data
    verbose : bool
        print progress bar
    compute_pairs : bool
        if True, compute pdockQ for each chain pair

    Returns
    -------
    data : AFData
        The modified AFData object with results added to the dataframe.

    References
    ----------
    .. [1] Bryant P, Pozzati G and Elofsson A. Improved prediction of
        protein-protein interactions using AlphaFold2. *Nature Communications*.
        vol. 13, 1265 (2022)
        https://www.nature.com/articles/s41467-022-28865-w
    """

    from pdb_numpy.analysis import compute_pdockQ
    import numpy as np

    pdockq_list = []
    disable = False if verbose else True

    for idx, pdb in enumerate(tqdm(data.df["pdb"], total=len(data.df["pdb"]), disable=disable)):
        if pdb is None or pdb is np.nan:
            pdockq_list.append(None)
            continue

        try:
            model = pdb_numpy.Coor(pdb)
            
            # 전체 복합체에 대한 pdockQ 계산
            scores = compute_pdockQ(model)
            pdockq_list.append(scores[0])  # 첫 번째 모델의 점수
            
            # 체인 쌍별 계산
            if compute_pairs:
                model_seq = model.get_aa_na_seq()
                chains = list(model_seq.keys())
                
                if verbose:
                    print(f"Processing model {idx} with chains: {chains}")
                
                for i, chain1 in enumerate(chains):
                    for j, chain2 in enumerate(chains):
                        if i < j:  # 각 쌍은 한 번만 계산
                            # 이 체인 쌍의 pdockQ 계산
                            pair_score = compute_pdockQ(
                                model,
                                rec_chains=[chain1],
                                lig_chains=[chain2]
                            )[0]
                            
                            # 결과 저장
                            pair_col = f"pdockq_{chain1}{chain2}"
                            if pair_col not in data.df.columns:
                                data.df[pair_col] = None
                            data.df.at[idx, pair_col] = pair_score
                            
                            if verbose:
                                print(f"  Chain pair {chain1}-{chain2}: pdockQ = {pair_score:.4f}")
                                
        except Exception as e:
            if verbose:
                print(f"Error processing model {idx}: {str(e)}")
            pdockq_list.append(None)

    data.df.loc[:, "pdockq"] = pdockq_list
    
    return data  # 메소드 체이닝 지원


def mpdockq(data, verbose=True, compute_pairs=True):
    r"""Compute the mpDockq [2]_ from the pdb file, optionally for each chain pair.

    .. math::
        pDockQ = \frac{L}{1 + e^{-k (x-x_{0})}} + b

    where:

    .. math::
        x = \overline{plDDT_{interface}} \cdot log(number \: of \: interface \: contacts)

    :math:`L = 0.728`, :math:`x0 = 309.375`, :math:`k = 0.098` and :math:`b = 0.262`.

    Implementation was inspired from https://gitlab.com/ElofssonLab/FoldDock/-/blob/main/src/pdockq.py

    Parameters
    ----------
    data : AFData
        object containing the data
    verbose : bool
        print progress bar
    compute_pairs : bool
        if True, compute mpDockq for each chain pair

    Returns
    -------
    data : AFData
        The modified AFData object with results added to the dataframe.

    References
    ----------
    .. [2] Bryant P, Pozzati G, Zhu W, Shenoy A, Kundrotas P & Elofsson A.
        Predicting the structure of large protein complexes using AlphaFold and Monte
        Carlo tree search. *Nature Communications*. vol. 13, 6028 (2022)
        https://www.nature.com/articles/s41467-022-33729-4
    """

    from pdb_numpy.analysis import compute_pdockQ
    import numpy as np

    pdockq_list = []
    disable = False if verbose else True

    # 전체 복합체에 대한 mpdockq 계산
    for idx, pdb in enumerate(tqdm(data.df["pdb"], total=len(data.df["pdb"]), disable=disable)):
        if pdb is None or pdb is np.nan:
            pdockq_list.append(None)
            continue

        try:
            model = pdb_numpy.Coor(pdb)
            
            # 전체 복합체에 대한 mpdockq 계산
            scores = compute_pdockQ(
                model, cutoff=8.0, L=0.728, x0=309.375, k=0.098, b=0.262
            )
            pdockq_list.append(scores[0])  # 첫 번째 모델의 점수
            
            # 각 체인 쌍에 대한 mpdockq 계산
            if compute_pairs:
                model_seq = model.get_aa_na_seq()
                chains = list(model_seq.keys())
                
                if verbose:
                    print(f"Processing model {idx} with chains: {chains}")
                
                for i, chain1 in enumerate(chains):
                    for j, chain2 in enumerate(chains):
                        if i < j:  # 각 쌍은 한 번만 계산 (AB = BA)
                            # 이 체인 쌍의 mpdockq 계산
                            pair_score = compute_pdockQ(
                                model,
                                rec_chains=[chain1],
                                lig_chains=[chain2],
                                cutoff=8.0,
                                L=0.728, 
                                x0=309.375, 
                                k=0.098, 
                                b=0.262
                            )[0]  # 첫 번째 모델의 점수
                            
                            # 결과 저장
                            pair_col = f"mpdockq_{chain1}{chain2}"
                            if pair_col not in data.df.columns:
                                data.df[pair_col] = None
                            data.df.at[idx, pair_col] = pair_score
                            
                            if verbose:
                                print(f"  Chain pair {chain1}-{chain2}: mpDockQ = {pair_score:.4f}")
                                
        except Exception as e:
            if verbose:
                print(f"Error processing model {idx}: {str(e)}")
            pdockq_list.append(None)

    data.df.loc[:, "mpdockq"] = pdockq_list
    
    return data  # 메소드 체이닝 지원


def pdockq2(data, verbose=True, compute_pairs=True):
    r"""Compute pdockq2 from the pdb file, optionally for each chain pair.

    Parameters
    ----------
    data : AFData
        데이터 객체
    verbose : bool
        진행 상황 표시 여부
    compute_pairs : bool
        체인 쌍별 계산 여부
        
    Returns
    -------
    data : AFData
        수정된 데이터 객체
    """
    from pdb_numpy.analysis import compute_pdockQ2
    import numpy as np

    if "data_file" not in data.df.columns:
        raise ValueError(
            "No \`data_file\` column found in the dataframe. pae scores are required to compute pdockq2."
        )

    disable = False if verbose else True
    
    # 각 모델별 결과 처리
    pdockq2_list = []
    
    for idx, (pdb, data_path) in enumerate(tqdm(
        zip(data.df["pdb"], data.df["data_file"]),
        total=len(data.df["pdb"]),
        disable=disable,
    )):
        if (
            pdb is not None
            and pdb is not np.nan
            and data_path is not None
            and data_path is not np.nan
        ):
            try:
                model = pdb_numpy.Coor(pdb)
                pae_array = get_pae(data_path)
                
                # 원본 함수를 사용하여 전체 pdockq2 계산
                pdockq2_values = compute_pdockQ2(model, pae_array)
                
                # 첫 번째 체인의 첫 번째 모델 점수를 전체 점수로 사용
                # (원본 함수는 체인별로 리스트의 리스트를 반환합니다)
                if len(pdockq2_values) > 0 and len(pdockq2_values[0]) > 0:
                    pdockq2_list.append(pdockq2_values[0][0])
                else:
                    pdockq2_list.append(None)
                
                # 체인 쌍별 계산
                if compute_pairs:
                    sel = "(protein and name CA) or (dna and name P) or ions or resname LIG"
                    models_CA = model.select_atoms(sel)
                    chains = np.unique(models_CA.chain)
                    
                    if verbose:
                        print(f"Processing model {idx} with chains: {chains}")
                    
                    # 체인 쌍별 계산
                    for i, chain1 in enumerate(chains):
                        for j, chain2 in enumerate(chains):
                            if i < j:  # 각 쌍은 한 번만 계산
                                # 체인 쌍의 pdockq2 직접 계산
                                try:
                                    # 인터페이스 원자 선택
                                    first_model = models_CA.models[0]
                                    chain1_sel = first_model.select_atoms(
                                        f"(chain {chain1} and within 8.0 of chain {chain2})"
                                    )
                                    chain2_sel = first_model.select_atoms(
                                        f"(chain {chain2} and within 8.0 of chain {chain1})"
                                    )
                                    
                                    if chain1_sel.len == 0 or chain2_sel.len == 0:
                                        pair_score = 0.0
                                    else:
                                        # 거리 행렬 계산 및 접촉 식별
                                        dist_mat = distance_matrix(chain1_sel.xyz, chain2_sel.xyz)
                                        indexes = np.where(dist_mat < 8.0)
                                        
                                        # 잔기 인덱스 추출
                                        x_indexes = chain1_sel.uniq_resid[indexes[0]]
                                        y_indexes = chain2_sel.uniq_resid[indexes[1]]
                                        
                                        if len(x_indexes) == 0 or len(y_indexes) == 0:
                                            pair_score = 0.0
                                        else:
                                            # PAE 값 추출
                                            pae_sel = pae_array[x_indexes - 1, y_indexes - 1]  # 0-인덱스로 보정
                                            
                                            # 정규화된 인터페이스 PAE
                                            d0 = 10.0
                                            norm_if_interpae = np.mean(1 / (1 + (pae_sel / d0) ** 2))
                                            
                                            # 인터페이스 pLDDT
                                            plddt_avg = np.mean(first_model.beta[x_indexes - 1])  # 0-인덱스로 보정
                                            
                                            # pdockQ2 계산
                                            L = 1.31034849e00
                                            x0 = 8.47326239e01
                                            k = 7.47157696e-02
                                            b = 5.01886443e-03
                                            
                                            x = norm_if_interpae * plddt_avg
                                            pair_score = float(L / (1 + np.exp(-k * (x - x0))) + b)
                                    
                                    # 결과 저장
                                    pair_col = f"pdockq2_{chain1}{chain2}"
                                    if pair_col not in data.df.columns:
                                        data.df[pair_col] = None
                                    data.df.at[idx, pair_col] = pair_score
                                    
                                    if verbose:
                                        print(f"  Chain pair {chain1}-{chain2}: pdockQ2 = {pair_score:.4f}")
                                
                                except Exception as e:
                                    print(f"  Error calculating pdockQ2 for chain pair {chain1}-{chain2}: {str(e)}")
                                    data.df.at[idx, f"pdockq2_{chain1}{chain2}"] = None
                                
            except Exception as e:
                print(f"Error processing model {idx}: {str(e)}")
                pdockq2_list.append(None)
        else:
            pdockq2_list.append(None)
    
    # 전체 pdockq2 값 저장
    data.df["pdockq2"] = pdockq2_list
    
    return data  # 메소드 체이닝 지원


def inter_chain_pae(data, fun=np.mean, verbose=True):
    """Read the PAE matrix and extract the average inter chain PAE.

    Parameters
    ----------
    data : AFData
        object containing the data
    fun : function
        function to apply to the PAE scores
    verbose : bool
        print progress bar

    Returns
    -------
    None
    """
    pae_list = []

    disable = False if verbose else True

    if "data_file" not in data.df.columns:
        raise ValueError(
            "No 'data_file' column found in the dataframe. pae scores are required to compute pdockq2."
        )

    for query, data_path in tqdm(
        zip(data.df["query"], data.df["data_file"]),
        total=len(data.df["data_file"]),
        disable=disable,
    ):
        if data_path is not None and data_path is not np.nan:
            pae_array = get_pae(data_path)

            chain_lens = data.chain_length[query]
            chain_len_sums = np.cumsum([0] + chain_lens)
            chain_ids = data.chains[query]

            pae_dict = {}

            for i in range(len(chain_lens)):
                for j in range(len(chain_lens)):
                    pae_val = fun(
                        pae_array[
                            chain_len_sums[i] : chain_len_sums[i + 1],
                            chain_len_sums[j] : chain_len_sums[j + 1],
                        ]
                    )
                    pae_dict[f"PAE_{chain_ids[i]}_{chain_ids[j]}"] = pae_val

            pae_list.append(pae_dict)
        else:
            pae_list.append({})

    pae_df = pd.DataFrame(pae_list)

    for col in pae_df.columns:
        data.df.loc[:, col] = pae_df.loc[:, col].to_numpy()


def compute_LIS_matrix(
    pae_array,
    chain_length,
    pae_cutoff=12.0,
):
    r"""Compute the LIS score as define in [1]_.

    Implementation was inspired from implementation in https://github.com/flyark/AFM-LIS

    Parameters
    ----------
    pae_array : np.array
        array of predicted PAE
    chain_length : list
        list of chain lengths
    pae_cutoff : float
        cutoff for native contacts, default is 8.0 A

    Returns
    -------
    list
        LIS scores

    References
    ----------

    .. [1] Kim AR, Hu Y, Comjean A, Rodiger J, Mohr SE, Perrimon N. "Enhanced
        Protein-Protein Interaction Discovery via AlphaFold-Multimer" bioRxiv (2024).
        https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

    """

    if pae_array is None:
        return None

    chain_len_sums = np.cumsum([0] + chain_length)

    # Use list instead of array, because
    # df[column].iloc[:] = LIS_list does not work with numpy array
    LIS_list = []

    trans_matrix = np.zeros_like(pae_array)
    mask = pae_array < pae_cutoff
    trans_matrix[mask] = 1 - pae_array[mask] / pae_cutoff

    for i in range(len(chain_length)):
        i_start = chain_len_sums[i]
        i_end = chain_len_sums[i + 1]
        local_LIS_list = []
        for j in range(len(chain_length)):
            j_start = chain_len_sums[j]
            j_end = chain_len_sums[j + 1]

            submatrix = trans_matrix[i_start:i_end, j_start:j_end]

            if np.any(submatrix > 0):
                local_LIS_list.append(submatrix[submatrix > 0].mean())
            else:
                local_LIS_list.append(0)
        LIS_list.append(local_LIS_list)

    return LIS_list


def LIS_matrix(data, pae_cutoff=12.0, verbose=True):
    """
    Compute the LIS score for each chain pair and add to the dataframe.
    
    Parameters
    ----------
    data : AFData
        object containing the data
    pae_cutoff : float
        cutoff for PAE matrix values, default is 12.0 A
    verbose : bool
        print progress bar
        
    Returns
    -------
    None
        The dataframe is modified in place.
    """
    disable = False if verbose else True
    
    # DataFrame에 이미 LIS 컬럼이 있다면 제거
    lis_cols = [col for col in data.df.columns if col.startswith("LIS_")]
    if lis_cols:
        data.df = data.df.drop(columns=lis_cols)
    
    # 전체 행렬도 저장
    data.df.loc[:, "LIS"] = None
    
    for idx in tqdm(range(len(data.df)), total=len(data.df), disable=disable):
        try:
            query = data.df.iloc[idx]["query"]
            data_path = data.df.iloc[idx]["data_file"]
            
            if data.chain_length[query] is None:
                continue
                
            # 체인 정보 가져오기
            chains = data.chains[query]
            chain_lens = data.chain_length[query]
            
            pae_array = get_pae(data_path)
            if pae_array is None:
                continue
                
            # LIS 행렬 계산
            LIS_matrix = compute_LIS_matrix(pae_array, chain_lens, pae_cutoff)
            
            # 전체 행렬 저장
            data.df.at[idx, "LIS"] = LIS_matrix
            
            # 체인 쌍 LIS 값을 저장할 딕셔너리
            pair_lis_values = {}
            
            # 모든 체인 쌍 조합 생성
            for i in range(len(chains)):
                for j in range(len(chains)):
                    if i != j:  # 서로 다른 체인 간의 LIS만 처리
                        chain_i = chains[i]
                        chain_j = chains[j]
                        
                        # 알파벳 순서로 정렬하여 키 생성
                        sorted_chains = sorted([chain_i, chain_j])
                        pair_id = f"{sorted_chains[0]}{sorted_chains[1]}"
                        
                        # 해당 쌍의 LIS 값 가져오기
                        lis_value = float(LIS_matrix[i][j])
                        
                        # 딕셔너리에 값 추가 또는 업데이트
                        if pair_id in pair_lis_values:
                            # 이미 있는 경우 값을 더함 (나중에 평균 계산)
                            pair_lis_values[pair_id].append(lis_value)
                        else:
                            # 새로운 키인 경우 리스트 시작
                            pair_lis_values[pair_id] = [lis_value]
            
            # 각 쌍의 평균 LIS 계산 및 저장
            for pair_id, values in pair_lis_values.items():
                avg_pair_lis = float(np.mean(values))
                data.df.at[idx, f"LIS_{pair_id}"] = avg_pair_lis
            
            # 모든 쌍의 평균 LIS 계산
            all_lis_values = []
            for values in pair_lis_values.values():
                all_lis_values.extend(values)
            
            if all_lis_values:
                avg_lis = float(np.mean(all_lis_values))
                data.df.at[idx, "avg_LIS"] = avg_lis
        
        except Exception as e:
            print(f"Error processing query at index {idx}: {e}")
    
    return data


def PAE_matrix(data, verbose=True, fun=np.average):
    """
    Compute the average (or something else) PAE matrix.

    Parameters
    ----------
    data : AFData
        object containing the data
    verbose : bool
        print progress bar
    fun : function
        function to apply to the PAE scores
    
    Returns
    -------
    None
        The dataframe is modified in place.
    """

    PAE_avg_list = []

    disable = False if verbose else True

    for query, data_path in tqdm(
        zip(data.df["query"], data.df["data_file"]),
        total=len(data.df["query"]),
        disable=disable,
    ):
        if data.chain_length[query] is None:
            PAE_avg_list.append(None)
            continue

        pae_array = get_pae(data_path)
        chain_len_cum = np.cumsum([data.chain_length[query]])
        chain_len_cum = np.insert(chain_len_cum, 0, 0)

        avg_matrix = np.zeros((len(chain_len_cum)-1, len(chain_len_cum)-1))
        
        for i in range(len(chain_len_cum)-1):
            for j in range(len(chain_len_cum)-1):
                avg_matrix[i,j] = fun(pae_array[chain_len_cum[i]:chain_len_cum[i+1], chain_len_cum[j]:chain_len_cum[j+1]])

        PAE_avg_list.append(avg_matrix)

    assert len(PAE_avg_list) == len(data.df["query"])
    data.df.loc[:, "PAE_fun"] = PAE_avg_list

def calculate_contact_map(model, distance_threshold=10.0):
    """단백질 구조에서 CA 원자 간 접촉 맵을 계산합니다."""
    import numpy as np
    
    # 첫 번째 모델 사용
    m = model.models[0]
    
    # CA 원자만 선택
    ca_mask = np.array(m.name) == 'CA'
    if not np.any(ca_mask):
        raise ValueError("No CA atoms found in the model")
    
    # CA 원자 좌표
    ca_coords = np.vstack([m.x[ca_mask], m.y[ca_mask], m.z[ca_mask]]).T
    
    # CA 원자의 체인과 잔기 정보
    ca_chains = np.array(m.chain)[ca_mask]
    ca_resids = np.array(m.resid)[ca_mask]
    
    # 인덱스별 체인-잔기 매핑
    n_ca = len(ca_chains)
    residue_info = [(ca_chains[i], ca_resids[i]) for i in range(n_ca)]
    
    # 접촉 맵 초기화
    contact_map = np.zeros((n_ca, n_ca), dtype=bool)
    
    # 거리 기반 접촉 계산
    for i in range(n_ca):
        chain_i = ca_chains[i]
        for j in range(i+1, n_ca):
            chain_j = ca_chains[j]
            if chain_i != chain_j:  # 다른 체인만 고려
                # 두 CA 원자 간 거리 계산
                dist = np.sqrt(np.sum((ca_coords[i] - ca_coords[j])**2))
                if dist < distance_threshold:
                    contact_map[i, j] = True
                    contact_map[j, i] = True
    
    return contact_map, residue_info

def build_residue_to_atom_map(model):
    """
    각 잔기에 속하는 원자들의 인덱스 매핑을 생성합니다.
    
    Parameters
    ----------
    model : pdb_numpy.Coor
        단백질 구조 객체
        
    Returns
    -------
    residue_to_atom_map : dict
        {잔기 인덱스: [원자 인덱스 리스트]} 형태의 딕셔너리
    """
    import numpy as np
    
    # 첫 번째 모델 사용
    m = model.models[0]
    
    # 각 원자의 잔기 식별자 (체인ID + 잔기번호) 생성
    residue_ids = np.array([f"{chain}_{resid}" for chain, resid in zip(m.chain, m.resid)])
    unique_residues = np.unique(residue_ids)
    
    # 잔기-원자 매핑 딕셔너리 생성
    residue_to_atom_map = {}
    for i, res_id in enumerate(unique_residues):
        atom_indices = np.where(residue_ids == res_id)[0]
        residue_to_atom_map[i] = atom_indices.tolist()
    
    return residue_to_atom_map

def compute_interface_plddt(model, plddt_data, contact_map=None, distance_threshold=8.0):
    """
    체인 간 인터페이스의 pLDDT를 계산합니다.
    
    Parameters
    ----------
    model : pdb_numpy.Coor
        단백질 구조 객체
    plddt_data : numpy.ndarray
        각 원자의 pLDDT 값 배열
    contact_map : numpy.ndarray, optional
        사전 계산된 접촉 맵. None이면 계산됨
    distance_threshold : float
        접촉으로 간주할 거리 임계값(Å)
        
    Returns
    -------
    interface_plddt_dict : dict
        {'체인1체인2': plddt_값} 형태의 딕셔너리
    avg_interface_plddt : float
        모든 인터페이스의 평균 pLDDT
    """
    import numpy as np
    
    # 첫 번째 모델 사용
    m = model.models[0]
    chains = np.unique(m.chain)
    
    # 접촉 맵이 제공되지 않은 경우 계산
    if contact_map is None:
        contact_map = calculate_contact_map(model, distance_threshold)
    
    # 잔기-원자 매핑 구축
    residue_to_atom_map = build_residue_to_atom_map(model)
    
    # 체인별 잔기 카운트
    chain_residue_counts = {}
    for chain in chains:
        chain_residues = np.unique(m.resid[m.chain == chain])
        chain_residue_counts[chain] = len(chain_residues)
    
    # 인터페이스 pLDDT 계산
    interface_plddt_dict = {}
    all_interface_plddts = []
    
    # 각 체인 쌍에 대해 계산
    for i in range(len(chains)):
        for j in range(i+1, len(chains)):
            chain1, chain2 = chains[i], chains[j]
            pair_id = f"{chain1}{chain2}"
            
            # 체인별 원자 인덱스
            atoms1 = np.where(m.chain == chain1)[0]
            atoms2 = np.where(m.chain == chain2)[0]
            
            # 체인별 잔기 인덱스 계산
            residues1 = set()
            residues2 = set()
            
            for res_idx, atom_indices in residue_to_atom_map.items():
                if any(idx in atoms1 for idx in atom_indices):
                    residues1.add(res_idx)
                if any(idx in atoms2 for idx in atom_indices):
                    residues2.add(res_idx)
            
            # 접촉 맵에서 인터페이스 잔기 식별
            interface_residues = set()
            for res1 in residues1:
                for res2 in residues2:
                    if contact_map[res1, res2] == 1:
                        interface_residues.add(res1)
                        interface_residues.add(res2)
            
            # 인터페이스 잔기에 속하는 원자들의 pLDDT 수집
            plddt_values = []
            for res_idx in interface_residues:
                for atom_idx in residue_to_atom_map.get(res_idx, []):
                    if atom_idx < len(plddt_data):
                        plddt_values.append(plddt_data[atom_idx])
            
            # 평균 계산
            if plddt_values:
                avg_plddt = np.round(np.mean(plddt_values), decimals=2)
                interface_plddt_dict[pair_id] = float(avg_plddt)
                all_interface_plddts.append(avg_plddt)
            else:
                interface_plddt_dict[pair_id] = np.nan
    
    # 전체 인터페이스 평균 계산
    valid_plddts = [v for v in interface_plddt_dict.values() if not np.isnan(v)]
    avg_interface_plddt = np.round(np.mean(valid_plddts), decimals=2) if valid_plddts else np.nan
    
    return interface_plddt_dict, float(avg_interface_plddt)

def calculate_interface_pae(model, pae_matrix, contact_map=None, distance_threshold=8.0):
    """
    체인 간 인터페이스의 PAE를 계산합니다.
    
    Parameters
    ----------
    model : pdb_numpy.Coor
        단백질 구조 객체
    pae_matrix : numpy.ndarray
        잔기 간 PAE 행렬
    contact_map : numpy.ndarray, optional
        사전 계산된 접촉 맵. None이면 계산됨
    distance_threshold : float
        접촉으로 간주할 거리 임계값(Å)
        
    Returns
    -------
    interface_pae_dict : dict
        {'체인1체인2': pae_값} 형태의 딕셔너리
    avg_interface_pae : float
        인터페이스의 평균 PAE
    avg_total_pae : float
        전체 체인 간 PAE의 평균
    """
    import numpy as np
    
    # 첫 번째 모델 사용
    m = model.models[0]
    chains = np.unique(m.chain)
    
    # 접촉 맵이 제공되지 않은 경우 계산
    if contact_map is None:
        contact_map = calculate_contact_map(model, distance_threshold)
    
    # 체인별 잔기 카운트 및 인덱스 범위 계산
    chain_residue_counts = {}
    chain_start_indices = {}
    
    start_idx = 0
    for chain in chains:
        chain_residues = np.unique(m.resid[m.chain == chain])
        count = len(chain_residues)
        chain_residue_counts[chain] = count
        chain_start_indices[chain] = start_idx
        start_idx += count
    
    # 인터페이스 PAE 계산
    interface_pae_dict = {}
    total_pae_values = []
    
    # 각 체인 쌍에 대해 계산
    for i in range(len(chains)):
        for j in range(i+1, len(chains)):
            chain1, chain2 = chains[i], chains[j]
            pair_id = f"{chain1}{chain2}"
            
            # 체인 범위 인덱스
            start1 = chain_start_indices[chain1]
            end1 = start1 + chain_residue_counts[chain1]
            start2 = chain_start_indices[chain2]
            end2 = start2 + chain_residue_counts[chain2]
            
            # PAE 및 접촉 맵 블록 추출
            pae_block = pae_matrix[start1:end1, start2:end2]
            contact_block = contact_map[start1:end1, start2:end2]
            
            # 모든 PAE 값 수집
            total_pae_values.append(pae_block.flatten())
            
            # 인터페이스 PAE 계산
            interface_values = pae_block[contact_block == 1]
            
            if interface_values.size > 0:
                interface_pae = float(np.mean(interface_values).round(2))
            else:
                interface_pae = np.nan
                
            interface_pae_dict[pair_id] = interface_pae
    
    # 전체 PAE 평균 계산
    if total_pae_values:
        total_pae_values = np.concatenate(total_pae_values)
        avg_total_pae = float(np.mean(total_pae_values).round(2))
    else:
        avg_total_pae = np.nan
    
    # 인터페이스 PAE 평균 계산
    valid_paes = [v for v in interface_pae_dict.values() if not np.isnan(v)]
    avg_interface_pae = float(np.mean(valid_paes).round(2)) if valid_paes else np.nan
    
    return interface_pae_dict, avg_interface_pae, avg_total_pae


def verify_atom_order_matching(model, json_data, verbose=True):
    """
    CIF 파일의 원자 순서와 JSON의 atom_plddts 순서가 일치하는지 검증합니다.
    
    Parameters
    ----------
    model : pdb_numpy.Coor
        파싱된 CIF 모델
    json_data : dict
        파싱된 JSON 데이터
    verbose : bool, optional
        검증 과정에서 상세 정보를 출력할지 여부, 기본값은 True
    
    Returns
    -------
    bool
        순서가 일치하면 True, 그렇지 않으면 False
    """
    from collections import Counter
    
    atoms = model.models[0]
    atom_plddts = np.array(json_data['atom_plddts'])
    atom_chain_ids = np.array(json_data['atom_chain_ids'])
    
    # 1. 원자 개수 비교
    cif_atom_count = len(atoms.name)
    json_atom_count = len(atom_plddts)
    if verbose:
        print(f"CIF atom count: {cif_atom_count}, JSON atom_plddts count: {json_atom_count}")
    if cif_atom_count != json_atom_count:
        if verbose:
            print("WARNING: Atom count mismatch!")
        return False
    
    # 2. 체인 ID 분포 비교
    cif_chain_ids = list(atoms.chain)
    json_chain_ids = atom_chain_ids
    
    cif_chain_counts = Counter(cif_chain_ids)
    json_chain_counts = Counter(json_chain_ids)
    
    chains_match = True
    for chain in set(sorted(list(cif_chain_ids)) + sorted(list(json_chain_ids))):
        if cif_chain_counts[chain] != json_chain_counts[chain]:
            chains_match = False
            if verbose:
                print(f"Chain {chain} count mismatch: CIF={cif_chain_counts[chain]}, JSON={json_chain_counts[chain]}")
    
    if not chains_match:
        if verbose:
            print("WARNING: Chain ID distributions don't match!")
        return False
    
    # 3. 샘플링 검증 - 첫 20개, 중간 20개, 마지막 20개 원자의 체인 ID 비교
    sample_points = [0, cif_atom_count//2, max(0, cif_atom_count-20)]
    for start in sample_points:
        end = min(start + 20, cif_atom_count)
        cif_sample = cif_chain_ids[start:end]
        json_sample = json_chain_ids[start:end]
        match = all(c1 == c2 for c1, c2 in zip(cif_sample, json_sample))
        if verbose:
            print(f"Chain IDs from index {start} to {end-1}: {'Match' if match else 'MISMATCH'}")
        if not match:
            return False
    
    if verbose:
        print("All verifications passed! CIF and JSON atom orders likely match.")
    return True

def add_interface_metrics(data_obj, distance_threshold=8.0, verbose=True):
    """
    [DEPRECATED] 이 함수는 향후 삭제될 예정입니다. Data 클래스의 analyze_interfaces 메서드를 사용하세요.
    
    Parameters
    ----------
    data_obj : AFData
        AlphaFold3 결과 데이터를 포함하는 객체
    distance_threshold : float, optional
        인터페이스 정의에 사용되는 거리 임계값 (Å), 기본값은 8.0Å
    verbose : bool, optional
        계산 과정에서 상세 정보를 출력할지 여부, 기본값은 True
        
    Returns
    -------
    pandas.DataFrame
        인터페이스 메트릭이 추가된 데이터프레임
    """
    import warnings
    warnings.warn(
        "add_interface_metrics 함수는 deprecated되었으며 향후 버전에서 제거될 예정입니다. "
        "대신 Data 클래스의 analyze_interfaces 메서드를 사용하세요.",
        DeprecationWarning, 
        stacklevel=2
    )
    
    # 내부적으로 analyze_interfaces 호출
    data_obj.analyze_interfaces(distance_threshold=distance_threshold, verbose=verbose)
    return data_obj.df

def distance_matrix(coords1, coords2):
    """
    두 좌표 세트 간의 거리 행렬을 계산합니다.
    """
    import numpy as np
    
    # 브로드캐스팅을 사용한 거리 계산
    delta = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    return np.sqrt(np.sum(delta**2, axis=2))

def pdockq_pairs(data, verbose=True):
    """
    각 체인 쌍에 대한 pdockQ 점수를 계산합니다.
    
    Parameters
    ----------
    data : AFData
        데이터 객체
    verbose : bool
        진행 상황 표시 여부
        
    Returns
    -------
    None
        데이터프레임이 제자리에서 수정됩니다.
    """
    from pdb_numpy.analysis import compute_pdockQ
    from tqdm import tqdm
    import numpy as np
    
    disable = False if verbose else True
    
    # 각 모델별 처리
    for idx, pdb in enumerate(tqdm(
        data.df["pdb"], 
        total=len(data.df["pdb"]),
        disable=disable,
    )):
        if pdb is not None and pdb is not np.nan:
            try:
                model = pdb_numpy.Coor(pdb)
                model_seq = model.get_aa_na_seq()
                chains = list(model_seq.keys())
                
                # 각 체인 쌍에 대한 pdockQ 계산
                for i, chain1 in enumerate(chains):
                    for j, chain2 in enumerate(chains):
                        if i < j:  # 각 쌍은It's the same calculation for AB and BA
                            # 이 체인 쌍의 pdockQ 계산
                            pdockq_score = compute_pdockQ(
                                model,
                                rec_chains=[chain1],
                                lig_chains=[chain2],
                                cutoff=8.0,
                            )[0]  # 첫 번째 모델의 점수
                            
                            # 결과 저장
                            pair_col = f"pdockq_{chain1}{chain2}"
                            if pair_col not in data.df.columns:
                                data.df[pair_col] = None
                            data.df.at[idx, pair_col] = pdockq_score
                            
                            if verbose:
                                print(f"  Chain pair {chain1}-{chain2}: pdockQ = {pdockq_score:.4f}")
            
            except Exception as e:
                print(f"Error processing model {idx}: {str(e)}")
    
    return data  # 메소드 체이닝 지원

def calculate_dockq(data_obj, model_idx=None, native_path=None, rec_chains=None, lig_chains=None, 
                   native_rec_chains=None, native_lig_chains=None, verbose=True, skip_existing=True):
    """
    AlphaFold 모델과 기준 구조(native) 사이의 dockQ 점수를 계산합니다.

    dockQ 점수는 단백질-단백질 상호작용(PPI) 예측의 품질을 평가하는 지표로,
    Fnat(native contact 분율), LRMS(ligand RMSD), iRMS(interface RMSD)의 
    조합으로 계산됩니다.

    Parameters
    ----------
    data_obj : Data
        AlphaFold 결과를 포함하는 Data 객체
    model_idx : int 또는 list, optional
        분석할 모델의 인덱스. None이면 모든 모델 분석
    native_path : str, optional
        기준 구조(native)의 PDB/mmCIF 파일 경로. None이면 data_obj.df['native_path'] 사용
    rec_chains : list 또는 str, optional
        모델의 receptor 체인. None이면 자동 감지
    lig_chains : list 또는 str, optional
        모델의 ligand 체인. None이면 자동 감지
    native_rec_chains : list 또는 str, optional
        기준 구조의 receptor 체인. None이면 자동 감지
    native_lig_chains : list 또는 str, optional
        기준 구조의 ligand 체인. None이면 자동 감지
    verbose : bool, optional
        진행 상태를 출력할지 여부, 기본값은 True
    skip_existing : bool, optional
        이미 DockQ 값이 있는 모델을 건너뛸지 여부, 기본값은 True
        
    Returns
    -------
    pandas.DataFrame
        dockQ 점수 및 구성 요소가 추가된 데이터프레임
    """
    import pdb_numpy
    import numpy as np
    import logging
    import pandas as pd
    from pdb_numpy.analysis import native_contact, interface_rmsd
    from pdb_numpy.alignement import coor_align, align_seq_based, rmsd_seq_based
    from pdb_numpy.select import remove_incomplete_backbone_residues
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    if not verbose:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    # 모델 인덱스 설정
    if model_idx is None:
        model_indices = range(len(data_obj.df))
    elif isinstance(model_idx, int):
        model_indices = [model_idx]
    else:
        model_indices = model_idx
    
    # DockQ 컬럼이 존재하는지 확인하고, 값이 있는 모델 건너뛰기 (skip_existing=True일 때)
    if skip_existing and 'DockQ' in data_obj.df.columns:
        # 이미 DockQ 값이 있는 인덱스 제외
        existing_dockq_indices = [i for i in model_indices if not pd.isna(data_obj.df.iloc[i].get('DockQ'))]
        if existing_dockq_indices:
            if verbose:
                print(f"Skipping {len(existing_dockq_indices)} models that already have DockQ values")
                
            # 건너뛸 인덱스 제외
            model_indices = [i for i in model_indices if i not in existing_dockq_indices or pd.isna(data_obj.df.iloc[i].get('DockQ'))]
            
            # 모든 모델을 건너뛰는 경우 조기 종료
            if not model_indices:
                if verbose:
                    print("All specified models already have DockQ values. Nothing to calculate.")
                return data_obj.df
    
    # native_path 컬럼 확인
    if native_path is None and 'native_path' not in data_obj.df.columns:
        raise ValueError("native_path가 제공되지 않았고 DataFrame에 'native_path' 컬럼이 없습니다.")
    
    # 결과 저장 딕셔너리
    results = {
        "Fnat": [],
        "Fnonnat": [],
        "rRMS": [],
        "iRMS": [],
        "LRMS": [],
        "DockQ": []
    }
    
    # 체인 리스트 형식 변환
    if rec_chains is not None and not isinstance(rec_chains, list):
        rec_chains = [rec_chains]
    if lig_chains is not None and not isinstance(lig_chains, list):
        lig_chains = [lig_chains]
    if native_rec_chains is not None and not isinstance(native_rec_chains, list):
        native_rec_chains = [native_rec_chains]
    if native_lig_chains is not None and not isinstance(native_lig_chains, list):
        native_lig_chains = [native_lig_chains]
    
    # 각 모델 분석
    for i in model_indices:
        try:
            model_path = data_obj.df.iloc[i]["pdb"]
            
            # 현재 모델의 native_path 결정
            current_native_path = native_path
            if current_native_path is None:
                current_native_path = data_obj.df.iloc[i]["native_path"]
                if pd.isna(current_native_path):
                    if verbose:
                        print(f"모델 {i}의 native_path가 None입니다. 건너뜁니다.")
                    # 결과에 None 추가
                    results["Fnat"].append(None)
                    results["Fnonnat"].append(None)
                    results["rRMS"].append(None)
                    results["iRMS"].append(None)
                    results["LRMS"].append(None)
                    results["DockQ"].append(None)
                    continue
            
            if verbose:
                print(f"\nAnalyzing model {i}: {model_path}")
                print(f"Native structure: {current_native_path}")
            
            # Native 구조 로드
            try:
                native_coor = pdb_numpy.Coor(current_native_path)
                if verbose:
                    print(f"Loaded native structure from {current_native_path}")
            except Exception as e:
                print(f"Failed to load native structure: {str(e)}")
                # 결과에 None 추가
                results["Fnat"].append(None)
                results["Fnonnat"].append(None)
                results["rRMS"].append(None)
                results["iRMS"].append(None)
                results["LRMS"].append(None)
                results["DockQ"].append(None)
                continue
            
            # 모델 로드
            coor = pdb_numpy.Coor(model_path)
            
            # 체인 정보 얻기
            model_seq = coor.get_aa_seq()
            native_seq = native_coor.get_aa_seq()
            
            if verbose:
                print(f"Model chains: {list(model_seq.keys())}")
                print(f"Native chains: {list(native_seq.keys())}")
            
            # Ligand 체인 결정
            if lig_chains is None:
                lig_chains = [min(model_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]]
            if verbose:
                print(f'Model ligand chains: {" ".join(lig_chains)}')
                
            # Receptor 체인 결정
            if rec_chains is None:
                rec_chains = [chain for chain in model_seq if chain not in lig_chains]
            if verbose:
                print(f'Model receptor chains: {" ".join(rec_chains)}')
                
            # Native ligand 체인 결정
            if native_lig_chains is None:
                native_lig_chains = [min(native_seq.items(), key=lambda x: len(x[1].replace("-", "")))[0]]
            if verbose:
                print(f'Native ligand chains: {" ".join(native_lig_chains)}')
                
            # Native receptor 체인 결정
            if native_rec_chains is None:
                native_rec_chains = [chain for chain in native_seq if chain not in native_lig_chains]
            if verbose:
                print(f'Native receptor chains: {" ".join(native_rec_chains)}')
            
            # 체인 존재 확인
            for chain_list, name, avail_chains in [
                (rec_chains, "model receptor", list(model_seq.keys())),
                (lig_chains, "model ligand", list(model_seq.keys())),
                (native_rec_chains, "native receptor", list(native_seq.keys())),
                (native_lig_chains, "native ligand", list(native_seq.keys()))
            ]:
                for chain in chain_list:
                    if chain not in avail_chains:
                        print(f"Warning: Chain {chain} not found in {name}. Available: {avail_chains}")
            
            # 수소 및 비단백질 원자 제거, altloc 제거
            clean_coor = coor.select_atoms(
                f"protein and not altloc B C D and chain {' '.join(rec_chains + lig_chains)}"
            )
            clean_native_coor = native_coor.select_atoms(
                f"protein and not altloc B C D and chain {' '.join(native_rec_chains + native_lig_chains)}"
            )
            
            if clean_coor.models[0].len == 0:
                raise ValueError(f"No atoms selected for model with chains {rec_chains + lig_chains}")
            if clean_native_coor.models[0].len == 0:
                raise ValueError(f"No atoms selected for native with chains {native_rec_chains + native_lig_chains}")
                
            # 체인 및 원자 수 확인
            if verbose:
                print(f"Model atoms: {clean_coor.models[0].len}, Native atoms: {clean_native_coor.models[0].len}")
                print(f"Model chains: {np.unique(clean_coor.models[0].chain)}")
                print(f"Native chains: {np.unique(clean_native_coor.models[0].chain)}")
            
            # 불완전한 백본 잔기 제거
            clean_coor = remove_incomplete_backbone_residues(clean_coor)
            clean_native_coor = remove_incomplete_backbone_residues(clean_native_coor)
            
            if verbose:
                print(f"After cleaning - Model atoms: {clean_coor.models[0].len}, Native atoms: {clean_native_coor.models[0].len}")
            
            # 체인 순서 재정렬
            try:
                clean_coor.change_order("chain", rec_chains + lig_chains)
                clean_native_coor.change_order("chain", native_rec_chains + native_lig_chains)
            except Exception as e:
                print(f"Warning: Cannot reorder chains: {str(e)}")
            
            # 백본 원자 정의
            back_atom = ["CA", "N", "C", "O"]
            
            # 모델을 native 구조에 정렬 (receptor 기준)
            try:
                rmsd_prot_list, indices = align_seq_based(
                    clean_coor,
                    clean_native_coor,
                    chain_1=rec_chains,
                    chain_2=native_rec_chains,
                    back_names=back_atom,
                )
                
                if verbose:
                    print(f"Receptor RMSD: {rmsd_prot_list[0]:.3f} Å")
                    print(f"Found {len(indices[0])//len(back_atom):d} residues in common (receptor)")
                
                align_rec_index, align_rec_native_index = indices
                
            except Exception as e:
                print(f"Error during receptor alignment: {str(e)}")
                raise
            
            # Ligand RMSD 계산
            try:
                lrmsd_list, ligand_indices = rmsd_seq_based(
                    clean_coor,
                    clean_native_coor,
                    chain_1=lig_chains,
                    chain_2=native_lig_chains,
                    back_names=back_atom,
                )
                
                if verbose:
                    print(f"Ligand RMSD: {lrmsd_list[0]:.3f} Å")
                    print(f"Found {len(ligand_indices[0])//len(back_atom):d} residues in common (ligand)")
                
                align_lig_index, align_lig_native_index = ligand_indices
                
            except Exception as e:
                print(f"Error during ligand RMSD calculation: {str(e)}")
                raise
            
            # 공통 잔기 설정
            try:
                coor_residue = clean_coor.models[0].residue[align_rec_index + align_lig_index]
                native_residue = clean_native_coor.models[0].residue[align_rec_native_index + align_lig_native_index]
                
                coor_residue_unique = np.unique(coor_residue)
                native_residue_unique = np.unique(native_residue)
                
                if len(coor_residue_unique) != len(native_residue_unique):
                    print(f"Warning: Number of unique residues differs: {len(coor_residue_unique)} vs {len(native_residue_unique)}")
                
                # 인터페이스 추출
                interface_coor = clean_coor.select_atoms(
                    f'residue {" ".join([str(i) for i in coor_residue_unique])}'
                )
                interface_native_coor = clean_native_coor.select_atoms(
                    f'residue {" ".join([str(i) for i in native_residue_unique])}'
                )
                
                # 잔기 인덱스 초기화
                interface_coor.reset_residue_index()
                interface_native_coor.reset_residue_index()
                
            except Exception as e:
                print(f"Error processing residue indices: {str(e)}")
                raise
            
            # 인터페이스 RMSD 계산
            try:
                irmsd_list = interface_rmsd(
                    interface_coor,
                    interface_native_coor,
                    native_rec_chains,
                    native_lig_chains,
                    cutoff=10.0,
                    back_atom=back_atom,
                )
                if verbose:
                    print(f"Interface RMSD: {irmsd_list[0]:.3f} Å")
                
            except Exception as e:
                print(f"Error calculating interface RMSD: {str(e)}")
                irmsd_list = [None]
            
            # Native contact 분석
            try:
                fnat_list, fnonnat_list = native_contact(
                    interface_coor,
                    interface_native_coor,
                    rec_chains,
                    lig_chains,
                    native_rec_chains,
                    native_lig_chains,
                    cutoff=5.0,
                )
                if verbose:
                    print(f"Fnat: {fnat_list[0]:.3f}   Fnonnat: {fnonnat_list[0]:.3f}")
                
            except Exception as e:
                print(f"Error calculating native contacts: {str(e)}")
                fnat_list, fnonnat_list = [0.0], [0.0]
            
            # DockQ 점수 계산
            def scale_rms(rms, d):
                if rms is None:
                    return 0.0
                return 1.0 / (1 + (rms / d) ** 2)
            
            d1 = 8.5  # LRMS 스케일링 상수
            d2 = 1.5  # iRMS 스케일링 상수
            
            dockq_score = (fnat_list[0] + scale_rms(lrmsd_list[0], d1) + scale_rms(irmsd_list[0], d2)) / 3
            if verbose:
                print(f"DockQ Score: {dockq_score:.3f}")
            
            # 결과 저장
            results["Fnat"].append(fnat_list[0])
            results["Fnonnat"].append(fnonnat_list[0])
            results["rRMS"].append(rmsd_prot_list[0])
            results["iRMS"].append(irmsd_list[0])
            results["LRMS"].append(lrmsd_list[0])
            results["DockQ"].append(dockq_score)
            
        except Exception as e:
            if verbose:
                print(f"Error analyzing model {i}: {str(e)}")
                import traceback
                traceback.print_exc()
            # 오류 발생 시 None 값 추가
            results["Fnat"].append(None)
            results["Fnonnat"].append(None)
            results["rRMS"].append(None)
            results["iRMS"].append(None)
            results["LRMS"].append(None)
            results["DockQ"].append(None)
    
    # 결과를 데이터프레임에 추가
    for i, idx in enumerate(model_indices):
        for key in results:
            if i < len(results[key]):
                data_obj.df.at[idx, key] = results[key][i]
    
    if verbose:
        print("\nDockQ analysis completed successfully")
    
    return data_obj.df

def compute_piTM_pIS(coor, pae_array, cutoff=8.0):
    """
    AlphaFold2 PAE (Predicted Aligned Error)를 사용하여 
    복합체의 piTM (predicted interface TM-score) 및 
    pIS (partner interface score)를 계산합니다.

    piTM은 상호작용 인터페이스의 전반적인 품질을 평가하고,
    pIS는 각 체인이 인터페이스 형성에 기여하는 정도를 나타냅니다.

    Args:
        coor: 원자 좌표 정보를 담고 있는 Coor 객체 (pdb_numpy 라이브러리).
              CA 원자 ('name CA')를 선택할 수 있어야 합니다.
        pae_array (np.ndarray): (N, N) 형태의 PAE 행렬. N은 CA 원자 수.
        cutoff (float, optional): 인터페이스 잔기를 정의하기 위한 거리 임계값(Å). 
                                   기본값은 8.0Å.

    Returns:
        tuple: 다음 세 가지 리스트를 포함하는 튜플:
            - piTM_list (list[float]): 모델별 복합체 전역 piTM 점수.
            - per_chain_scores (list[list[float]]): 체인별 piTMₚ 점수.
                                                    [n_chain][n_model] 형태.
            - pIS_list (list[float]): 모델별 pIS 점수 (체인별 점수의 합).
    
    References:
        Gao, M., Nakajima An, D., Parks, J.M. & Skolnick, J. 
        AF2Complex predicts direct physical interactions in multimeric proteins 
        with deep learning. Nat Commun 13, 1744 (2022). 
        https://doi.org/10.1038/s41467-022-29394-2
        
        Feldman, J., & Skolnick, J. (2025). 
        AF3Complex yields improved structural predictions of protein complexes. 
        BioRxiv, 2025.02.27.640585. 
        https://doi.org/10.1101/2025.02.27.640585
        
        
    """
    models_CA     = coor.select_atoms("name CA")
    chains        = np.unique(models_CA.chain)
    
    # 입력 PAE 배열의 차원과 CA 원자 수가 일치하는지 확인
    if pae_array.shape != (models_CA.len, models_CA.len):
        raise ValueError(
            f"PAE array shape {pae_array.shape} mismatch "
            f"with CA atoms number ({models_CA.len}, {models_CA.len})"
        )

    piTM_list        = []
    per_chain_scores = [ [] for _ in chains ]
    pIS_list         = []

    # 각 모델에 대해 반복 (일반적으로 하나의 모델만 있음)
    for model_idx, model in enumerate(models_CA.models):
        # 1) 인터페이스 잔기 집합 𝓘 (Interface residues)
        iface_idx = set()
        for ch_A in chains:
            # chain ch_A에 속하면서 다른 체인(not ch_A)과 cutoff 이내에 있는 잔기
            # 또는 chain ch_A에 속하지 않으면서 chain ch_A와 cutoff 이내에 있는 잔기
            # 즉, 두 체인 간의 인터페이스에 있는 모든 잔기를 찾음
            within = model.get_index_select  # 원자 인덱스를 가져오는 함수
            
            # ch_A에 속하고, 다른 체인과 상호작용하는 잔기
            iface_idx.update(
                within(f"(chain {ch_A} and within {cutoff} of not chain {ch_A})")
            )
            # ch_A에 속하지 않고, ch_A와 상호작용하는 잔기 (대칭적 인터페이스 정의)
            # 이 부분은 위의 정의에 이미 포함될 수 있지만, 명확성을 위해 추가
            # 실제로는 첫 번째 update만으로도 충분할 수 있음
            # (A and within B) or (B and within A)와 유사
            iface_idx.update(
                within(f"(not chain {ch_A} and within {cutoff} of chain {ch_A})")
            )
            
        I = len(iface_idx) # 인터페이스 잔기의 총 개수
        
        # 인터페이스 잔기가 없으면 모든 점수를 0으로 처리
        if I == 0:
            piTM_list.append(0.0)
            for lst in per_chain_scores: lst.append(0.0)
            pIS_list.append(0.0)
            continue

        # d0: 인터페이스 크기 I에 따라 스케일링되는 거리 임계값
        d0 = 1.24 * (I - 15) ** (1/3) - 1.8 if I >= 22 else 0.02 * I

        # 2) 전체 πiTM 계산 (Global piTM)
        # 인터페이스 잔기 i에 대해, 다른 모든 인터페이스 잔기 j와의 PAE 기반 점수 합의 최대값을 찾음
        best_global_score = 0.0
        for i in iface_idx:
            # i번째 잔기와 다른 모든 인터페이스 잔기 j 사이의 점수 합
            current_sum_score = sum( 1 / (1 + (pae_array[i, j] / d0) ** 2) for j in iface_idx if i != j)
            # Gao et al. 2022 수식에서는 sum over j in I 이지만, i=j 경우는 pae_array[i,i]는 보통 0이고, 1/(1+0)=1이 됨
            # 원본 pdb_numpy 구현은 i!=j 조건을 명시. 여기서는 i=j일 때 pae_array[i,i]가 0이면 1을 더하게 됨.
            # 논문 수식을 더 정확히 따르려면 if i != j 조건 제거 또는 pae_array[i,i]를 어떻게 처리할지 명확히 해야함
            # 여기서는 원본 아이디어를 따라 모든 j에 대해 합산 (i=j 포함 시 pae_array[i,i]는 보통 0이므로 기여도 1)
            current_sum_score = sum( 1 / (1 + (pae_array[i, j] / d0) ** 2) for j in iface_idx )

            if current_sum_score > best_global_score: 
                best_global_score = current_sum_score
        
        piTM_val = best_global_score / I # 최대 점수를 인터페이스 크기로 정규화
        piTM_list.append(piTM_val)

        # 3) 체인별 πiTMₚ 계산 (Per-chain piTM)
        current_model_chain_scores = [] # 현재 모델의 각 체인별 점수 저장
        for chain_idx, ch_P in enumerate(chains): # 각 파트너 체인 P에 대해
            # I_p: 체인 P에 속하는 인터페이스 잔기 집합
            I_p = set(model.get_index_select(
                f"(chain {ch_P} and within {cutoff} of not chain {ch_P})"
            ))
            
            if not I_p: # 해당 체인이 인터페이스를 형성하지 않으면 점수는 0
                per_chain_scores[chain_idx].append(0.0)
                current_model_chain_scores.append(0.0)
                continue
            
            # Anchors: 체인 P에 속하지 *않는* 인터페이스 잔기 집합 (상대방 체인의 인터페이스 잔기들)
            anchors = iface_idx - I_p # 전체 인터페이스 잔기에서 체인 P의 인터페이스 잔기를 제외
            
            if not anchors: # 상대방 체인이 인터페이스를 형성하지 않으면 점수는 0
                per_chain_scores[chain_idx].append(0.0)
                current_model_chain_scores.append(0.0)
                continue

            best_per_chain_score = 0.0
            # 앵커 잔기 i (체인 P 바깥) 와 체인 P의 인터페이스 잔기 j (I_p 내부) 사이의 점수 합의 최대값
            for i in anchors:
                current_sum_score_p = sum( 1 / (1 + (pae_array[i, j] / d0) ** 2) for j in I_p )
                if current_sum_score_p > best_per_chain_score: 
                    best_per_chain_score = current_sum_score_p
            
            score_p = best_per_chain_score / I # 전체 인터페이스 크기 I로 정규화
            per_chain_scores[chain_idx].append(score_p)
            current_model_chain_scores.append(score_p)

        # 4) pIS 계산 = 현재 모델의 모든 체인별 점수의 합
        pIS_list.append(sum(current_model_chain_scores))

    return piTM_list, per_chain_scores, pIS_list