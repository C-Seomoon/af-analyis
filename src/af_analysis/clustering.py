import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import scipy.cluster.hierarchy  # import linkage, dendrogram, fcluster
import sklearn.manifold
import sklearn.decomposition
from MDAnalysis.analysis import align, diffusionmap, pca
from MDAnalysis.coordinates.memory import MemoryReader
import pandas as pd
import pdb_numpy
import logging
from collections import Counter
import os
import tempfile
from pdb_numpy.alignement import align_seq_based, rmsd_seq_based


# Autorship information
__author__ = "Alaa Reguei, Samuel Murail"
__copyright__ = "Copyright 2023, RPBS"
__credits__ = ["Samuel Murail", "Alaa Reguei"]
__license__ = "GNU General Public License version 2"
__version__ = "0.1.4"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"


"""
The module contains functions to cluster AlphaFold models.
"""

# Logging
logger = logging.getLogger(__name__)


def read_numerous_pdb(pdb_files, batch_size=1000):
    """Read a large number of PDB files in batches and combine
    them into a single MDAnalysis Universe.

    Discussed in:
    https://github.com/MDAnalysis/mdanalysis/issues/4590

    Parameters
    ----------
    pdb_files : list of str
        List of file paths to the PDB files to be read.
    batch_size : int, optional
        Number of PDB files to read in each batch. Default is 1000.

    Returns
    -------
    MDAnalysis.Universe
        A single MDAnalysis Universe containing the combined frames from all the PDB files.

    Notes
    -----
    - This function reads PDB files in batches to avoid memory issues.
    - Each batch of PDB files is loaded into a temporary Universe, and the positions of each frame
      are stored in a list.
    - The list of frames is then converted into a numpy array and used to create a new Universe with
      a MemoryReader, combining all the frames.

    Example
    -------
    >>> pdb_files = ['file1.pdb', 'file2.pdb', ...]
    >>> combined_universe = read_numerous_pdb(pdb_files, batch_size=1000)
    >>> print(combined_universe)
    """

    all_frames = []

    if pdb_files[0].endswith(".cif"):
        model = pdb_numpy.Coor(pdb_files[0])
        for file in pdb_files[1:]:
            local_model = pdb_numpy.Coor(file)
            model.models.append(local_model.models[0])
        model.write("tmp.pdb", overwrite=True)
        return mda.Universe("tmp.pdb", "tmp.pdb")

    for i in range(0, len(pdb_files), batch_size):
        # print(f"Reading frames {i:5} to {i+batch_size:5}, total : {len(pdb_files[i:i+batch_size])} frames")
        local_u = mda.Universe(pdb_files[0], pdb_files[i : i + batch_size])
        for ts in local_u.trajectory:
            all_frames.append(ts.positions.copy())
        del local_u

    # print("Convert to numpy")
    frames_array = np.array(all_frames)
    del all_frames

    # print(frames_array.shape)
    return mda.Universe(pdb_files[0], frames_array, format=MemoryReader, order="fac")


def scale(rms, d0=8.5):
    """Scaling RMS values.
    To avoid the problem of arbitrarily large RMS values that are essentially equally bad,
    this function scales RMS values using the inverse square scaling technique adapted
    from the S-score formula.

    [Sankar Basu, Björn Wallner https://doi.org/10.1371/journal.pone.0161879]

    Parameters
    ----------
    rms : float or numpy.ndarray
        RMS values as single values or matrix.

    d0 : float
        Scaling factor, fixed to 8.5 for LRMS.

    Returns
    -------
    rms_scale : float or numpy.ndarray
        Scaled RMS values as single values or matrix.
    """

    rms_scale = 1 / (1 + (rms / d0) ** 2)
    return rms_scale


def compute_distance_matrix(
    pdb_files,
    align_selection=None,
    distance_selection=None,
):
    """Compute distance matrix.

    This function computes the distance matrix of the different peptide chains.

    Parameters
    ----------
    pdb_files : list of str
        List of file paths to the PDB files to be read.
    align_selection : str, optional
        The selection string to align the protein chains (default is None or "backbone").
    distance_selection : str, optional
        The selection string to compute the distance matrix (default is None or "backbone").

    Returns
    -------
    None
        The function modifies the input DataFrame by appending the DistanceMatrix column, but does not return a value.
    """

    if align_selection is None and distance_selection is None:
        distance_selection = "backbone"
        align_selection = "backbone"
    elif align_selection is None:
        align_selection = distance_selection
    elif distance_selection is None:
        distance_selection = align_selection

    u = read_numerous_pdb(pdb_files)
    assert u.trajectory.n_frames == len(
        pdb_files
    ), f"Number of frames {u.trajectory.n_frames} different from number of files {len(files)}"

    logger.info("align structures")
    align.AlignTraj(u, u, select=align_selection, in_memory=True).run(verbose=True)

    logger.info("Compute distance Matrix")
    matrix = diffusionmap.DistanceMatrix(
        u,
        select=distance_selection,
    ).run(verbose=True)

    return matrix.results.dist_matrix


def hierarchical(
    df,
    threshold=2.0,
    align_selection=None,
    distance_selection=None,
    show_dendrogram=True,
    MDS_coors=True,
    rmsd_scale=False,
):
    """Clustering of AlphaFold models.

    After checking for the absence of missing values, the function starts by aligning the protein
    models using the `align_selection` variable or using `"backbone`" selection if not defined.

    Then, the function computes the distance matrix using the `distance_selection` variable or using
    the `align_selection` variable if not defined.

    The distance matrix can be scaled using the `scale` function if `rmsd_scale` is set to `True`.

    The function then computes the hierarchical clustering using the average linkage method and the
    distance matrix. The threshold value is used to cut the dendrogram and define the clusters.

    Optionally, this function can also plot the dendrogram of each PDB and the clusters distribution
    plot using the `clusters_distribution`.

    Multidimensional scaling coordinates can be computed from the distance matrix if `MDS_coors` is set
    to `True`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing AlphaFold data for the protein-peptide complex.
    threshold : float
        The value where to cut the dendrogram in order to define clusters.
    align_selection : str or dict, optional
        The selection string to align the protein chains (default is None).
    distance_selection : str or dict, optional
        The selection string to compute the distance matrix (default is None).
    show_dendrogram : bool, optional
        Whether to plot the different dendrograms of each PDB (default is `True`).
    MDS_coors : bool, optional
        Whether to compute Multidimensional scaling coordinates from the distance matrix (default is `True`).
    rmsd_scale : bool, optional
        Whether to scale the RMS values using the `scale` function (default is `False`).

    Returns
    -------
    None
        The function modifies the input DataFrame by appending the Cluster column, but does not return a value.
    """

    cluster_list = []
    MDS_1 = []
    MDS_2 = []

    query_list = df["query"].unique().tolist()

    if align_selection is None:
        align_selection = {pdb: "backbone" for pdb in query_list}
    elif isinstance(align_selection, str):
        align_selection = {pdb: align_selection for pdb in query_list}

    if distance_selection is None:
        distance_selection = {pdb: "backbone" for pdb in query_list}
    elif isinstance(distance_selection, str):
        distance_selection = {pdb: distance_selection for pdb in query_list}

    for pdb in query_list:
        files = [file for file in df[df["query"] == pdb]["pdb"] if not pd.isnull(file)]

        # Check that only the last df rows are missing
        null_number = sum(pd.isnull(df[df["query"] == pdb]["pdb"]))
        assert null_number == sum(
            pd.isnull(df[df["query"] == pdb]["pdb"][-null_number:])
        ), f"Missing pdb data in the middle of the query {pdb}"

        logger.info("Read all structures")
        dist_matrix = compute_distance_matrix(
            files,
            align_selection=align_selection[pdb],
            distance_selection=distance_selection[pdb],
        )

        logger.info(f"Max RMSD is {np.max(dist_matrix):.2f} A")
        if rmsd_scale:
            dist_matrix = 1 - scale(dist_matrix)

        logger.info("Compute Linkage clustering")
        h, _ = dist_matrix.shape
        Z = scipy.cluster.hierarchy.linkage(
            dist_matrix[np.triu_indices(h, 1)], method="average"
        )
        cluster_list += scipy.cluster.hierarchy.fcluster(
            Z, float(threshold), criterion="distance"
        ).tolist()

        logger.info(f"{len(np.unique(cluster_list))} clusters founded for {pdb}")

        # Add Multidimensional scaling coordinates from the distance matrix
        if MDS_coors:
            mds = sklearn.manifold.MDS(dissimilarity="precomputed", n_components=2)
            coordinates = mds.fit_transform(dist_matrix)
            MDS_1 += coordinates.T[0].tolist()
            MDS_2 += coordinates.T[1].tolist()

        # PCA analysis don't add any information:
        # pca = sklearn.decomposition.PCA(n_components=2)  # n_components defines how many principal components you want
        # pca_result = pca.fit_transform(coordinates)
        # coordinates = pca_result
        # print(pca_result)

        if show_dendrogram:
            # plot the dendrogram with the threshold line
            plt.figure()
            scipy.cluster.hierarchy.dendrogram(Z, color_threshold=float(threshold))
            plt.axhline(float(threshold), color="k", ls="--")
            plt.title("Hierarchical Cluster Dendrogram -{}".format(pdb))
            plt.xlabel("Data Point Indexes")
            plt.ylabel("Distance")
            plt.show()

    df["cluster"] = reorder_by_size(cluster_list) + null_number * [None]
    df["cluster"] = df["cluster"].astype("category")
    if MDS_coors:
        df["MDS 1"] = MDS_1 + null_number * [None]
        df["MDS 2"] = MDS_2 + null_number * [None]

    return


def reorder_by_size(clust_list):
    """Reorder clusters by size.

    This function reorders the clusters by size, with the largest cluster first.

    Parameters
    ----------
    clust_list : list
        List of clusters.

    Returns
    -------
    list
        The reordered list of clusters.
    """

    # Step 1: Count the occurrences of each cluster
    cluster_counts = Counter(clust_list)

    # Step 2: Sort clusters by frequency (most frequent first)
    sorted_clusters_by_frequency = [
        cluster for cluster, count in cluster_counts.most_common()
    ]

    # Step 3: Create a mapping from old cluster labels to new ones
    cluster_mapping = {
        old: new for new, old in enumerate(sorted_clusters_by_frequency, start=1)
    }
    # Treat None as a special case: it should be mapped to None
    if None in cluster_mapping:
        cluster_mapping[None] = None

    # Step 4: Relabel the clusters using the mapping
    reordered_clusters = [cluster_mapping[cluster] for cluster in clust_list]

    return reordered_clusters


def compute_chain_rmsd_flexible(pdb_files, align_chain='A', rmsd_chain='H', backbone_atoms=['N', 'CA', 'C', 'O']):
    """체인 선택에 유연한 RMSD 계산 함수"""
    
    # 다양한 선택 문법 시도
    def try_selection(universe, chain_id, atoms):
        selections = [
            f"chain {chain_id} and name {' '.join(atoms)}",  # 표준 체인
            f"segid {chain_id} and name {' '.join(atoms)}",  # 세그먼트 ID
            f"name {' '.join(atoms)}"  # 체인 무시, 이름만으로 선택
        ]
        
        for sel in selections:
            try:
                atoms = universe.select_atoms(sel)
                if atoms.n_atoms > 0:
                    logger.info(f"성공적인 선택: {sel}")
                    return atoms
            except:
                continue
        
        # 체인 무시하고 모든 백본 원자 반환
        return universe.select_atoms(f"name {' '.join(atoms)}")
    
    # CIF 파일을 PDB로 변환
    # ... (기존 코드와 동일)
    
    # RMSD 계산 시 수정된 원자 선택 방법 사용
    ref_align_atoms = try_selection(ref_u, align_chain, backbone_atoms)
    ref_rmsd_atoms = try_selection(ref_u, rmsd_chain, backbone_atoms)
    
    # ... (나머지 부분은 기존 코드와 유사)


def compute_chain_rmsd_pdb_numpy(pdb_files, align_chain='A', rmsd_chain='H', backbone_atoms=['CA', 'N', 'C', 'O']):
    """
    pdb_numpy를 사용하여 A 체인으로 정렬 후 H 체인의 RMSD를 계산하는 함수
    
    Parameters
    ----------
    pdb_files : list
        PDB 파일 경로 목록
    align_chain : str, optional
        정렬할 체인 ID, 기본값은 'A'
    rmsd_chain : str, optional
        RMSD를 계산할 체인 ID, 기본값은 'H'
    backbone_atoms : list, optional
        사용할 백본 원자 목록, 기본값은 ['CA', 'N', 'C', 'O']
        
    Returns
    -------
    numpy.ndarray
        RMSD 거리 행렬
    """
    import numpy as np
    import logging
    import pdb_numpy
    from pdb_numpy.alignement import align_seq_based, rmsd_seq_based
    
    logger = logging.getLogger(__name__)
    
    n_models = len(pdb_files)
    dist_matrix = np.zeros((n_models, n_models))
    
    # 모델 쌍별로 RMSD 계산
    for i in range(n_models):
        try:
            # 참조 모델 로드
            ref_coor = pdb_numpy.Coor(pdb_files[i])
            
            # 나머지 모델들과 RMSD 계산
            for j in range(i+1, n_models):
                try:
                    # 타겟 모델 로드
                    mobile_coor = pdb_numpy.Coor(pdb_files[j])
                    
                    # receptor 체인으로 정렬
                    _, _ = align_seq_based(
                        mobile_coor, 
                        ref_coor,
                        chain_1=[align_chain], 
                        chain_2=[align_chain],
                        back_names=backbone_atoms
                    )
                    
                    # ligand 체인의 RMSD 계산
                    rmsd_values, _ = rmsd_seq_based(
                        mobile_coor,
                        ref_coor,
                        chain_1=[rmsd_chain],
                        chain_2=[rmsd_chain],
                        back_names=backbone_atoms
                    )
                    
                    # 행렬에 저장 (대칭 값)
                    if rmsd_values and len(rmsd_values) > 0:
                        rmsd_value = rmsd_values[0]
                        dist_matrix[i, j] = rmsd_value
                        dist_matrix[j, i] = rmsd_value
                        
                except Exception as e:
                    logger.error(f"모델 {j+1} 처리 오류: {str(e)}")
                    
        except Exception as e:
            logger.error(f"참조 모델 {i+1} 로드 오류: {str(e)}")
    
    return dist_matrix

def calc_model_rmsd_scaled(model_rmsd, query_rmsd):
    """Scaling RMS values.
    To avoid the problem of arbitrarily large RMS values that are essentially equally bad,
    this function scales RMS values using the inverse square scaling technique adapted
    from the S-score formula.

    [Sankar Basu, Björn Wallner https://doi.org/10.1371/journal.pone.0161879]

    Parameters
    ----------
    rms : float or numpy.ndarray
        RMS values as single values or matrix.

    d0 : float
        Scaling factor, fixed to 8.5 for LRMS.

    Returns
    -------
    rms_scale : float or numpy.ndarray
        Scaled RMS values as single values or matrix.
    """
    

    model_rmsd_scale = 1 / (1 + (model_rmsd / query_rmsd) ** 2)
    return model_rmsd_scale