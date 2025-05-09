#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
import json
import pdb_numpy
import seaborn as sns
import matplotlib.pyplot as plt
from cmcrameri import cm
from tqdm.auto import tqdm
import json
import logging
import ipywidgets as widgets
import glob

from .format import colabfold_1_5, af3_webserver, afpulldown, boltz1, chai1, default, af3
from . import sequence, plot
from .analysis import get_pae, extract_fields_file


# Autorship information
__author__ = "Alaa Reguei, Samuel Murail"
__copyright__ = "Copyright 2023, RPBS"
__credits__ = ["Samuel Murail", "Alaa Reguei"]
__license__ = "GNU General Public License version 2"
__version__ = "0.1.4"
__maintainer__ = "Samuel Murail"
__email__ = "samuel.murail@u-paris.fr"
__status__ = "Beta"

# Logging
logger = logging.getLogger(__name__)


plddt_main_atom_list = [
    "CA",
    "P",
    "ZN",
    "MG",
    "CL",
    "CA",
    "NA",
    "MN",
    "K",
    "FE",
    "CU",
    "CO",
]


class Data:
    """Data class

    Parameters
    ----------
    dir : str
        Path to the directory containing the `log.txt` file.
    format : str
        Format of the data.
    df : pandas.DataFrame
        Dataframe containing the information extracted from the `log.txt` file.
    chains : dict
        Dictionary containing the chains of each query.
    chain_length : dict
        Dictionary containing the length of each chain of each query.

    Methods
    -------
    read_directory(directory, keep_recycles=False)
        Read a directory.
    export_csv(path)
        Export the dataframe to a csv file.
    import_tsv(path)
        Import a tsv file to the dataframe.
    add_json()
        Add json files to the dataframe.
    extract_data()
        Extract json/npz files to the dataframe.
    add_pdb()
        Add pdb files to the dataframe.
    add_fasta(csv)
        Add fasta sequence to the dataframe.
    keep_last_recycle()
        Keep only the last recycle for each query.
    plot_maxscore_as_col(score, col, hue='query')
        Plot the maxscore as a function of a column.
    plot_pae(index, cmap=cm.vik)
        Plot the PAE matrix.
    plot_plddt(index_list)
        Plot the pLDDT.
    show_3d(index)
        Show the 3D structure.
    plot_msa(filter_qid=0.15, filter_cov=0.4)
        Plot the msa from the a3m file.
    show_plot_info()
        Show the plot info.
    analyze_interfaces(self, idx=None, distance_threshold=8.0, verbose=True)
        Analyze interfaces in the data object.
    add_ranking_scores(self, ranking_file=None, verbose=True)
        Add ranking scores to the dataframe.
    prep_dockq(self, native_dir="/home/cseomoon/project/ABAG/AbNb_benchmark/Native_PDB", verbose=True)
        Prepare model paths and native structure paths for DockQ calculation.
    add_chain_rmsd(self, align_chain='A', rmsd_chain='H', backbone_atoms=['CA', 'N', 'C', 'O'])
        Add chain RMSD to the dataframe.
    add_rmsd_scale(self)
        Add model_rmsd_scale to the dataframe.
    analyze_chains(self, idx=None, verbose=True)
        Analyze chains in the data object.

    """

    def __init__(
        self, directory=None, data_dict=None, csv=None, verbose=True, format=None
    ):
        """ """

        if directory is not None:
            self.read_directory(directory, verbose=verbose, format=format)
        elif csv is not None:
            self.format = "csv"
            self.import_csv(csv)
        elif data_dict is not None:
            assert "pdb" in data_dict.keys()
            assert "query" in data_dict.keys()
            assert "data_file" in data_dict.keys()

            self.df = pd.DataFrame(data_dict)
            self.dir = None
            self.df["format"] = "custom"
            self.set_chain_length()

    def read_directory(self, directory, keep_recycles=False, verbose=True, format=None):
        """Read a directory.

        If the directory contains a `log.txt` file, the format is set to `colabfold_1.5`.

        Parameters
        ----------
        directory : str
            Path to the directory containing the `log.txt` file.
        keep_recycles : bool
            Keep only the last recycle for each query.
        verbose : bool
            Print information about the directory.

        Returns
        -------
        None
        """
        self.dir = directory

        if format == "colabfold_1.5" or os.path.isfile(
            os.path.join(directory, "log.txt")
        ):
            self.format = "colabfold_1.5"
            self.df = colabfold_1_5.read_log(directory, keep_recycles)
            self.df["format"] = "colabfold_1.5"
            self.add_pdb(verbose=verbose)
            self.add_json(verbose=verbose)
        elif format == "AF3_webserver" or os.path.isfile(
            os.path.join(directory, "terms_of_use.md")
        ):
            self.format = "AF3_webserver"
            self.df = af3_webserver.read_dir(directory)
            self.df["format"] = "AF3_webserver"
        elif format == "af3" or format == "AF3" or os.path.isfile(
            os.path.join(directory, "TERMS_OF_USE.md")
        ):
            self.format = "AF3"
            self.df = af3.read_dir(directory)
            self.df["format"] = "AF3"
        elif format == "AlphaPulldown" or os.path.isfile(
            os.path.join(directory, "ranking_debug.json")
        ):
            self.format = "AlphaPulldown"
            self.df = afpulldown.read_dir(directory)
            self.df["format"] = "AlphaPulldown"
        elif format == "boltz1" or (
            os.path.isdir(os.path.join(directory, "predictions"))
        ):
            self.format = "boltz1"
            self.df = boltz1.read_dir(directory)
            self.df["format"] = "boltz1"
        elif format == "chai1" or os.path.isfile(
            os.path.join(directory, "msa_depth.pdf")
        ):
            self.format = "chai1"
            self.df = chai1.read_dir(directory)
            self.df["format"] = "chai1"
        else:
            self.format = "default"
            self.df = default.read_dir(directory)
            self.df["format"] = "default"
            self.add_json()

        self.set_chain_length()

    def set_chain_length(self):
        """Find chain information from the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.chains = {}
        self.chain_length = {}
        for querie in self.df["query"].unique():
            # print(querie, self.df[self.df['query'] == querie])
            first_model = pdb_numpy.Coor(
                self.df[self.df["query"] == querie].iloc[0]["pdb"]
            )
            self.chains[querie] = list(np.unique(first_model.models[0].chain))
            self.chain_length[querie] = [
                len(
                    np.unique(
                        first_model.models[0].uniq_resid[
                            first_model.models[0].chain == chain
                        ]
                    )
                )
                for chain in self.chains[querie]
            ]

    def export_file(self, path, format='csv'):
        """Export the dataframe to a CSV or TSV file.

        Parameters
        ----------
        path : str
            Path to the output file.
        format : str, optional
            File format, either 'csv' or 'tsv' (default: 'csv').

        Returns
        -------
        None
        """
        if format.lower() == 'tsv':
            separator = '\t'
        else:  # default to csv
            separator = ','
        
        self.df.to_csv(path, index=False, sep=separator)

    def import_file(self, path, format=None):
        """Import a CSV or TSV file to the dataframe.

        Parameters
        ----------
        path : str
            Path to the input file.
        format : str, optional
            File format, either 'csv', 'tsv', or None (auto-detect from file extension).

        Returns
        -------
        None
        """
        # 파일 형식 자동 감지 (확장자 기반)
        if format is None:
            if path.lower().endswith('.tsv'):
                separator = '\t'
            else:  # default to csv
                separator = ','
        else:
            separator = '\t' if format.lower() == 'tsv' else ','
        
        self.df = pd.read_csv(path, sep=separator)
        self.dir = os.path.dirname(self.df["pdb"][0])

        # 체인 정보 초기화
        self.chains = {}
        self.chain_length = {}
        
        # 각 쿼리에 대한 체인 정보 설정
        for querie in self.df["query"].unique():
            first_model = pdb_numpy.Coor(
                self.df[self.df["query"] == querie].iloc[0]["pdb"]
            )
            self.chains[querie] = list(np.unique(first_model.models[0].chain))
            self.chain_length[querie] = [
                len(
                    np.unique(
                        first_model.models[0].uniq_resid[
                            first_model.models[0].chain == chain
                        ]
                    )
                )
                for chain in self.chains[querie]
            ]

    def add_json(self, verbose=True):
        """Add json files to the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.format == "colabfold_1.5":
            colabfold_1_5.add_json(self.df, self.dir, verbose=verbose)
        elif self.format == "default":
            default.add_json(self.df, self.dir)

    def extract_data(self):
        """Extract json/npz files to the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        index_list = []
        data_list = []
        for index, data_path in zip(self.df.index, self.df["data_file"]):
            if data_path is not None:
                if data_path.endswith(".json"):
                    with open(data_path, "r") as f:
                        data = json.load(f)
                    data_list.append(data)
                    index_list.append(index)
                elif data_path.endswith(".npz"):
                    data_npz = np.load(data_path)
                    data = {}
                    for key in data_npz.keys():
                        data[key] = data_npz[key]
                    data_list.append(data)
                    index_list.append(index)

        new_column = {}
        for key in data_list[0].keys():
            new_column[key] = []
        for data in data_list:
            for key in data.keys():
                new_column[key].append(data[key])

        for key in new_column.keys():
            self.df.loc[:, key] = None
            self.df.loc[index_list, key] = pd.Series(new_column[key], index=index_list)

    def extract_fields(self, fields, disable=False):
        """Extract fields from data files to the dataframe.

        Parameters
        ----------
        fields : list
            List of fields to extract.
        disable : bool
            Disable the progress bar.

        Returns
        -------
        None
        """

        values_list = []
        for field in fields:
            values_list.append([])
        for data_path in tqdm(
            self.df["data_file"], total=len(self.df["data_file"]), disable=disable
        ):
            if data_path is not None:
                local_values = extract_fields_file(data_path, fields)

                for i in range(len(fields)):
                    values_list[i].append(local_values[i])

            else:
                for i in range(len(fields)):
                    values_list[i].append(None)

        for i, field in enumerate(fields):
            self.df[field] = None
            new_col = pd.Series(values_list[i])
            self.df.loc[:, field] = new_col

    def add_pdb(self, verbose=True):
        """Add pdb files to the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.format == "colabfold_1.5":
            colabfold_1_5.add_pdb(self.df, self.dir, verbose=verbose)

    def add_fasta(self, csv):
        """Add fasta sequence to the dataframe.

        Parameters
        ----------
        csv : str
            Path to the csv file containing the fasta sequence.

        Returns
        -------
        None
        """

        if self.format == "colabfold_1.5":
            colabfold_1_5.add_fasta(self.df, csv)

    def keep_last_recycle(self):
        """Keep only the last recycle for each query."""

        idx = (
            self.df.groupby(["query", "seed", "model", "weight"])["recycle"].transform(
                "max"
            )
            == self.df["recycle"]
        )
        self.df = self.df[idx]

    def plot_maxscore_as_col(self, score, col, hue="query"):
        col_list = self.df[col].unique()
        query_list = self.df[hue].unique()
        # print(col_list)
        # print(query_list)

        out_list = []

        for query in query_list:
            # print(query)
            query_pd = self.df[self.df[hue] == query]

            for column in col_list:
                # print(column)
                # ~print()

                col_pd = query_pd[query_pd[col] <= column]
                # print(col_pd[score])
                # print(column, len(col_pd))
                # print(col, col_pd.columns)

                if len(col_pd) > 0:
                    out_list.append(
                        {hue: query, score: col_pd[score].max(), col: column}
                    )
                    # print(column, len(col_pd), col_pd[score].max())

        max_pd = pd.DataFrame(out_list)

        fig, ax = plt.subplots()
        sns.lineplot(max_pd, x=col, y=score, hue=hue)

        return (fig, ax)

    def plot_pae(self, index, cmap=cm.vik):
        row = self.df.iloc[index]

        if row["data_file"] is None:
            return None
        pae_array = get_pae(row["data_file"])

        fig, ax = plt.subplots()
        res_max = sum(self.chain_length[row["query"]])

        img = ax.imshow(
            pae_array,
            cmap=cmap,
            vmin=0.0,
            vmax=30.0,
        )

        plt.hlines(
            np.cumsum(self.chain_length[row["query"]][:-1]) - 0.5,
            xmin=-0.5,
            xmax=res_max,
            colors="black",
        )

        plt.vlines(
            np.cumsum(self.chain_length[row["query"]][:-1]) - 0.5,
            ymin=-0.5,
            ymax=res_max,
            colors="black",
        )

        plt.xlim(-0.5, res_max - 0.5)
        plt.ylim(res_max - 0.5, -0.5)
        chain_pos = []
        len_sum = 0
        for longueur in self.chain_length[row["query"]]:
            chain_pos.append(len_sum + longueur / 2)
            len_sum += longueur

        ax.set_yticks(chain_pos)
        ax.set_yticklabels(self.chains[row["query"]])
        cbar = plt.colorbar(img)
        cbar.set_label("Predicted Aligned Error (Å)", rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

        return (fig, ax)

    def get_plddt(self, index):
        """Extract the pLDDT array either from the pdb file or form the
        json/plddt files.

        Parameters
        ----------
        index : int
            Index of the dataframe.

        Returns
        -------
        np.array
            pLDDT array.
        """

        row = self.df.iloc[index]

        if row["format"] in ["AF3_webserver", "csv", "AlphaPulldown", "af3"]:
            model = pdb_numpy.Coor(row["pdb"])
            plddt_array = model.models[0].beta[
                np.isin(model.models[0].name, plddt_main_atom_list)
            ]
            return plddt_array

        if row["format"] in ["boltz1"]:
            data_npz = np.load(row["plddt"])
            plddt_array = data_npz["plddt"]
            return plddt_array * 100

        if row["data_file"] is None:
            return None
        elif row["data_file"].endswith(".json"):
            with open(row["data_file"]) as f:
                local_json = json.load(f)

            if "plddt" in local_json:
                plddt_array = np.array(local_json["plddt"])
            elif "atom_plddts" in local_json:
                plddt_array = np.array(local_json["atom_plddts"])
            else:
                return None
        elif row["data_file"].endswith(".npz"):
            data_npz = np.load(row["data_file"])
            if "plddt" in data_npz:
                plddt_array = data_npz["plddt"]
            else:
                return None

        return plddt_array

    def plot_plddt(self, index_list=None):
        if index_list is None:
            index_list = range(len(self.df))

        fig, ax = plt.subplots()

        for index in index_list:
            plddt_array = self.get_plddt(index)
            plt.plot(np.arange(1, len(plddt_array) + 1), plddt_array)

        plt.vlines(
            np.cumsum(self.chain_length[self.df.iloc[index_list[0]]["query"]][:-1]),
            ymin=0,
            ymax=100.0,
            colors="black",
        )
        plt.ylim(0, 100)
        plt.xlim(0, sum(self.chain_length[self.df.iloc[index_list[0]]["query"]]))
        plt.xlabel("Residue")
        plt.ylabel("predicted LDDT")

        return (fig, ax)

    def show_3d(self, index):
        row = self.df.iloc[index]

        if row["pdb"] is None:
            return (None, None)

        import nglview as nv

        # Bug with show_file
        # view = nv.show_file(row['pdb'])
        view = nv.show_structure_file(row["pdb"])
        # view.add_component(ref_coor[0])
        # view.clear_representations(1)
        # view[1].add_cartoon(selection="protein", color='blue')
        # view[1].add_licorice(selection=":A", color='blue')
        # view[0].add_licorice(selection=":A")
        return view

    def plot_msa(self, filter_qid=0.15, filter_cov=0.4):
        """
        Plot the msa from the a3m file.

        Parameters
        ----------
        filter_qid : float
            Minimal sequence identity to keep a sequence.
        filter_cov : float
            Minimal coverage to keep a sequence.

        Returns
        -------
        None

        ..Warning only tested with colabfold 1.5
        """

        raw_list = os.listdir(self.dir)
        file_list = []
        for file in raw_list:
            if file.endswith(".a3m"):
                file_list.append(file)

        for a3m_file in file_list:
            logger.info(f"Reading MSA file:{a3m_file}")
            querie = a3m_file.split("/")[-1].split(".")[0]

            a3m_lines = open(os.path.join(self.dir, a3m_file), "r").readlines()[1:]
            seqs, mtx, nams = sequence.parse_a3m(
                a3m_lines=a3m_lines, filter_qid=filter_qid, filter_cov=filter_cov
            )
            logger.info(f"- Keeping {len(seqs):6} sequences for plotting.")
            feature_dict = {}
            feature_dict["msa"] = sequence.convert_aa_msa(seqs)
            feature_dict["num_alignments"] = len(seqs)

            if len(seqs) == sum(self.chain_length[querie]):
                feature_dict["asym_id"] = []
                for i, chain_len in enumerate(self.chain_length[querie]):
                    feature_dict["asym_id"] += [i + 1.0] * chain_len
                feature_dict["asym_id"] = np.array(feature_dict["asym_id"])

            fig = plot.plot_msa_v2(feature_dict)
            plt.show()

    def count_msa_seq(self):
        """
        Count for each chain the number of sequences in the MSA.

        Parameters
        ----------
        None

        Returns
        -------
        None

        ..Warning only tested with colabfold 1.5
        """

        raw_list = os.listdir(self.dir)
        file_list = []
        for file in raw_list:
            if file.endswith(".a3m"):
                file_list.append(file)

        alignement_len = {}

        for a3m_file in file_list:
            logger.info(f"Reading MSA file:{a3m_file}")
            querie = a3m_file.split("/")[-1].split(".")[0]

            a3m_lines = open(os.path.join(self.dir, a3m_file), "r").readlines()[1:]
            seqs, mtx, nams = sequence.parse_a3m(
                a3m_lines=a3m_lines, filter_qid=0, filter_cov=0
            )
            feature_dict = {}
            feature_dict["msa"] = sequence.convert_aa_msa(seqs)
            feature_dict["num_alignments"] = len(seqs)

            seq_dict = {}
            for chain in self.chains[querie]:
                seq_dict[chain] = 0

            chain_len_list = self.chain_length[querie]
            chain_list = self.chains[querie]
            seq_len = sum(chain_len_list)

            # Treat the cases of homomers
            # I compare the length of each sequence with the other ones
            # It is wrong and should be FIXED
            # The original sequence should be retrieved from eg. the pdb file
            if len(seqs[0]) != seq_len:
                new_chain_len = []
                new_chain_list = []
                for i, seq_len in enumerate(chain_len_list):
                    if seq_len not in chain_len_list[:i]:
                        new_chain_len.append(seq_len)
                        new_chain_list.append(chain_list[i])

                chain_len_list = new_chain_len
                chain_list = new_chain_list
                seq_len = sum(chain_len_list)

            assert (
                len(seqs[0]) == seq_len
            ), f"len(seqs[0])={len(seqs[0])} != seq_len={seq_len}"

            for seq in seqs:
                start = 0
                for i, num in enumerate(chain_len_list):
                    gap_num = seq[start : start + num].count("-")
                    if gap_num < num:
                        seq_dict[chain_list[i]] += 1
                    start += num

            alignement_len[
                querie
            ] = seq_dict  # [seq_dict[chain] for chain in self.chains[querie]]
        return alignement_len

    def show_plot_info(self, cmap=cm.vik):
        """
        Need to solve the issue with:

        ```
        %matplotlib ipympl
        ```

        plots don´t update when changing the model number.

        """

        model_widget = widgets.IntSlider(
            value=1,
            min=1,
            max=len(self.df),
            step=1,
            description="model:",
            disabled=False,
        )
        display(model_widget)

        def show_model(rank_num):
            fig, (ax_plddt, ax_pae) = plt.subplots(1, 2, figsize=(10, 4))
            plddt_array = self.get_plddt(rank_num - 1)
            (plddt_plot,) = ax_plddt.plot(plddt_array)
            query = self.df.iloc[model_widget.value - 1]["query"]
            data_file = self.df.iloc[model_widget.value - 1]["data_file"]
            ax_plddt.vlines(
                np.cumsum(self.chain_length[query][:-1]),
                ymin=0,
                ymax=100.0,
                colors="black",
            )
            ax_plddt.set_ylim(0, 100)
            res_max = sum(self.chain_length[query])
            ax_plddt.set_xlim(0, res_max)
            ax_plddt.set_xlabel("Residue")
            ax_plddt.set_ylabel("predicted LDDT")

            pae_array = get_pae(data_file)
            ax_pae.imshow(
                pae_array,
                cmap=cmap,
                vmin=0.0,
                vmax=30.0,
            )
            ax_pae.vlines(
                np.cumsum(self.chain_length[query][:-1]),
                ymin=-0.5,
                ymax=res_max,
                colors="yellow",
            )
            ax_pae.hlines(
                np.cumsum(self.chain_length[query][:-1]),
                xmin=-0.5,
                xmax=res_max,
                colors="yellow",
            )
            ax_pae.set_xlim(-0.5, res_max - 0.5)
            ax_pae.set_ylim(res_max - 0.5, -0.5)
            chain_pos = []
            len_sum = 0
            for longueur in self.chain_length[query]:
                chain_pos.append(len_sum + longueur / 2)
                len_sum += longueur
            ax_pae.set_yticks(chain_pos)
            ax_pae.set_yticklabels(self.chains[query])
            plt.show(fig)

        output = widgets.Output(layout={"width": "95%"})
        display(output)

        with output:
            show_model(model_widget.value)
            # logger.info(results['metric'][0][rank_num - 1]['print_line'])

        def on_value_change(change):
            output.clear_output()
            with output:
                show_model(model_widget.value)

        model_widget.observe(on_value_change, names="value")

    def analyze_interfaces(self, idx=None, distance_threshold=8.0, verbose=True):
        """
        Data 객체 내 모델들의 인터페이스를 분석하고 결과를 현재 DataFrame에 추가합니다.
        
        Parameters
        ----------
        idx : int or list, optional
            분석할 모델의 인덱스. None이면 모든 모델 분석
        distance_threshold : float
            접촉으로 간주할 거리 임계값(Å)
        verbose : bool, optional
            진행 상태를 출력할지 여부, 기본값은 True
            
        Returns
        -------
        self : Data
            업데이트된 Data 객체 (메서드 체이닝 지원)
        """
        import numpy as np
        import pandas as pd
        import json
        from collections import Counter
        from scipy.spatial.distance import pdist, squareform
        from af_analysis.analysis import get_pae
        import pdb_numpy
        
        # 결과 저장 리스트
        results = []
        
        # 분석할 인덱스 목록 설정
        if idx is None:
            indices = range(len(self.df))
        elif isinstance(idx, (list, tuple)):
            indices = idx
        else:
            indices = [idx]
        
        for i in indices:
            try:
                # 모델 및 신뢰도 데이터 로드
                row = self.df.iloc[i]
                model_path = row["pdb"]
                json_path = row["data_file"]
                
                if verbose:
                    print(f"Analyzing model {i}: {model_path}")
                
                # JSON 파일에서 신뢰도 데이터 로드
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                # atom pLDDT 및 체인 정보 추출
                atom_plddts = np.array(json_data['atom_plddts'])
                atom_plddts = np.nan_to_num(atom_plddts)  # NaN 값 처리
                
                atom_chain_ids = json_data['atom_chain_ids']  # 각 원자의 체인 ID
                unique_chains = sorted(list(set(atom_chain_ids)))
                
                # PAE 데이터 추출
                pae_matrix = np.array(json_data['pae'])
                pae_matrix = np.nan_to_num(pae_matrix)
                
                if verbose:
                    print(f"Loaded pLDDT data for {len(atom_plddts)} atoms")
                    print(f"Chains identified: {unique_chains}")
                    print(f"PAE matrix shape: {pae_matrix.shape}")
                
                # CIF 파일 파싱하여 원자 정보 추출
                model = pdb_numpy.Coor(model_path)
                atoms = model.models[0]
                
                # Heavy atom 선택 (수소 제외)
                atom_mask = np.array([not name.startswith('H') for name in atoms.name])
                heavy_atoms = np.where(atom_mask)[0]
                
                if verbose:
                    print(f"Selected {len(heavy_atoms)} heavy atoms out of {len(atoms.name)} total atoms")
                
                # 원자 좌표 및 정보
                atom_chains = np.array(atoms.chain)[heavy_atoms]
                atom_resids = np.array(atoms.resid)[heavy_atoms]
                atom_coords = np.column_stack((
                    np.array(atoms.x)[heavy_atoms],
                    np.array(atoms.y)[heavy_atoms], 
                    np.array(atoms.z)[heavy_atoms]
                ))
                
                # 모델의 유니크한 체인 식별 (CIF 파일 기준)
                model_chains = np.unique(atom_chains)
                
                # 원자 간 거리 계산
                if verbose:
                    print("Calculating atom distances...")
                
                distances = squareform(pdist(atom_coords))
                
                # 결과 저장 딕셔너리 (인덱스를 키로 사용)
                model_result = {}
                
                # 인터페이스 pLDDT와 PAE 저장을 위한 딕셔너리
                interface_plddt_pairs = {}
                interface_pae_pairs = {}
                total_contacts = 0
                
                # 체인 쌍 분석
                for idx1, chain1 in enumerate(model_chains):
                    for chain2 in model_chains[idx1+1:]:
                        pair_id = f"{chain1}{chain2}"
                        if verbose:
                            print(f"Analyzing chain pair {pair_id}")
                        
                        # 체인별 원자 마스크 및 인덱스
                        chain1_mask = atom_chains == chain1
                        chain2_mask = atom_chains == chain2
                        chain1_indices = np.where(chain1_mask)[0]
                        chain2_indices = np.where(chain2_mask)[0]
                        
                        if verbose:
                            print(f"Chain {chain1}: {len(chain1_indices)} atoms, Chain {chain2}: {len(chain2_indices)} atoms")
                        
                        # 인터페이스 원자 및 잔기 식별
                        interface_atoms1 = []  # 체인1의 인터페이스 원자
                        interface_atoms2 = []  # 체인2의 인터페이스 원자
                        interface_residues = set()  # 인터페이스 잔기 (체인, 잔기번호)
                        contacts = 0
                        
                        # 체인 간 거리 계산 및 접촉 확인
                        for idx1 in chain1_indices:
                            for idx2 in chain2_indices:
                                if distances[idx1, idx2] < distance_threshold:
                                    contacts += 1
                                    # 원자 인덱스 (CIF 파일 내)
                                    atom1_idx = heavy_atoms[idx1]
                                    atom2_idx = heavy_atoms[idx2]
                                    
                                    interface_atoms1.append(atom1_idx)
                                    interface_atoms2.append(atom2_idx)
                                    
                                    # 인터페이스 잔기 추가
                                    interface_residues.add((atom_chains[idx1], atom_resids[idx1]))
                                    interface_residues.add((atom_chains[idx2], atom_resids[idx2]))
                        
                        total_contacts += contacts
                        model_result[f'contacts_{pair_id}'] = contacts
                        
                        if verbose:
                            print(f"Found {contacts} atom contacts between chains {chain1} and {chain2}")
                            print(f"Interface atoms: {len(interface_atoms1) + len(interface_atoms2)}, Interface residues: {len(interface_residues)}")
                        
                        # 접촉이 있는 경우만 처리
                        if contacts > 0:
                            # pLDDT 계산
                            interface_plddt_values = []
                            for atom_idx in interface_atoms1 + interface_atoms2:
                                if atom_idx < len(atom_plddts):
                                    interface_plddt_values.append(atom_plddts[atom_idx])
                            
                            if interface_plddt_values:
                                avg_plddt = float(np.mean(interface_plddt_values))
                                interface_plddt_pairs[pair_id] = avg_plddt
                                model_result[f'interface_plddt_{pair_id}'] = avg_plddt
                                if verbose:
                                    print(f"Interface pLDDT for {pair_id}: {avg_plddt:.2f}")
                            
                            # PAE 계산
                            try:
                                # 체인별 잔기 매핑 생성
                                chain_residue_map = {}
                                for chain in model_chains:
                                    chain_residue_map[chain] = sorted(np.unique(atom_resids[atom_chains == chain]))
                                
                                # PAE 매트릭스 인덱스 계산을 위한 정보
                                residue_positions = {}
                                position = 0
                                for chain in sorted(model_chains):
                                    residue_positions[chain] = {}
                                    for res in chain_residue_map[chain]:
                                        residue_positions[chain][res] = position
                                        position += 1
                                
                                # 인터페이스 잔기 쌍의 PAE 값 추출
                                interface_pae_values = []
                                
                                chain1_residues = [r[1] for r in interface_residues if r[0] == chain1]
                                chain2_residues = [r[1] for r in interface_residues if r[0] == chain2]
                                
                                for res1 in chain1_residues:
                                    for res2 in chain2_residues:
                                        if res1 in residue_positions[chain1] and res2 in residue_positions[chain2]:
                                            idx1 = residue_positions[chain1][res1]
                                            idx2 = residue_positions[chain2][res2]
                                            
                                            if idx1 < pae_matrix.shape[0] and idx2 < pae_matrix.shape[1]:
                                                interface_pae_values.append(pae_matrix[idx1, idx2])
                                
                                if interface_pae_values:
                                    avg_pae = float(np.mean(interface_pae_values))
                                    interface_pae_pairs[pair_id] = avg_pae
                                    model_result[f'interface_pae_{pair_id}'] = avg_pae
                                    if verbose:
                                        print(f"Interface PAE for {pair_id}: {avg_pae:.2f}")
                            except Exception as e:
                                if verbose:
                                    print(f"Error calculating PAE for {pair_id}: {e}")
                
                # 전체 인터페이스 정보 추가
                model_result['total_contacts'] = total_contacts
                
                # 평균값 계산
                if interface_plddt_pairs:
                    model_result['avg_interface_plddt'] = float(np.mean(list(interface_plddt_pairs.values())))
                if interface_pae_pairs:
                    model_result['avg_interface_pae'] = float(np.mean(list(interface_pae_pairs.values())))
                
                # 결과 저장 (인덱스로 저장)
                results.append((i, model_result))
                
            except Exception as e:
                if verbose:
                    print(f"Error analyzing model {i}: {str(e)}")
        
        # 결과를 기존 DataFrame에 추가
        if results:
            for idx, result_dict in results:
                for key, value in result_dict.items():
                    self.df.at[idx, key] = value
        
        # 자기 자신을 반환 (메서드 체이닝을 위해)
        return self

    def add_ranking_scores(self, ranking_file=None, verbose=True):
        """
        AF3 ranking_scores.csv 파일을 읽고 Data 객체의 DataFrame을 업데이트합니다.
        ranking_confidence 컬럼이 있는 경우 ranking_score 값으로 대체합니다.
        
        Parameters
        ----------
        ranking_file : str, optional
            ranking_scores.csv 파일 경로. None인 경우 기본 위치에서 찾습니다.
        verbose : bool, optional
            진행 정보를 출력할지 여부, 기본값은 True
            
        Returns
        -------
        self : Data
            업데이트된 Data 객체 (메서드 체이닝 지원)
        """
        # AF3 형식인지 확인
        if self.format != "AF3":
            if verbose:
                print(f"Warning: This function is designed for AF3 format, current format is {self.format}")
                print("No update performed.")
            return self
        
        # 파일 경로 결정
        if ranking_file is None:
            ranking_file = os.path.join(self.dir, "ranking_scores.csv")
        
        # 파일이 존재하는지 확인
        if not os.path.exists(ranking_file):
            if verbose:
                print(f"Warning: Ranking file not found at {ranking_file}")
                print("No update performed.")
            return self
        
        try:
            # ranking_scores.csv 파일 로드
            if verbose:
                print(f"Loading ranking scores from {ranking_file}")
            ranking_df = pd.read_csv(ranking_file)
            
            # 필요한 컬럼이 있는지 확인
            required_columns = ["seed", "sample", "ranking_score"]
            missing_columns = [col for col in required_columns if col not in ranking_df.columns]
            if missing_columns:
                if verbose:
                    print(f"Warning: Missing columns in ranking file: {', '.join(missing_columns)}")
                    print("No update performed.")
                return self
            
            # 순위 점수를 기존 DataFrame에 추가
            # seed와 sample 컬럼이 없으면 추가
            if "seed" not in self.df.columns:
                if verbose:
                    print("Warning: 'seed' column not found in DataFrame. Trying to extract from other columns...")
                # seed 정보를 다른 컬럼에서 추출 시도 (예: model 컬럼)
                try:
                    self.df["seed"] = self.df["model"].apply(lambda x: int(x.split("_")[0]) if isinstance(x, str) else None)
                except:
                    if verbose:
                        print("Failed to extract 'seed' information. Adding empty 'seed' column.")
                    self.df["seed"] = None
            
            if "sample" not in self.df.columns:
                if verbose:
                    print("Warning: 'sample' column not found in DataFrame. Trying to extract from other columns...")
                # sample 정보를 다른 컬럼에서 추출 시도 (예: model 컬럼)
                try:
                    self.df["sample"] = self.df["model"].apply(lambda x: int(x.split("_")[1]) if isinstance(x, str) and len(x.split("_")) > 1 else None)
                except:
                    if verbose:
                        print("Failed to extract 'sample' information. Adding empty 'sample' column.")
                    self.df["sample"] = None
            
            # ranking_confidence 컬럼 확인 및 처리
            has_confidence = "ranking_confidence" in self.df.columns
            if has_confidence:
                if verbose:
                    print("Found 'ranking_confidence' column. Values will be replaced with 'ranking_score'.")
            else:
                # ranking_score 컬럼 추가 (없는 경우)
                if "ranking_score" not in self.df.columns:
                    self.df["ranking_score"] = None
            
            # 업데이트 수행
            updated_rows = 0
            for _, row in ranking_df.iterrows():
                matching_rows = (self.df["seed"] == row["seed"]) & (self.df["sample"] == row["sample"])
                update_count = sum(matching_rows)
                
                if update_count > 0:
                    # ranking_confidence 컬럼이 있으면 해당 컬럼 값 업데이트
                    if has_confidence:
                        self.df.loc[matching_rows, "ranking_confidence"] = row["ranking_score"]
                    # ranking_score 컬럼 업데이트
                    else:
                        self.df.loc[matching_rows, "ranking_score"] = row["ranking_score"]
                    
                    updated_rows += update_count
            
            if verbose:
                if has_confidence:
                    print(f"Updated {updated_rows} rows in 'ranking_confidence' with ranking scores.")
                else:
                    print(f"Updated {updated_rows} rows in 'ranking_score' with ranking scores.")
            
        except Exception as e:
            if verbose:
                print(f"Error loading ranking scores: {str(e)}")
        
        return self

    def prep_dockq(self, native_dir="/home/cseomoon/project/ABAG/AbNb_benchmark/Native_PDB", verbose=True):
        """
        Prepare model paths and native structure paths for DockQ calculation.
        
        This function adds two new columns to the DataFrame:
        1. model_path: the path to the pdb file, modified to PDBs/model.pdb
        2. native_path: the path to the PDB file in the native_dir that matches the query prefix
        
        Parameters
        ----------
        native_dir : str, optional
            The directory where native PDB files are stored
        verbose : bool, optional
            Whether to print progress information, default is True
        
        Returns
        -------
        self : Data
            Updated Data object (method chaining support)
        """
        # model_path 컬럼 생성
        self.df['model_path'] = self.df['pdb'].apply(
            lambda x: os.path.join(os.path.dirname(x), 'PDBs', 'model.pdb')
        )
        
        # 해당 디렉토리의 모든 PDB 파일 목록 가져오기
        all_native_files = glob.glob(os.path.join(native_dir, "*.pdb"))
        
        # native_path 컬럼 생성
        native_paths = []
        missing_native = []
        
        for _, row in self.df.iterrows():
            query_prefix = row['query'][:4]  # query 값의 앞 4글자 추출
            
            # query 접두사와 일치하는 파일 찾기
            matching_files = [f for f in all_native_files if query_prefix in os.path.basename(f)]
            
            if matching_files:
                native_paths.append(matching_files[0])  # 첫 번째 일치하는 파일 사용
                if verbose and len(matching_files) > 1:
                    print(f"Multiple matches found for {query_prefix}, using {os.path.basename(matching_files[0])}")
            else:
                native_paths.append(None)
                missing_native.append(query_prefix)
        
        self.df['native_path'] = native_paths
        
        # 결과 요약
        if verbose:
            found_count = len(native_paths) - native_paths.count(None)
            print(f"Found native structures for {found_count} of {len(self.df)} models")
            if missing_native:
                print(f"Missing native structures for prefixes: {', '.join(set(missing_native))}")
        
        # 모델 파일 존재 확인
        if verbose:
            missing_models = [path for path in self.df['model_path'] if not os.path.exists(path)]
            if missing_models:
                print(f"Warning: {len(missing_models)} model PDB files do not exist")
                # 첫 5개만 출력
                for path in missing_models[:5]:
                    print(f"  - {path}")
                if len(missing_models) > 5:
                    print(f"  - ... and {len(missing_models) - 5} more")
        
        return self

    def add_chain_rmsd(self, align_chain='A', rmsd_chain='H', backbone_atoms=['CA', 'N', 'C', 'O']):
        """
        같은 query 내 모델들 간의 RMSD를 계산하고 결과를 DataFrame에 추가합니다.
        A 체인으로 정렬 후 H 체인의 RMSD를 계산합니다.
        
        Parameters
        ----------
        align_chain : str, optional
            구조 정렬에 사용할 체인 ID, 기본값은 'A'
        rmsd_chain : str, optional
            RMSD 계산에 사용할 체인 ID, 기본값은 'H'
        backbone_atoms : list, optional
            계산에 사용할 백본 원자 목록, 기본값은 ['CA', 'N', 'C', 'O']
            
        Returns
        -------
        self : Data
            업데이트된 Data 객체 (메소드 체이닝용)
        """
        import numpy as np
        import logging
        from af_analysis.clustering import compute_chain_rmsd_pdb_numpy
        
        logger = logging.getLogger(__name__)
        
        # 각 query별로 처리
        for query in self.df['query'].unique():
            logger.info(f"처리 중: {query}")
            
            # 해당 query의 모델들 선택
            query_df = self.df[self.df['query'] == query]
            query_indices = query_df.index.tolist()
            
            # PDB 파일 경로 추출
            query_files = [f for f in query_df['pdb'].tolist() if isinstance(f, str)]
            n_models = len(query_files)
            
            if n_models <= 1:
                # 모델이 1개 이하면 RMSD 계산 불가
                self.df.loc[query_df.index, 'model_avg_RMSD'] = None
                self.df.loc[query_df.index, 'query_avg_RMSD'] = None
                logger.warning(f"{query}: 모델이 1개 이하여서 RMSD 계산 불가")
                continue
            
            try:
                # pdb_numpy 기반 RMSD 거리 행렬 계산
                dist_matrix = compute_chain_rmsd_pdb_numpy(
                    query_files, 
                    align_chain=align_chain,
                    rmsd_chain=rmsd_chain,
                    backbone_atoms=backbone_atoms
                )
                
                # 전체 평균 RMSD 계산 (대각선 요소 제외)
                valid_mask = (dist_matrix > 0) & ~np.eye(n_models, dtype=bool)
                if np.any(valid_mask):
                    query_avg = float(np.mean(dist_matrix[valid_mask]))
                    logger.info(f"{query}: 평균 RMSD = {query_avg:.2f} Å")
                else:
                    query_avg = None
                    logger.info(f"{query}: 평균 RMSD = N/A (유효한 값 없음)")
                
                # 각 모델별 평균 RMSD 계산 및 할당
                for i in range(n_models):
                    idx = query_indices[i]
                    row_values = dist_matrix[i, :]
                    valid_values = row_values[(row_values > 0) & (np.arange(n_models) != i)]
                    
                    model_avg = float(np.mean(valid_values)) if len(valid_values) > 0 else None
                    self.df.loc[idx, 'model_avg_RMSD'] = model_avg
                    self.df.loc[idx, 'query_avg_RMSD'] = query_avg
                
                # 누락된 모델 처리
                missing_indices = query_indices[n_models:]
                if missing_indices:
                    self.df.loc[missing_indices, 'model_avg_RMSD'] = None
                    self.df.loc[missing_indices, 'query_avg_RMSD'] = None
            
            except Exception as e:
                logger.error(f"{query} 처리 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
                self.df.loc[query_df.index, 'model_avg_RMSD'] = None
                self.df.loc[query_df.index, 'query_avg_RMSD'] = None
        
        return self

    def add_rmsd_scale(self):
        """
        DataFrame에 model_rmsd_scale 열을 추가합니다.
        
        model_avg_RMSD와 query_avg_RMSD를 사용하여 모델의 RMSD를 스케일링합니다.
        스케일링은 1 / (1 + (model_avg_RMSD / query_avg_RMSD) ** 2) 공식을 사용합니다.
        
        Returns
        -------
        self : Data
            업데이트된 Data 객체 (메소드 체이닝용)
        """
        import numpy as np
        import pandas as pd
        import logging
        
        logger = logging.getLogger(__name__)
        
        # model_avg_RMSD 및 query_avg_RMSD가 존재하는지 확인
        if 'model_avg_RMSD' not in self.df.columns or 'query_avg_RMSD' not in self.df.columns:
            logger.warning("model_avg_RMSD 또는 query_avg_RMSD 열이 없습니다. add_chain_rmsd()를 먼저 실행하세요.")
            return self
        
        # model_rmsd_scale 계산
        def scale_rmsd(row):
            model_rmsd = row['model_avg_RMSD']
            query_rmsd = row['query_avg_RMSD']
            
            if pd.isna(model_rmsd) or pd.isna(query_rmsd) or query_rmsd == 0:
                return None
            
            return 1 / (1 + (model_rmsd / query_rmsd) ** 2)
        
        self.df['model_rmsd_scale'] = self.df.apply(scale_rmsd, axis=1)
        
        logger.info("model_rmsd_scale 열이 추가되었습니다.")
        
        return self

    def analyze_chains(self, idx=None, verbose=True):
        """
        Data 객체 내 모델들의 체인별 pLDDT와 PAE 값을 분석하고 결과를 현재 DataFrame에 추가합니다.
        각 체인의 평균 pLDDT, 체인 내부 PAE, 그리고 체인 쌍 간의 PAE를 계산합니다.
        
        Parameters
        ----------
        idx : int or list, optional
            분석할 모델의 인덱스. None이면 모든 모델 분석
        verbose : bool, optional
            진행 상태를 출력할지 여부, 기본값은 True
            
        Returns
        -------
        self : Data
            업데이트된 Data 객체 (메서드 체이닝 지원)
        """
        import numpy as np
        import pandas as pd
        import json
        from collections import Counter
        import pdb_numpy
        
        # 결과 저장 리스트
        results = []
        
        # 분석할 인덱스 목록 설정
        if idx is None:
            indices = range(len(self.df))
        elif isinstance(idx, (list, tuple)):
            indices = idx
        else:
            indices = [idx]
        
        for i in indices:
            try:
                # 모델 및 신뢰도 데이터 로드
                row = self.df.iloc[i]
                model_path = row["pdb"]
                json_path = row["data_file"]
                
                if verbose:
                    print(f"Analyzing model {i}: {model_path}")
                
                # JSON 파일에서 신뢰도 데이터 로드
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                # atom pLDDT 및 체인 정보 추출
                atom_plddts = np.array(json_data['atom_plddts'])
                atom_plddts = np.nan_to_num(atom_plddts)  # NaN 값 처리
                
                atom_chain_ids = json_data['atom_chain_ids']  # 각 원자의 체인 ID
                unique_chains = sorted(list(set(atom_chain_ids)))
                
                # PAE 데이터 추출
                pae_matrix = np.array(json_data['pae'])
                pae_matrix = np.nan_to_num(pae_matrix)
                
                if verbose:
                    print(f"Loaded pLDDT data for {len(atom_plddts)} atoms")
                    print(f"Chains identified: {unique_chains}")
                    print(f"PAE matrix shape: {pae_matrix.shape}")
                
                # CIF 파일 파싱하여 원자 정보 추출
                model = pdb_numpy.Coor(model_path)
                atoms = model.models[0]
                
                # Heavy atom 선택 (수소 제외)
                atom_mask = np.array([not name.startswith('H') for name in atoms.name])
                heavy_atoms = np.where(atom_mask)[0]
                
                if verbose:
                    print(f"Selected {len(heavy_atoms)} heavy atoms out of {len(atoms.name)} total atoms")
                
                # 원자 좌표 및 정보
                atom_chains = np.array(atoms.chain)[heavy_atoms]
                atom_resids = np.array(atoms.resid)[heavy_atoms]
                
                # 모델의 유니크한 체인 식별 (CIF 파일 기준)
                model_chains = np.unique(atom_chains)
                
                # 결과 저장 딕셔너리
                model_result = {}
                
                # 체인별 분석
                for chain in model_chains:
                    if verbose:
                        print(f"Analyzing chain {chain}")
                    
                    # 체인별 원자 마스크 및 인덱스
                    chain_mask = atom_chains == chain
                    chain_indices = np.where(chain_mask)[0]
                    
                    if verbose:
                        print(f"Chain {chain}: {len(chain_indices)} atoms")
                    
                    # 체인에 속한 원자들의 pLDDT 계산
                    chain_plddt_values = []
                    for idx in chain_indices:
                        atom_idx = heavy_atoms[idx]
                        if atom_idx < len(atom_plddts):
                            chain_plddt_values.append(atom_plddts[atom_idx])
                    
                    if chain_plddt_values:
                        avg_plddt = float(np.mean(chain_plddt_values))
                        model_result[f'chain_plddt_{chain}'] = avg_plddt
                        if verbose:
                            print(f"Chain {chain} average pLDDT: {avg_plddt:.2f}")
                    
                    # 체인에 속한 잔기들의 PAE 계산
                    try:
                        # 체인의 잔기 목록
                        chain_residues = sorted(np.unique(atom_resids[chain_mask]))
                        
                        # PAE 매트릭스 인덱스 계산을 위한 정보
                        residue_positions = {}
                        position = 0
                        for ch in sorted(model_chains):
                            residue_positions[ch] = {}
                            ch_residues = sorted(np.unique(atom_resids[atom_chains == ch]))
                            for res in ch_residues:
                                residue_positions[ch][res] = position
                                position += 1
                        
                        # 체인 내부 잔기 쌍의 PAE 값 추출
                        chain_pae_values = []
                        
                        for res1_idx, res1 in enumerate(chain_residues):
                            for res2 in chain_residues[res1_idx+1:]:  # 모든 쌍을 고려
                                if res1 in residue_positions[chain] and res2 in residue_positions[chain]:
                                    idx1 = residue_positions[chain][res1]
                                    idx2 = residue_positions[chain][res2]
                                    
                                    if idx1 < pae_matrix.shape[0] and idx2 < pae_matrix.shape[1]:
                                        # PAE 행렬에서 값 가져오기 (양방향 평균)
                                        pae_val1 = pae_matrix[idx1, idx2]
                                        pae_val2 = pae_matrix[idx2, idx1]
                                        chain_pae_values.append((pae_val1 + pae_val2) / 2)
                    
                        if chain_pae_values:
                            avg_pae = float(np.mean(chain_pae_values))
                            model_result[f'chain_pae_{chain}'] = avg_pae
                            if verbose:
                                print(f"Chain {chain} average internal PAE: {avg_pae:.2f}")
                    except Exception as e:
                        if verbose:
                            print(f"Error calculating PAE for chain {chain}: {e}")
                
                # 체인 쌍 간의 PAE 분석
                for idx1, chain1 in enumerate(model_chains):
                    for chain2 in model_chains[idx1+1:]:
                        pair_id = f"{chain1}{chain2}"
                        if verbose:
                            print(f"Analyzing chain pair {pair_id} PAE")
                        
                        try:
                            # 체인별 잔기 목록
                            chain1_mask = atom_chains == chain1
                            chain2_mask = atom_chains == chain2
                            chain1_residues = sorted(np.unique(atom_resids[chain1_mask]))
                            chain2_residues = sorted(np.unique(atom_resids[chain2_mask]))
                            
                            # 체인 쌍 간의 PAE 값 추출
                            pair_pae_values = []
                            
                            for res1 in chain1_residues:
                                for res2 in chain2_residues:
                                    if (res1 in residue_positions[chain1] and 
                                        res2 in residue_positions[chain2]):
                                        idx1 = residue_positions[chain1][res1]
                                        idx2 = residue_positions[chain2][res2]
                                        
                                        if (idx1 < pae_matrix.shape[0] and 
                                            idx2 < pae_matrix.shape[1]):
                                            # 양방향 PAE 평균 (chain1->chain2, chain2->chain1)
                                            pae_val1 = pae_matrix[idx1, idx2]
                                            pae_val2 = pae_matrix[idx2, idx1]
                                            pair_pae_values.append((pae_val1 + pae_val2) / 2)
                        
                            if pair_pae_values:
                                avg_pair_pae = float(np.mean(pair_pae_values))
                                model_result[f'chain_pair_pae_{pair_id}'] = avg_pair_pae
                                if verbose:
                                    print(f"Chain pair {pair_id} average PAE: {avg_pair_pae:.2f}")
                        except Exception as e:
                            if verbose:
                                print(f"Error calculating PAE for chain pair {pair_id}: {e}")
                
                # 모델 전체 평균값 계산
                plddt_values = [v for k, v in model_result.items() if 'chain_plddt_' in k]
                internal_pae_values = [v for k, v in model_result.items() if 'chain_pae_' in k]
                pair_pae_values = [v for k, v in model_result.items() if 'chain_pair_pae_' in k]
                
                if plddt_values:
                    model_result['avg_model_plddt'] = float(np.mean(plddt_values))
                    if verbose:
                        print(f"Model average pLDDT: {model_result['avg_model_plddt']:.2f}")
                if internal_pae_values:
                    model_result['avg_internal_pae'] = float(np.mean(internal_pae_values))
                    if verbose:
                        print(f"Model average internal PAE: {model_result['avg_internal_pae']:.2f}")
                if pair_pae_values:
                    model_result['avg_pair_pae'] = float(np.mean(pair_pae_values))
                    if verbose:
                        print(f"Model average chain pair PAE: {model_result['avg_pair_pae']:.2f}")
                
                # 결과 저장 (인덱스로 저장)
                results.append((i, model_result))
                
            except Exception as e:
                if verbose:
                    print(f"Error analyzing model {i}: {str(e)}")
        
        # 결과를 기존 DataFrame에 추가
        if results:
            for idx, result_dict in results:
                for key, value in result_dict.items():
                    self.df.at[idx, key] = value
        
        # 자기 자신을 반환 (메서드 체이닝을 위해)
        return self


def concat_data(data_list):
    """Concatenate data from a list of Data objects.

    Parameters
    ----------
    data_list : list
        List of Data objects.

    Returns
    -------
    Data
        Concatenated Data object.
    """

    concat = Data(directory=None, csv=None)

    concat.df = pd.concat([data.df for data in data_list], ignore_index=True)
    concat.chains = data_list[0].chains
    concat.chain_length = data_list[0].chain_length
    concat.format = data_list[0].format
    for i in range(1, len(data_list)):
        concat.chains.update(data_list[i].chains)
        concat.chain_length.update(data_list[i].chain_length)

    return concat


def read_multiple_alphapulldown(directory):
    """Read multiple directories containing AlphaPulldown data.

    Parameters
    ----------
    directory : str
        Path to the directory containing the directories.

    Returns
    -------
    Data
        Concatenated Data object.
    """

    dir_list = [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    data_list = []

    for dir in dir_list:
        if "ranking_debug.json" in os.listdir(os.path.join(directory, dir)):
            data_list.append(Data(os.path.join(directory, dir)))

    if len(data_list) == 0:
        raise ValueError("No AlphaPulldown data found in the directory.")
    return concat_data(data_list)

def calculate_contact_map(model, distance_threshold=8.0):
    """
    단백질 구조에서 잔기 간 접촉 맵을 계산합니다.
    
    Parameters
    ----------
    model : pdb_numpy.Coor
        단백질 구조 객체
    distance_threshold : float
        접촉으로 간주할 거리 임계값(Å)
        
    Returns
    -------
    contact_map : numpy.ndarray
        이진 접촉 맵 (접촉=1, 비접촉=0)
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    
    # CA, CB 또는 P 원자만 선택
    ca_atoms = model.select_atoms("name CA or name CB or name P")
    
    # 올바른 방식으로 원자 좌표 가져오기
    # models[0]에서 x, y, z 좌표를 직접 가져옴
    m = ca_atoms.models[0]
    coordinates = np.column_stack((m.x, m.y, m.z))
    
    # 원자 간 거리 계산
    distances = squareform(pdist(coordinates))
    
    # 인 원자(P) 처리를 위한 보정
    is_phosphorus = np.array(['P' in name for name in m.name])
    phosphorus_mask = np.outer(is_phosphorus, is_phosphorus) | np.outer(is_phosphorus, np.ones(len(is_phosphorus)))
    adjusted_distances = np.where(phosphorus_mask, distances - 4.0, distances)
    
    # 접촉 맵 생성
    contact_map = np.where(adjusted_distances < distance_threshold, 1, 0)
    return contact_map
