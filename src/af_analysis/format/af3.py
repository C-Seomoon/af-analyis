#!/usr/bin/env python3
# coding: utf-8

import os
import re
import logging
import json
from tqdm.auto import tqdm
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def read_dir(directory):
    """Extract pdb list from a directory with AF3 format.

    Parameters
    ----------
    directory : str
        Path to the directory containing seed-{seed}_sample-{sample} folders.

    Returns
    -------
    log_pd : pandas.DataFrame
        Dataframe containing the information extracted from the directory.
    """

    logger.info(f"Reading {directory}")

    log_dict_list = []
    
    # Find all subdirectories matching the pattern seed-{seed}_sample-{sample}
    pattern = re.compile(r"seed-(\d+)_sample-(\d+)")
    
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        
        if not os.path.isdir(folder_path):
            continue
            
        match = pattern.match(folder)
        if not match:
            continue
            
        seed = int(match.group(1))
        sample = int(match.group(2))
        
        # Check if required files exist
        model_path = os.path.join(folder_path, "model.cif")
        summary_path = os.path.join(folder_path, "summary_confidences.json")
        conf_path = os.path.join(folder_path, "confidences.json")
        
        if not (os.path.exists(model_path) and os.path.exists(summary_path) and os.path.exists(conf_path)):
            continue
        
        # Extract query name from directory path
        parent_dir = os.path.basename(os.path.normpath(directory))
        query = parent_dir  # 언더스코어 유무와 관계없이 디렉토리 이름을 query로 사용
        
        # Read summary confidences
        with open(summary_path, "r") as f_in:
            json_dict = json.load(f_in)
        
        info_dict = {
            "pdb": model_path,
            "query": query,
            "seed": seed,
            "sample": sample,
            "data_file": conf_path,
        }
        info_dict.update(json_dict)
        log_dict_list.append(info_dict)

    log_pd = pd.DataFrame(log_dict_list)
    
    # Update column names if they exist
    if not log_pd.empty:
        rename_cols = {
            "ranking_score": "ranking_confidence",
            "ptm": "pTM",
            "iptm": "ipTM",
        }
        # Only rename columns that exist
        cols_to_rename = {k: v for k, v in rename_cols.items() if k in log_pd.columns}
        if cols_to_rename:
            log_pd = log_pd.rename(columns=cols_to_rename)

    # Sort the dataframe by seed and sample for consistency
    if not log_pd.empty:
        log_pd = log_pd.sort_values(by=["seed", "sample"]).reset_index(drop=True)
    
    return log_pd
