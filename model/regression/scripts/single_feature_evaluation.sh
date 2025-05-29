#!/bin/bash

python ../final_model_codes/evaluate_single_feature_regression.py \
    --data-path /home/cseomoon/appl/af_analysis-0.1.4/model/regression/data/ABAG_final_test_dataset_20250512.csv \
    --target-col DockQ \
    --feature-cols scaled_model_RMSD scaled_query_RMSD iptm_A iptm_H iptm_L chain_pair_iptm_AH pdockq_AH pdockq_AL pdockq_HL pdockq pdockq2_AH pdockq2_AL pdockq2_HL pdockq2 LIS_AH LIS_AL LIS_HL avg_LIS \
    --output-dir /home/cseomoon/appl/af_analysis-0.1.4/model/regression/final_models
