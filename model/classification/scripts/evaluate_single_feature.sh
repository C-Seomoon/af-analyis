#!/bin/bash

python ../evaluate_single_feature.py \
    --data-path /home/cseomoon/appl/af_analysis-0.1.4/model/classification/data/test/pipeline_ABAG_native_data.csv \
    --feature-col iptm_A iptm_H iptm_L ptm_A ptm_H ptm_L chain_pair_iptm_AH chain_pair_iptm_AL chain_pair_iptm_HL chain_pair_pae_min_AH chain_pair_pae_min_AL chain_pair_pae_min_HL chain_plddt_A chain_pae_A chain_plddt_H chain_pae_H chain_plddt_L chain_pae_L chain_pair_pae_AH chain_pair_pae_AL chain_pair_pae_HL avg_model_plddt avg_internal_pae avg_pair_pae plddt_H3 plddt_L3 contacts_AH interface_plddt_AH interface_pae_AH contacts_AL interface_plddt_AL interface_pae_AL contacts_HL interface_plddt_HL interface_pae_HL total_contacts avg_interface_plddt avg_interface_pae pdockq_AH pdockq_AL pdockq_HL pdockq pdockq2_AH pdockq2_AL pdockq2_HL pdockq2 LIS LIS_AH LIS_AL LIS_HL avg_LIS piTM pIS piTM_A piTM_H piTM_L model_avg_RMSD query_avg_RMSD scaled_RMSD_ratio scaled_model_RMSD scaled_query_RMSD dG_separated dSASA_int nres_int delta_unsatHbonds packstat dG_dSASA_norm \
    --dockq-threshold 0.23 \
    --output-dir /home/cseomoon/appl/af_analysis-0.1.4/model/classification/features/
