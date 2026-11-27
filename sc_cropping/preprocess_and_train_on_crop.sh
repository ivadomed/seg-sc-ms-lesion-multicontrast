#!/bin/bash
#
# Preprocess and train nnUNet on the cropped dataset (Dataset903_msLesionAgnostic).
#
# Usage:
#   bash preprocess_and_train.sh
#
#
# Author: Pierre-Louis Benveniste

set -eo pipefail

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# nnUNet paths export
export nnUNet_raw="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_raw_cropped"
export nnUNet_preprocessed="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_preprocessed_cropped"
export nnUNet_results="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_results_cropped"

# Dataset values
dataset_number=903
configurations="3d_fullres"
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetLPlans"   # plans name created by nnUNetPlannerResEncL
trainer="nnUNetTrainerDiceCELoss_noSmooth"
fold=0
gpu=0


# 2. Plan with the default planner + verify dataset integrity (no preprocessing: these plans are not used)
echo ""
echo "Verifying dataset $dataset_number integrity and planning (default planner) ..."
nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations --verify_dataset_integrity --no_pp

# 3. Plan and preprocess with the ResEnc L planner
echo ""
echo "Planning and preprocessing dataset $dataset_number ($planner) ..."
nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations -pl $planner

# 4. Train (--c resumes from the latest checkpoint if there is one, otherwise starts from scratch)
echo ""
echo "Training fold $fold on GPU $gpu with $trainer (log: $log) ..."
CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train $dataset_number $configurations $fold -p $plans -tr $trainer --c

echo "Done: $(date)"
