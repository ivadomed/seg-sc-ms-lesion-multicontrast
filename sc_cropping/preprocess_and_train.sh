#!/bin/bash
#
# Reorient (RPI) + resample (1 mm iso) the nnUNet dataset, then plan, preprocess
# (default + ResEnc L planners) and train nnUNet on one fold / one GPU.
# Dataset Dataset902_msLesionAgnostic is transformed into Dataset903_msLesionAgnostic.
#
# Usage:
#   bash preprocess_and_train.sh
#
# Requires an environment with nnUNetv2, nibabel and tqdm, and SCT in the PATH.
# Safe to re-run: already resampled cases and finished preprocessing are skipped, and training
# resumes from the latest checkpoint.
#
# Author: Pierre-Louis Benveniste

set -eo pipefail

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# nnUNet paths export
export nnUNet_raw="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_raw"
export nnUNet_preprocessed="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_preprocessed"
export nnUNet_results="/home/plbenveniste/net/ms-lesion-agnostic/202611_retrain_sc_crop/data/nnUNet_results"

# Dataset values
input_dataset="Dataset902_msLesionAgnostic"
dataset_number=903
dataset_name="Dataset${dataset_number}_msLesionAgnostic"
configurations="3d_fullres"
planner="nnUNetPlannerResEncL"
plans="nnUNetResEncUNetLPlans"   # plans name created by nnUNetPlannerResEncL
trainer="nnUNetTrainerDiceCELoss_noSmooth"
fold=0
gpu=3
n_workers=8                      # cases resampled in parallel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$nnUNet_results/logs"
mkdir -p "$LOG_DIR"

# 1. Reorient to RPI and resample to 1 mm isotropic
echo ""
echo "Reorienting and resampling $input_dataset -> $dataset_name ..."
python "$SCRIPT_DIR/reorient_resample_nnunet_raw.py" \
    -i "$nnUNet_raw/$input_dataset" \
    -o "$nnUNet_raw/$dataset_name" \
    -w $n_workers

# Steps 2-3 are skipped if they already completed (marker written at the end of step 3)
preproc_done="$nnUNet_preprocessed/$dataset_name/.preprocessing_done"
if [ -f "$preproc_done" ]; then
    echo ""
    echo "Preprocessing already done ($preproc_done), skipping."
else
    # 2. Plan with the default planner + verify dataset integrity (no preprocessing: these plans are not used)
    echo ""
    echo "Verifying dataset $dataset_number integrity and planning (default planner) ..."
    nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations --verify_dataset_integrity --no_pp

    # 3. Plan and preprocess with the ResEnc L planner
    echo ""
    echo "Planning and preprocessing dataset $dataset_number ($planner) ..."
    nnUNetv2_plan_and_preprocess -d $dataset_number -c $configurations -pl $planner
    touch "$preproc_done"
fi

# 4. Train (--c resumes from the latest checkpoint if there is one, otherwise starts from scratch)
ts=$(date '+%Y-%m-%d-%H-%M-%S')
log="$LOG_DIR/train_${dataset_number}_fold${fold}_${ts}.txt"
echo ""
echo "Training fold $fold on GPU $gpu with $trainer (log: $log) ..."
CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train $dataset_number $configurations $fold \
    -p $plans -tr $trainer --c 2>&1 | tee "$log"

echo "Done: $(date)"
