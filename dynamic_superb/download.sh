#!/bin/bash

# TASK_NAME
TASKS=(
  "SuperbASR_LibriSpeech-TestClean"
  "SuperbPR_LibriSpeech-TestClean"
  "SuperbKS_SpeechCommandsV1-Test"
  "SuperbIC_SLURP"
  "SuperbSF_AudioSnips-Test"
  "SuperbQbE_Quesst14-Eval"
  "SuperbSV_SuperbHiddenSet"
  "SuperbSD_Libri2Mix-Test"
  "SuperbER_RAVDESS"
)
LOCAL_DIR="/work/hdd/bbjs/chuang14/benchmark_data/dynamic_superb"
mkdir -p ${LOCAL_DIR}

for TASK in "${TASKS[@]}"; do
    echo "Running: ${TASK}"
    hf download "DynamicSuperb/${TASK}" \
        --repo-type dataset \
        --local-dir "${LOCAL_DIR}/${TASK}"
done
