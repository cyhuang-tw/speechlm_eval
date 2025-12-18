#!/bin/bash

# Please run "hf auth login" prior to the download.

HF_PATH="gamma-lab-umd/MMAU-Pro"
LOCAL_DIR="mmau_pro"

hf download ${HF_PATH} \
  --repo-type dataset \
  --local-dir ${LOCAL_DIR}

UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip ${LOCAL_DIR}/data.zip -d ${LOCAL_DIR}
