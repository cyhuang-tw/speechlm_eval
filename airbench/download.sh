#!/bin/bash

echo "To download AIR-Bench, you will need around 50GB of free disk space."

HF_PATH=qyang1021/AIR-Bench-Dataset
LOCAL_PATH=/work/hdd/bbjs/chuang14/benchmark_data/AIR-Bench

hf download ${HF_PATH} \
  --repo-type dataset \
  --local-dir ${LOCAL_PATH}

echo "AIR-Bench dataset has been downloaded to ${LOCAL_PATH}"
