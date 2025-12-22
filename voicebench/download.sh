#!/bin/bash
HF_PATH=hlt-lab/voicebench
LOCAL_PATH=/work/hdd/bbjs/chuang14/benchmark_data/voicebench

hf download ${HF_PATH} \
    --repo-type=dataset \
    --local-dir ${LOCAL_PATH}
