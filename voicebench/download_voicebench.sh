#!/bin/bash
HF_PATH=hlt-lab/voicebench
LOCAL_PATH=voicebench

hf download ${HF_PATH} \
    --repo-type=dataset \
    --local-dir ${LOCAL_PATH}