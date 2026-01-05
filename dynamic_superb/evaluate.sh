#!/bin/bash

RET_DIR=$1
OUT_DIR=$2
mkdir -p tmp

for FILE in ${RET_DIR}/*.jsonl; do
    BASE="${FILE##*/}"
    STEM="${BASE%.*}"
    python evaluation/convert_format.py --jsonl_path $FILE --output_path tmp/${STEM}.json
done

python -m evaluation.evaluation --result_dir tmp --save_dir ${OUT_DIR}

rm -rf tmp
