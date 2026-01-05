#!/bin/bash

# python evaluate.py --src_file result-naive-alpacaeval-test-audio.jsonl --evaluator open

JSONL_DIR=$1
OUT_DIR=$2

mkdir -p tmp
mkdir -p ${OUT_DIR}

for FILE in $JSONL_DIR/*; do
    python convert_format.py --jsonl_path $FILE --save_path tmp/${FILE##*/}
done

for FILE in tmp/*; do
    BASE="${FILE##*/}"
    STEM="${BASE%.*}"
    if [[ "$BASE" == *"sd-qa"* ]]; then
    EVALUATOR="qa"
    elif [[ "$BASE" == *"ifeval"* ]]; then
    EVALUATOR="ifeval"
    elif [[ "$BASE" == *"advbench"* ]]; then
    EVALUATOR="harm"
    elif [[ "$BASE" == *"openbookqa"* || "$BASE" == *"mmsu"* ]]; then
    EVALUATOR="mcq"
    elif [[ "$BASE" == *"bbh"* ]]; then
    EVALUATOR="bbh"
    else
    EVALUATOR="unknown"
    fi
    echo "$EVALUATOR"
    if [ "$EVALUATOR" == "unknown" ]; then
        echo "Skipping file $FILE due to unknown evaluator."
        continue
    fi
    python evaluate.py --src_file $FILE --evaluator $EVALUATOR 2>&1 | tee ${OUT_DIR}/${STEM}_eval.txt
done

rm -rf tmp
