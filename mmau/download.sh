#!/bin/bash
GDRIVE_PATH="1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA"
LOCAL_DIR="/work/hdd/bbjs/chuang14/benchmark_data/mmau"
METADATA_PATH="https://raw.githubusercontent.com/Sakshi113/MMAU/refs/heads/main/mmau-test-mini.json"

mkdir -p ${LOCAL_DIR}

gdown "${GDRIVE_PATH}" -O "${LOCAL_DIR}/test-mini-audios.tar.gz"
tar -xvf "${LOCAL_DIR}/test-mini-audios.tar.gz" -C ${LOCAL_DIR}
rm "${LOCAL_DIR}/test-mini-audios.tar.gz"

wget "${METADATA_PATH}" -O "${LOCAL_DIR}/mmau-test-mini.json"
