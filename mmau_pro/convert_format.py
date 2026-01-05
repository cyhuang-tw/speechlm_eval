import argparse
import json
from pathlib import Path

import pandas as pd

def main(parquet_path: Path, jsonl_path: Path, save_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    predictions = {}
    for item in metadata:
        predictions[item["id"]] = item["sllm_response"]
    df["prediction"] = df["id"].map(predictions)
    df.to_parquet(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=Path, required=True)
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
