import argparse
import json
from pathlib import Path


def main(jsonl_path: Path, save_path: Path) -> None:
    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    with save_path.open(mode="w") as f:
        for item in metadata:
            item["response"] = item["sllm_response"]
            del item["sllm_response"]
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
