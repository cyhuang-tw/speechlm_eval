import argparse
import json
from pathlib import Path

def main(jsonl_path: Path, output_path: Path) -> None:
    """
    Convert a JSONL file to a JSON file.
    """
    data = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    with output_path.open(mode="w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    main(**vars(parser.parse_args()))