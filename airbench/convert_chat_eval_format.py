import argparse
import json
from pathlib import Path


def main(input_jsonl: Path, output_json: Path) -> None:
    data = []
    with open(input_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            item["gpt-score"] = item.pop("gen")  # rename key
            data.append(item)

    with open(output_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(data)} items to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    main(**vars(parser.parse_args()))
