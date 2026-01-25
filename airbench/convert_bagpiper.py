import argparse
import json
from pathlib import Path

def main(metadata_path: Path, pred_path: Path, save_path: Path) -> None:
    metadata = [json.loads(line) for line in metadata_path.open(mode="r").readlines()]
    preds = json.load(pred_path.open(mode="r"))
    outputs = []
    fail_count = 0
    fail_id = []
    for item in metadata:
        ex_id = item["example_id"]
        pred = preds[ex_id][0][-1][0]
        think_end = pred.find("</think>")
        if think_end >= 0:
            response = pred[think_end + len("</think>"):]
        else:
            response = pred
            fail_count += 1
            fail_id.append(ex_id)
        item["sllm_response"] = response.strip()
        outputs.append(item)
    with save_path.open(mode="w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")
    print(f"{fail_count} examples do not contain thinking tokens.")
    print(f"Example IDs: {fail_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))

