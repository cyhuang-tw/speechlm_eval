import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main(data_dir: Path, save_path: Path) -> None:
    dataset = load_dataset(str(data_dir), split="test")
    outputs = []
    for item in tqdm(dataset):
        example_id = item["id"]
        if len(item["audio_path"]) > 1:
            print(
                f"Audio file {item['audio_path']} has more than one path. We will skip it for now."
            )
            continue
        audio_path = data_dir / item["audio_path"][0]
        question = item["question"]
        choices = item["choices"]
        instruction = f"Question: {question}\nChoices: {', '.join(choices)}"
        item["example_id"] = example_id
        item["messages"] = [
            ("user", "audio", str(audio_path)),
            ("user", "text", instruction),
        ]
        outputs.append(item)
    with save_path.open(mode="w") as f:
        for out in outputs:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
