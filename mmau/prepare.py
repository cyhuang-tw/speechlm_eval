import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main(data_dir: Path, save_path: Path) -> None:
    metadata_path = data_dir / "mmau-test-mini.json"
    metadata = json.load(metadata_path.open(mode="r"))
    outputs = []
    for item in tqdm(metadata):
        example_id = item["id"]
        audio_path = data_dir / Path(item["audio_id"])
        if not audio_path.exists():
            print(f"Audio file {audio_path} does not exist.")
            continue
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
