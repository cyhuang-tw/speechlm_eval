import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main(data_dir: Path, save_path: Path) -> None:
    root_path = data_dir / "Chat"
    input_file = root_path / "Chat_meta.json"
    data = json.load(input_file.open(mode="r"))
    outputs = []
    for item in tqdm(data):
        example_id = item["uniq_id"]
        wav = item["path"]
        task_name = item["task_name"]
        dataset_name = item["dataset_name"]
        meta_info = item["meta_info"]
        data_path = root_path / f"{task_name}_{dataset_name}" / wav
        if not data_path.exists():
            print(f"Audio file {data_path} does not exist.")
            continue
        instruction = item["question"]
        answer_gt = item["answer_gt"]

        example = {
            # The first two keys are mandatory.
            "example_id": str(example_id),
            "messages": [
                ["system", "text", "You are a helpful audio understanding assistant. When given an audio and a question about it, first think through what you hear in the audio and how it relates to the question, then provide a clear and accurate answer."],
                ["user", "audio", str(data_path.absolute())],
                ["user", "text", instruction],
            ],
            # These are optional. They may be used in the official AIR-Bench code.
            "question": instruction,
            "answer_gt": answer_gt,
            "task_name": task_name,
            "dataset_name": dataset_name,
            "meta_info": meta_info,
            "path": wav,
        }
        outputs.append(example)

    with save_path.open(mode="w") as f:
        for out in outputs:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
