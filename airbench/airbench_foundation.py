import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main(data_dir: Path, save_path: Path) -> None:
    root_path = data_dir / "Foundation"
    input_file = root_path / "Foundation_meta.json"
    data = json.load(input_file.open(mode="r"))
    outputs = []
    for item in tqdm(data):
        example_id = item["uniq_id"]
        wav = item["path"]
        task_name = item["task_name"]
        dataset_name = item["dataset_name"]
        if task_name == "Audio_Grounding":
            data_path = root_path / f"{task_name}_{dataset_name}" / f"{wav[:-3]}flac"
        else:
            data_path = root_path / f"{task_name}_{dataset_name}" / wav
        if not data_path.exists():
            print(f"Audio file {data_path} does not exist.")
            continue
        question = item["question"]
        question_prompts = "Choose the most suitable answer from options A, B, C, and D to respond the question in next line, you may only choose A or B or C or D."
        choice_a = item["choice_a"]
        choice_b = item["choice_b"]
        choice_c = item.get("choice_c", None)
        choice_d = item.get("choice_d", None)
        choices = f"A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}"
        instruction = question_prompts + "\n" + question + "\n" + choices
        answer_gt = item["answer_gt"]

        example = {
            # The first two keys are mandatory.
            "example_id": example_id,
            "messages": [
                ("user", "audio", str(data_path)),
                ("user", "text", instruction),
            ],
            # These are optional. They may be used in the official AIR-Bench code.
            "question": question,
            "choice_a": choice_a,
            "choice_b": choice_b,
            "choice_c": choice_c,
            "choice_d": choice_d,
            "answer_gt": answer_gt,
            "task_name": task_name,
            "dataset_name": dataset_name,
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
