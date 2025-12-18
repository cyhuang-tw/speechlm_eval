import argparse
import json
import soundfile as sf
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def process_task(data_dir: Path, audio_dir: Path, save_path: Path) -> None:
    dataset = load_dataset(str(data_dir), split="test")
    outputs = []
    for item in tqdm(dataset):
        example_id = item["file"]
        audio_path = audio_dir / f"{example_id}.wav"
        sf.write(
            str(audio_path), item["audio"]["array"], item["audio"]["sampling_rate"]
        )
        instruction = item["instruction"]
        label = item["label"]
        example = {
            "example_id": example_id,
            "messages": [
                ("user", "audio", str(audio_path)),
                ("user", "text", instruction),
            ],
            "instruction": instruction,
            "label": label,
        }
        outputs.append(example)
    with save_path.open(mode="w") as f:
        for out in outputs:
            f.write(json.dumps(out) + "\n")


def main(data_dir: Path, audio_dir: Path, save_dir: Path) -> None:
    tasks = [
        "SuperbASR_LibriSpeech-TestClean",
        "SuperbPR_LibriSpeech-TestClean",
        "SuperbKS_SpeechCommandsV1-Test",
        "SuperbIC_SLURP",
        "SuperbSF_AudioSnips-Test",
        "SuperbQbE_Quesst14-Eval",
        # "SuperbSV_SuperbHiddenSet",
        "SuperbSD_Libri2Mix-Test",
        "SuperbER_RAVDESS",
    ]
    for task in tasks:
        task_data_dir = data_dir / task
        task_audio_dir = audio_dir / task
        task_audio_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{task}.jsonl"
        process_task(task_data_dir, task_audio_dir, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    main(**vars(parser.parse_args()))
