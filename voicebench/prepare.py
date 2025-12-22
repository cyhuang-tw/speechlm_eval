import argparse
import json
import soundfile as sf
from pathlib import Path

from datasets import load_dataset, Audio
from tqdm import tqdm


def process_data(
    subset: str, split: str, data_dir: Path, audio_dir: Path, save_path: Path
) -> None:
    data = load_dataset(str(data_dir), subset, split=split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dir = audio_dir / subset / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, item in tqdm(enumerate(data)):
        example_id = f"{index:05d}"
        audio_path = audio_dir / f"{example_id}.wav"
        sf.write(audio_path, item["audio"]["array"], item["audio"]["sampling_rate"])
        item["example_id"] = example_id
        item["messages"] = [("user", "audio", str(audio_path))]
        del item["audio"]
        """
        example = {
            # The first two keys are mandatory.
            "example_id": example_id,
            "messages": [
                ("user", "audio", str(audio_path)),
            ],
            # These are optional. They may be used in the official VoiceBench code.
            "prompt": prompt,
            "reference": reference,
        }
        """
        outputs.append(item)
    with save_path.open(mode="w") as f:
        for out in outputs:
            f.write(json.dumps(out) + "\n")


def main(data_dir: Path, audio_dir: Path, save_dir: Path) -> None:
    subsets = [
        "advbench",
        "alpacaeval",
        "alpacaeval_full",
        "alpacaeval_speaker",
        "bbh",
        "commoneval",
        "ifeval",
        "mmsu",
        "mtbench",
        "openbookqa",
        "sd-qa",
        "wildvoice",
    ]
    sd_qa_opts = [
        "aus",
        "gbr",
        "ind_n",
        "ind_s",
        "irl",
        "kenya",
        "nga",
        "nzl",
        "phl",
        "usa",
        "zaf",
    ]
    alpacaeval_speaker_opts = [
        "en_AU_Wavenet_A_1.0_0.0_0.0",
        "en_AU_Wavenet_B_1.0_0.0_0.0",
        "en_IN_Wavenet_A_1.0_0.0_0.0",
        "en_IN_Wavenet_B_1.0_0.0_0.0",
        "en_GB_Wavenet_A_1.0_0.0_0.0",
        "en_GB_Wavenet_B_1.0_0.0_0.0",
        "en_US_Wavenet_A_1.0_0.0_0.0",
        "en_US_Wavenet_C_1.0_0.0_0.0",
        "en_US_Wavenet_A_1.5_0.0_0.0",
        "en_US_Wavenet_A_2.0_0.0_0.0",
        "en_US_Wavenet_A_0.5_0.0_0.0",
    ]

    mmsu_opts = [
        "law",
        "engineering",
        "other",
        "biology",
        "business",
        "economics",
        "health",
        "philosophy",
        "psychology",
        "history",
        "chemistry",
        "physics",
    ]

    save_dir.mkdir(parents=True, exist_ok=True)
    for sb in subsets:
        if sb == "sd-qa":
            splits = sd_qa_opts
        elif sb == "alpacaeval_speaker":
            splits = alpacaeval_speaker_opts
        elif sb == "mmsu":
            splits = mmsu_opts
        elif sb == "mtbench":
            print("Skipping mtbench for now...")
            continue
        else:
            splits = ["test"]
        for sp in splits:
            save_path = save_dir / f"{sb}_{sp}.jsonl"
            print(f"processing {sb}, {sp}")
            process_data(sb, sp, data_dir, audio_dir, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--audio_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    main(**vars(parser.parse_args()))
