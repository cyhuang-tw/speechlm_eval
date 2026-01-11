import argparse
import json
from pathlib import Path
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm


def main(jsonl_path: Path, output_path: Path) -> None:
    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    outputs = []

    # default: Load the model on the available device(s)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto"
    )
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a audio understanding model. Choose one of the options without any explanation.",
                # "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    }

    for data in tqdm(metadata):
        ex_id = data["example_id"]  # a string
        messages = data["messages"]  # a list of tuples

        user_messages = []
        for msg in messages:
            _, msg_type, msg_content = msg
            user_messages.append({"type": msg_type, msg_type: msg_content})
        user_messages = {"role": "user", "content": user_messages}
        conversation = [system_message, user_messages]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        if len(audios[0]) == 0:
            print(f"Example ID: {ex_id} has corrupted audio, skipping it for now.")
            data["sllm_response"] = ""
            outputs.append(data)
            continue

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids = model.generate(**inputs, return_audio=False)
        text_ids = text_ids[:, inputs.input_ids.shape[-1] :]
        text = processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(text)
        data["sllm_response"] = text
        outputs.append(data)
    with output_path.open(mode="w") as f:
        for line in outputs:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
