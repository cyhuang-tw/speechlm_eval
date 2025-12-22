import argparse
import json
from pathlib import Path

# import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info


def main(jsonl_path: Path, save_path: Path) -> None:
    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    metadata = [json.loads(line) for line in jsonl_path.open(mode="r").readlines()]
    outputs = []

    for data in metadata:
        ex_id = data["id"]
        messages = data["messages"]  # a list of tuples

        user_messages = []
        for msg in messages:
            _, msg_type, msg_content = msg
            user_messages.append({"type": msg_type, msg_type: msg_content})
        conversation = {"role": "user", "content": user_messages}

        # Preparation for inference
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
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

        # Inference: Generation of the output text and audio
        text_ids, _ = model.generate(
            **inputs, thinker_return_dict_in_generate=True, use_audio_in_video=False
        )

        text = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(text)
        data["sllm_response"] = text
        outputs.append(data)
    with save_path.open(mode="w") as f:
        for out in outputs:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    main(**vars(parser.parse_args()))
