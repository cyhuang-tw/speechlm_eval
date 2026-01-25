#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare AudioBench datasets for inference in custom JSONL format.

This script converts AudioBench datasets to a JSONL format suitable for
custom speech LLM pipelines. Each line contains:
- example_id: unique identifier
- messages: list of [role, type, content] triplets
- Additional metadata for evaluation (reference, task_type, etc.)

Usage:
    python prepare_audiobench.py \
        --dataset_name librispeech_test_clean \
        --output_jsonl data/librispeech_test_clean.jsonl \
        --audio_dir data/audio \
        --number_of_samples -1
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import soundfile as sf
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import Dataset

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# =============================================================================
# Constants
# =============================================================================
DEFAULT_SYSTEM_PROMPT = "You are a helpful audio understanding assistant."


def save_audio_file(
    audio_array: np.ndarray,
    sampling_rate: int,
    output_path: str
) -> str:
    """Save audio array to a WAV file.
    
    Args:
        audio_array: Audio samples as numpy array
        sampling_rate: Audio sampling rate in Hz
        output_path: Path to save the WAV file
        
    Returns:
        Absolute path to the saved audio file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ensure audio is in the correct format
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Normalize if needed (avoid clipping), but handle silent audio
    max_val = np.abs(audio_array).max()
    if max_val > 1.0:
        audio_array = audio_array / max_val
    
    sf.write(output_path, audio_array, sampling_rate)
    return os.path.abspath(output_path)


def convert_sample_to_jsonl_format(
    sample: dict,
    example_id: str,
    audio_path: str,
    dataset_name: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> dict:
    """Convert an AudioBench sample to the target JSONL format.
    
    Args:
        sample: AudioBench sample dict with 'audio', 'instruction', 'reference', etc.
        example_id: Unique identifier for this sample
        audio_path: Path to the saved audio file
        dataset_name: Name of the source dataset
        system_prompt: System prompt to use
        
    Returns:
        Dict in target JSONL format
    """
    # Get instruction - ensure it's a string
    instruction = sample.get("instruction", "")
    if instruction is None:
        instruction = ""
    
    # Build messages list
    messages = [
        ["system", "text", system_prompt],
        ["user", "audio", audio_path],
        ["user", "text", instruction],
    ]
    
    # Build output dict with all necessary fields for evaluation
    output = {
        "example_id": example_id,
        "messages": messages,
        "dataset_name": dataset_name,
        "task_type": sample.get("task_type", ""),
    }
    
    # Handle reference field (might be 'reference' or 'answer' depending on dataset)
    if "reference" in sample:
        output["reference"] = sample["reference"]
    if "answer" in sample:
        output["answer"] = sample["answer"]
    
    # Preserve additional fields that may be needed for evaluation
    # These vary by dataset type
    optional_fields = ["task", "choices", "other_attributes", "audio_gt"]
    for field in optional_fields:
        if field in sample:
            output[field] = sample[field]
    
    return output


def prepare_dataset(
    dataset_name: str,
    output_jsonl: str,
    audio_dir: str,
    number_of_samples: int = -1,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> None:
    """Prepare an AudioBench dataset for inference.
    
    Args:
        dataset_name: Name of the AudioBench dataset
        output_jsonl: Path to output JSONL file
        audio_dir: Directory to save audio files
        number_of_samples: Number of samples to process (-1 for all)
        system_prompt: System prompt to use in messages
    """
    logger.info(f"Preparing dataset: {dataset_name}")
    logger.info(f"Output JSONL: {output_jsonl}")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Number of samples: {number_of_samples}")
    
    # Load dataset using AudioBench's Dataset class
    dataset = Dataset(dataset_name, number_of_samples)
    input_data = dataset.input_data
    
    logger.info(f"Loaded {len(input_data)} samples")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    
    # Process each sample
    audio_subdir = os.path.join(audio_dir, dataset_name)
    os.makedirs(audio_subdir, exist_ok=True)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(input_data, desc="Processing samples")):
            # Generate example ID
            example_id = f"{dataset_name}_{idx:06d}"
            
            # Save audio file
            audio_data = sample["audio"]
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
            
            audio_filename = f"{example_id}.wav"
            audio_path = os.path.join(audio_subdir, audio_filename)
            audio_path = save_audio_file(audio_array, sampling_rate, audio_path)
            
            # Convert to target format
            jsonl_sample = convert_sample_to_jsonl_format(
                sample=sample,
                example_id=example_id,
                audio_path=audio_path,
                dataset_name=dataset_name,
                system_prompt=system_prompt,
            )
            
            # Write to JSONL
            f.write(json.dumps(jsonl_sample, ensure_ascii=False) + "\n")
    
    logger.info(f"Successfully wrote {len(input_data)} samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AudioBench datasets for inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the AudioBench dataset (e.g., librispeech_test_clean)"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory to save audio files"
    )
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use in messages"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        dataset_name=args.dataset_name,
        output_jsonl=args.output_jsonl,
        audio_dir=args.audio_dir,
        number_of_samples=args.number_of_samples,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
