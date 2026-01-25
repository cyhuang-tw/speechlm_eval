#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate inference results using AudioBench metrics.

This script takes a JSONL file with model responses (sllm_response) and
computes evaluation metrics using AudioBench's evaluation methods.

Supported metrics:
- wer: Word Error Rate for ASR tasks
- bleu: BLEU score for translation tasks
- llama3_70b_judge: LLM-as-judge using Llama-3-70B
- gpt4o_judge: LLM-as-judge using GPT-4o
- string_match: Token-based matching for MCQ tasks

Usage:
    python evaluate_audiobench.py \
        --input_jsonl results/librispeech_test_clean_with_response.jsonl \
        --metrics wer \
        --output_json results/librispeech_test_clean_scores.json
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    return data


def extract_instruction_from_messages(messages: List[List[str]]) -> str:
    """Extract the instruction text from messages list.
    
    Args:
        messages: List of [role, type, content] triplets
        
    Returns:
        The instruction text (user's text message)
    """
    if not messages:
        return ""
    
    for msg in messages:
        # Ensure message has exactly 3 elements
        if not isinstance(msg, (list, tuple)) or len(msg) != 3:
            continue
        role, msg_type, content = msg
        if role == "user" and msg_type == "text":
            return content if content is not None else ""
    return ""


def convert_to_audiobench_format(
    data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert JSONL data to AudioBench's expected format for evaluation.
    
    Args:
        data: List of samples with sllm_response
        
    Returns:
        List of dicts with instruction, reference, model_prediction
    """
    converted = []
    for sample in data:
        # Handle sllm_response - ensure it's a string or empty string
        sllm_response = sample.get("sllm_response")
        if sllm_response is None:
            sllm_response = ""
        
        converted_sample = {
            "instruction": extract_instruction_from_messages(sample.get("messages", [])),
            "model_prediction": sllm_response,
        }
        
        # Handle reference field (might be called 'reference' or 'answer')
        if "reference" in sample:
            converted_sample["reference"] = sample["reference"]
        elif "answer" in sample:
            converted_sample["reference"] = sample["answer"]
        else:
            converted_sample["reference"] = ""
        
        # Also preserve 'answer' separately if it exists (for acc metric)
        if "answer" in sample:
            converted_sample["answer"] = sample["answer"]
        
        # Copy over additional fields needed for evaluation
        for field in ["task_type", "task", "choices", "example_id", "dataset_name", "audio_gt"]:
            if field in sample:
                converted_sample[field] = sample[field]
        
        converted.append(converted_sample)
    
    return converted


# =============================================================================
# Metric Implementations
# =============================================================================

def compute_wer(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Word Error Rate for ASR tasks.
    
    Args:
        data: List of samples with model_prediction and reference
        
    Returns:
        Dict with WER score and per-sample details
    """
    from jiwer import compute_measures, wer
    from dataset_src.text_normalizer.preprocess_text import preprocess_text_asr
    
    predictions = []
    references = []
    
    for item in data:
        # Handle model_prediction - ensure it's a string
        model_pred = item.get("model_prediction")
        if model_pred is None:
            model_pred = ""
        else:
            model_pred = str(model_pred)
        
        # Handle reference - ensure it's a string
        ref = item.get("reference")
        if ref is None:
            ref = ""
        else:
            ref = str(ref)
        
        # Apply text normalization
        model_prediction = preprocess_text_asr(model_pred)
        answer = preprocess_text_asr(ref)
        
        if len(model_prediction) == 0:
            model_prediction = "empty"
        if len(answer) == 0:
            answer = "empty"
        
        predictions.append(model_prediction)
        references.append(answer)
    
    # Compute overall WER
    sample_wer = []
    incorrect = 0
    total = 0
    
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        
        wer_score = wer(reference, prediction)
        sample_wer.append({
            "reference": reference,
            "prediction": prediction,
            "wer": wer_score,
        })
    
    total_wer = incorrect / total if total > 0 else 0.0
    
    return {"wer": total_wer, "sample_wer": sample_wer}


def compute_bleu(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute BLEU score for translation tasks.
    
    Uses HuggingFace evaluate library with flores101 tokenizer for
    compatibility with AudioBench's original implementation.
    
    Args:
        data: List of samples with model_prediction and reference
        
    Returns:
        Dict with BLEU score and per-sample details
    """
    import evaluate
    
    predictions = []
    references = []
    
    for item in data:
        # Handle model_prediction - ensure it's a string
        pred = item.get("model_prediction")
        if pred is None:
            pred = ""
        else:
            pred = str(pred)
        
        # Handle reference - ensure it's a string
        ref = item.get("reference")
        if ref is None:
            ref = ""
        else:
            ref = str(ref)
        
        if len(pred) == 0:
            pred = "empty"
        if len(ref) == 0:
            ref = "empty"
        
        predictions.append(pred)
        references.append(ref)
    
    # Use flores101 tokenizer as in AudioBench's original implementation
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(
        predictions=predictions,
        references=references,
        tokenize='flores101'
    )
    
    # Per-sample BLEU (approximate, as sacrebleu is primarily corpus-level)
    sample_bleu = []
    for pred, ref in zip(predictions, references):
        sample_result = sacrebleu.compute(
            predictions=[pred],
            references=[ref],
            tokenize='flores101'
        )
        sample_bleu.append({
            "reference": ref,
            "prediction": pred,
            "bleu": sample_result['score'],
        })
    
    return {"bleu": results['score'], "sample_bleu": sample_bleu}


def compute_accuracy(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy for math/reasoning tasks (spoken-mqa).
    
    Args:
        data: List of samples with model_prediction and reference/answer
        
    Returns:
        Dict with accuracy and per-sample details
    """
    from dataset_src.math_utils import utils
    
    def get_seperation_trigger(dataset: str):
        triggers = ['The answer is:', 'The answer is', 'the answer is']
        if dataset == 'gsm8k':
            triggers.append('####')
        return triggers
    
    def clean_answer(ans):
        """Clean an answer string following the original implementation."""
        ans = str(ans)
        if "####" in ans:
            ans = ans.split("#### ")[-1].replace(",", "")
        ans = utils.delete_extra_zero(ans)
        return ans
    
    predictions = []
    references = []
    
    for item in data:
        # Handle model prediction
        pred = item.get("model_prediction", "")
        if pred is None or pred == "":
            pred = "empty"
        else:
            pred = utils.answer_clean('gsm8k', get_seperation_trigger('gsm8k'), pred)
        
        if not pred:
            pred = "empty"
        
        # Handle reference (might be called 'answer' in some datasets)
        ref = item.get("reference", item.get("answer", ""))
        if isinstance(ref, list):
            # Clean each answer in list
            ref = [clean_answer(a) for a in ref]
        else:
            ref = [clean_answer(ref)]
        
        predictions.append(pred)
        references.append(ref)
    
    # Compute accuracy
    details = []
    correct, wrong = 0, 0
    
    for prediction, reference in zip(predictions, references):
        if len(prediction) > 100:
            prediction = prediction[:100]
        
        is_correct = utils.compare_answer_with_groundtruth(prediction, *reference)
        if is_correct:
            correct += 1
        else:
            wrong += 1
        
        details.append({
            "reference": reference,
            "prediction": prediction,
            "correct": is_correct,
        })
    
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0.0
    
    return {"acc": accuracy, "details": details}


def compute_llama3_70b_judge(data: List[Dict[str, Any]], binary: bool = False) -> Dict[str, Any]:
    """Compute LLM-as-judge score using Llama-3-70B.
    
    Args:
        data: List of samples with instruction, model_prediction, reference
        binary: If True, use binary scoring (0/1), else use 0-5 scale
        
    Returns:
        Dict with judge score and per-sample details
    """
    questions = [item["instruction"] for item in data]
    references = [item["reference"] for item in data]
    predictions = [item["model_prediction"] for item in data]
    
    if binary:
        from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge_binary
        judge_results, all_details = llama3_70b_as_judge_binary(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            [questions, references, predictions]
        )
    else:
        from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge
        judge_results, all_details = llama3_70b_as_judge(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            [questions, references, predictions]
        )
    
    return {"llama3_70b_judge": judge_results, "details": all_details}


def compute_gpt4o_judge(data: List[Dict[str, Any]], binary: bool = False) -> Dict[str, Any]:
    """Compute LLM-as-judge score using GPT-4o.
    
    Args:
        data: List of samples with instruction, model_prediction, reference
        binary: If True, use binary scoring (0/1), else use 0-5 scale
        
    Returns:
        Dict with judge score and per-sample details
    """
    questions = [item["instruction"] for item in data]
    references = [item["reference"] for item in data]
    predictions = [item["model_prediction"] for item in data]
    
    if binary:
        from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge_binary
        judge_results, all_details = gpt4o_as_judge_binary(
            "",
            [questions, references, predictions]
        )
    else:
        from dataset_src.eval_methods.eval_gpt4o import gpt4o_as_judge
        judge_results, all_details = gpt4o_as_judge(
            "",
            [questions, references, predictions]
        )
    
    return {"gpt4o_judge": judge_results, "details": all_details}


def compute_string_match(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute string matching score for MCQ tasks.
    
    Args:
        data: List of samples with instruction, model_prediction, reference, choices
        
    Returns:
        Dict with string match score and per-sample details
    """
    from dataset_src.eval_methods.string_match import mmau_string_match
    
    questions = [item["instruction"] for item in data]
    references = [item["reference"] for item in data]
    predictions = [item["model_prediction"] for item in data]
    choices = [item.get("choices", []) for item in data]
    
    string_match_results, all_details = mmau_string_match(
        [questions, references, predictions, choices]
    )
    
    return {"string_match": string_match_results, "details": all_details}


# =============================================================================
# Task Type to Metric Mapping
# =============================================================================

# Default metrics for different task types
TASK_TYPE_DEFAULT_METRICS = {
    "ASR": "wer",
    "ST": "bleu",  # Speech Translation
    "ST-EN-ZH": "bleu",
    "ST-EN-ID": "bleu",
    "ST-EN-TA": "bleu",
    "ST-ZH-EN": "bleu",
    "ST-ID-EN": "bleu",
    "ST-TA-EN": "bleu",
    "SQA": "llama3_70b_judge",  # Speech Question Answering
    "ER": "llama3_70b_judge",  # Emotion Recognition (binary)
    "GR": "llama3_70b_judge",  # Gender Recognition (binary)
    "AR": "llama3_70b_judge",  # Accent Recognition (binary)
    "Audio-Understanding-Reasoning": "string_match",  # MMAU
    "MathQA": "acc",  # Spoken-MQA
}

# Task types that use binary judge scoring
BINARY_JUDGE_TASK_TYPES = {"ER", "GR", "AR", "Audio-Understanding-Reasoning"}


def evaluate(
    input_jsonl: str,
    metrics: str,
    output_json: str = None,
) -> Dict[str, Any]:
    """Evaluate inference results using specified metrics.
    
    Args:
        input_jsonl: Path to JSONL file with sllm_response
        metrics: Evaluation metric to use (wer, bleu, llama3_70b_judge, gpt4o_judge, string_match)
        output_json: Optional path to save results
        
    Returns:
        Dict with evaluation results
    """
    logger.info(f"Loading data from: {input_jsonl}")
    data = load_jsonl(input_jsonl)
    logger.info(f"Loaded {len(data)} samples")
    
    # Check for sllm_response
    missing_response = sum(1 for d in data if "sllm_response" not in d)
    if missing_response > 0:
        logger.warning(f"{missing_response} samples missing 'sllm_response' field")
    
    # Convert to AudioBench format
    converted_data = convert_to_audiobench_format(data)
    
    # Determine if binary scoring is needed based on task types
    task_types = set(d.get("task_type", "") for d in converted_data)
    use_binary = bool(task_types & BINARY_JUDGE_TASK_TYPES)
    
    logger.info(f"Task types found: {task_types}")
    logger.info(f"Using metric: {metrics}")
    if metrics in ["llama3_70b_judge", "gpt4o_judge"]:
        logger.info(f"Binary scoring: {use_binary}")
    
    # Compute metrics
    if metrics == "wer":
        results = compute_wer(converted_data)
    elif metrics == "bleu":
        results = compute_bleu(converted_data)
    elif metrics == "acc":
        results = compute_accuracy(converted_data)
    elif metrics == "llama3_70b_judge":
        results = compute_llama3_70b_judge(converted_data, binary=use_binary)
    elif metrics == "gpt4o_judge":
        results = compute_gpt4o_judge(converted_data, binary=use_binary)
    elif metrics == "string_match":
        results = compute_string_match(converted_data)
    else:
        raise ValueError(f"Unsupported metric: {metrics}")
    
    # Add metadata
    results["metadata"] = {
        "input_jsonl": input_jsonl,
        "metrics": metrics,
        "num_samples": len(data),
        "task_types": list(task_types),
    }
    
    # Log summary
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    if metrics == "wer":
        logger.info(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    elif metrics == "bleu":
        logger.info(f"BLEU: {results['bleu']:.2f}")
    elif metrics == "acc":
        logger.info(f"Accuracy: {results['acc']:.4f} ({results['acc']*100:.2f}%)")
    elif metrics in ["llama3_70b_judge", "gpt4o_judge"]:
        judge_key = metrics
        logger.info(f"Judge Score: {results[judge_key]['judge_score']:.2f}")
        logger.info(f"Success Rate: {results[judge_key]['success_rate']:.2%}")
    elif metrics == "string_match":
        logger.info(f"Accuracy: {results['string_match']['judge_score']:.2f}%")
    logger.info("=" * 50)
    
    # Save results
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inference results using AudioBench metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with sllm_response"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        choices=["wer", "bleu", "acc", "llama3_70b_judge", "gpt4o_judge", "string_match"],
        help="Evaluation metric to use"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results (optional)"
    )
    
    args = parser.parse_args()
    
    evaluate(
        input_jsonl=args.input_jsonl,
        metrics=args.metrics,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
