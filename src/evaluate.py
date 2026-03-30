"""
evaluate.py - WER Evaluation on FLEURS Hindi (Question 1b-c)
============================================================
Evaluates both pretrained and fine-tuned Whisper-small on the
Hindi portion of the FLEURS test dataset.

Reports WER in a structured table for the assignment.

Usage:
    python src/evaluate.py                                    # Default
    python src/evaluate.py --model-path outputs/whisper-small-hi-finetuned
    python src/evaluate.py --max-samples 50                   # Quick evaluation
"""

import os
import sys
import json
import argparse
import csv

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import normalize_hindi_text

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_FT_PATH = os.path.join(OUTPUT_DIR, "whisper-small-hi-finetuned")


def load_fleurs_test(max_samples: int = None) -> list:
    """
    Load the Hindi portion of the FLEURS test dataset.

    FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
    contains read speech in 102 languages including Hindi.

    Args:
        max_samples: Maximum number of test samples to load

    Returns:
        List of dicts with: audio (numpy array), text (reference), sr (sample rate)
    """
    from datasets import load_dataset

    print("[→] Loading FLEURS Hindi test set...")

    # Load FLEURS Hindi test split
    try:
        fleurs = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"[!] Could not load FLEURS from HuggingFace: {e}")
        print("[!] Trying alternate loading method...")
        try:
            fleurs = load_dataset("google/fleurs", "hi_in", split="test")
        except Exception as e2:
            print(f"[✗] Failed to load FLEURS: {e2}")
            print("[i] Make sure HF_TOKEN is set in .env and you have internet access.")
            sys.exit(1)

    if max_samples:
        fleurs = fleurs.select(range(min(max_samples, len(fleurs))))

    print(f"[✓] Loaded {len(fleurs)} FLEURS Hindi test samples.")
    return fleurs


def transcribe_with_whisper(model, processor, audio_array, sr=16000) -> str:
    """
    Transcribe a single audio sample using Whisper.

    Args:
        model: WhisperForConditionalGeneration model
        processor: WhisperProcessor
        audio_array: Audio as numpy array
        sr: Sample rate

    Returns:
        Transcribed text string
    """
    # Resample if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # Prepare input features
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Move to same device as model
    device = next(model.parameters()).device
    input_features = input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="hi",
            task="transcribe",
            max_new_tokens=225,
        )

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def evaluate_model(model, processor, dataset, model_name: str) -> dict:
    """
    Evaluate a Whisper model on the FLEURS test set and compute WER.

    Args:
        model: Whisper model
        processor: Whisper processor
        dataset: FLEURS test dataset
        model_name: Name for logging

    Returns:
        Dict with: wer, predictions (list of dicts with ref, hyp, wer)
    """
    from jiwer import wer as compute_wer

    print(f"\n[→] Evaluating: {model_name}")

    references = []
    hypotheses = []
    predictions = []

    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        # Get audio and reference
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        reference = sample["transcription"]

        # Transcribe
        hypothesis = transcribe_with_whisper(model, processor, audio, sr)

        # Normalize both for fair comparison
        ref_norm = normalize_hindi_text(reference)
        hyp_norm = normalize_hindi_text(hypothesis)

        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        # Compute per-utterance WER
        try:
            utt_wer = compute_wer(ref_norm, hyp_norm) if ref_norm else 0.0
        except Exception:
            utt_wer = 1.0

        predictions.append({
            "id": i,
            "reference": reference,
            "hypothesis": hypothesis,
            "reference_normalized": ref_norm,
            "hypothesis_normalized": hyp_norm,
            "wer": round(utt_wer, 4),
        })

    # Compute overall WER
    overall_wer = compute_wer(references, hypotheses)

    print(f"[✓] {model_name} WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")

    return {
        "model_name": model_name,
        "wer": round(overall_wer, 4),
        "num_samples": len(references),
        "predictions": predictions,
    }


def save_results(pretrained_results: dict, finetuned_results: dict, output_dir: str):
    """
    Save WER comparison results to CSV and JSON.

    Args:
        pretrained_results: Results from pretrained model
        finetuned_results: Results from fine-tuned model (can be None)
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Save WER Table (CSV) =====
    wer_csv_path = os.path.join(output_dir, "wer_results.csv")
    with open(wer_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Hindi WER", "Num Samples"])
        writer.writerow([
            "Whisper Small (Pretrained)",
            f"{pretrained_results['wer']:.4f}",
            pretrained_results["num_samples"]
        ])
        if finetuned_results:
            writer.writerow([
                "FT Whisper Small (yours)",
                f"{finetuned_results['wer']:.4f}",
                finetuned_results["num_samples"]
            ])

    print(f"\n[✓] WER results saved to: {wer_csv_path}")

    # ===== Save Detailed Predictions (JSON) =====
    predictions_path = os.path.join(output_dir, "evaluation_predictions.json")
    all_predictions = {
        "pretrained": pretrained_results["predictions"],
    }
    if finetuned_results:
        all_predictions["finetuned"] = finetuned_results["predictions"]

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print(f"[✓] Detailed predictions saved to: {predictions_path}")

    # ===== Print Summary Table =====
    print("\n" + "=" * 50)
    print("  WER COMPARISON TABLE")
    print("=" * 50)
    print(f"{'Model':<35} {'Hindi WER':<12}")
    print("-" * 50)
    print(f"{'Whisper Small (Pretrained)':<35} {pretrained_results['wer']:.4f}")
    if finetuned_results:
        print(f"{'FT Whisper Small (yours)':<35} {finetuned_results['wer']:.4f}")
        improvement = pretrained_results['wer'] - finetuned_results['wer']
        print("-" * 50)
        print(f"{'Improvement':<35} {improvement:.4f} ({improvement*100:.2f}%)")
    print("=" * 50)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate Whisper models on FLEURS Hindi")
    parser.add_argument("--model-path", type=str, default=DEFAULT_FT_PATH,
                        help="Path to fine-tuned model")
    parser.add_argument("--max-samples", type=int, default=5,
                        help="Limit test samples (for quick testing)")
    parser.add_argument("--pretrained-only", action="store_true",
                        help="Only evaluate pretrained model")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 3: WER EVALUATION ON FLEURS HINDI")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}")

    # Load test dataset
    fleurs_test = load_fleurs_test(args.max_samples)

    # ===== Evaluate Pretrained Model =====
    print("\n[→] Loading pretrained Whisper-small...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    pretrained_processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="hi", task="transcribe"
    )
    pretrained_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device)
    pretrained_model.eval()

    pretrained_results = evaluate_model(
        pretrained_model, pretrained_processor, fleurs_test, "Whisper Small (Pretrained)"
    )

    # Free memory
    del pretrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== Evaluate Fine-tuned Model =====
    finetuned_results = None

    if not args.pretrained_only and os.path.exists(args.model_path):
        print(f"\n[→] Loading fine-tuned model from {args.model_path}...")
        try:
            ft_processor = WhisperProcessor.from_pretrained(args.model_path)
            ft_model = WhisperForConditionalGeneration.from_pretrained(
                args.model_path
            ).to(device)
            ft_model.eval()

            finetuned_results = evaluate_model(
                ft_model, ft_processor, fleurs_test, "FT Whisper Small (yours)"
            )

            del ft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[!] Could not load fine-tuned model: {e}")
            print("[i] Run 'python src/train.py' first to create the fine-tuned model.")
    elif not args.pretrained_only:
        print(f"\n[i] Fine-tuned model not found at: {args.model_path}")
        print("[i] Run 'python src/train.py' first, then re-run evaluation.")

    # ===== Save Results =====
    save_results(pretrained_results, finetuned_results, OUTPUT_DIR)

    print(f"\n[✓] Evaluation complete!")
    print(f"    Next step: python src/error_analysis.py")


if __name__ == "__main__":
    main()
