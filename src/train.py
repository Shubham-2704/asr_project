"""
train.py - Whisper Fine-Tuning on Hindi Data (Question 1b)
=========================================================
Fine-tunes openai/whisper-small on the preprocessed Hindi dataset.

OPTIMIZED FOR CPU:
- Uses smaller batch sizes
- Gradient accumulation for effective larger batches
- No fp16 (CPU doesn't support it well)
- Gradient checkpointing to save memory

Usage:
    python src/train.py                         # Default settings
    python src/train.py --epochs 3              # Custom epochs
    python src/train.py --batch-size 2          # Smaller batch for low memory
    python src/train.py --max-steps 100         # Quick test run
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "whisper-small-hi-finetuned")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "manifest.json")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.

    Handles:
    - Padding input features (log-mel spectrograms) to same length
    - Padding labels (tokenized text) with -100 for loss masking
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 (ignored by loss function)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was prepended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_manifest(manifest_path: str) -> list:
    """
    Load the preprocessed dataset manifest.

    Args:
        manifest_path: Path to manifest.json

    Returns:
        List of utterance dicts with audio_path and text
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[✓] Loaded {len(data)} utterances from manifest.")
    return data


def prepare_dataset(utterances: list, processor, max_duration: float = 30.0):
    """
    Convert raw utterances into Whisper-compatible features.

    Each utterance gets:
    - input_features: log-mel spectrogram from audio
    - labels: tokenized transcription text

    Args:
        utterances: List of dicts with audio_path and text
        processor: WhisperProcessor
        max_duration: Max audio duration in seconds

    Returns:
        List of feature dicts
    """
    import soundfile as sf
    import librosa

    features = []
    skipped = 0

    print("[→] Preparing features...")
    for utt in utterances:
        try:
            # Load audio
            audio, sr = sf.read(utt["audio_path"])

            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            duration = len(audio) / 16000

            # Skip if too long (Whisper max is 30s)
            if duration > max_duration:
                skipped += 1
                continue

            # Skip if too short
            if duration < 0.5:
                skipped += 1
                continue

            # Extract log-mel spectrogram features
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000
            ).input_features[0]

            # Tokenize the transcription
            labels = processor.tokenizer(utt["text"]).input_ids

            features.append({
                "input_features": input_features,
                "labels": labels,
            })

        except Exception as e:
            print(f"  [!] Error processing {utt['audio_path']}: {e}")
            skipped += 1

    print(f"[✓] Prepared {len(features)} features (skipped {skipped}).")
    return features


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Hindi ASR data")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size (default: 2, keep small for CPU)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max-steps", type=int, default=2,
                        help="Max training steps (-1 for full training)")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small",
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 2: WHISPER FINE-TUNING (CPU Optimized)")
    print("=" * 60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}")
    if device == "cpu":
        print("[i] Running on CPU - training will be slower.")
        print(f"[i] Using batch_size={args.batch_size}, grad_accum={args.gradient_accumulation}")
        print(f"[i] Effective batch size: {args.batch_size * args.gradient_accumulation}")

    # ===== Load Model & Processor =====
    print("\n[→] Loading Whisper model and processor...")
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language="hi",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Note: Disabled gradient checkpointing to avoid PyTorch Retain_Graph RuntimeError on CPU
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True

    # Set Hindi language for generation
    model.generation_config.language = "hi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="hi", task="transcribe"
    )

    print(f"[✓] Model loaded: {args.model_name}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===== Load & Prepare Dataset =====
    utterances = load_manifest(MANIFEST_PATH)
    features = prepare_dataset(utterances, processor)

    if not features:
        print("[✗] No features prepared. Check preprocessing output.")
        sys.exit(1)

    # Split into train/eval (90/10)
    split_idx = int(len(features) * 0.9)
    train_features = features[:split_idx]
    eval_features = features[split_idx:]

    print(f"[✓] Train: {len(train_features)} | Eval: {len(eval_features)}")

    # ===== Create Simple Dataset Class =====
    class ASRDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = ASRDataset(train_features)
    eval_dataset = ASRDataset(eval_features)

    # ===== Data Collator =====
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ===== Training Arguments (CPU Optimized) =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=50,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs if args.max_steps == -1 else -1,

        # CPU-specific optimizations
        fp16=False,                     # CPU doesn't benefit from fp16
        gradient_checkpointing=False,   # Disabled to prevent retain_graph RuntimeError
        optim="adamw_torch",            # Standard optimizer
        dataloader_num_workers=0,       # Avoid multiprocessing issues on Windows

        # Evaluation & saving
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        

        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to=[],  # Disable wandb/tensorboard

        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=225,
    )

    # ===== Trainer =====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    # ===== Train! =====
    print("\n[→] Starting training...")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Gradient accumulation: {args.gradient_accumulation}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Max steps: {args.max_steps if args.max_steps > 0 else 'Full training'}")

    trainer.train()

    # ===== Save =====
    print("\n[→] Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"\n[✓] Training complete!")
    print(f"    Model saved to: {args.output_dir}")
    print(f"    Next step: python src/evaluate.py")


if __name__ == "__main__":
    main()
