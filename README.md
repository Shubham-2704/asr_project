# Hindi ASR Pipeline — Josh Talks Assignment

> AI Researcher Intern — Speech & Audio

## Overview

This project implements a complete Hindi Automatic Speech Recognition (ASR) pipeline covering all 4 assignment questions:

| Question | Module | Description |
|----------|--------|-------------|
| **Q1** | `preprocess.py`, `train.py`, `evaluate.py`, `error_analysis.py` | Whisper fine-tuning, WER evaluation, error taxonomy |
| **Q2a** | `cleanup.py` | Number normalization (Hindi words → digits) |
| **Q2b** | `english_detect.py` | English word detection & tagging in Hindi text |
| **Q3** | `spelling.py` | Classify ~1,75,000 words as correct/incorrect spelling |
| **Q4** | `lattice.py` | Lattice-based WER evaluation with multi-model consensus |

## Project Structure

```
voice_assignment/
├── data/                          # Downloaded data (created automatically)
│   ├── audio/                     # WAV audio files
│   ├── transcriptions/            # Transcription JSONs
│   ├── metadata/                  # Metadata JSONs
│   └── processed/                 # Processed dataset + manifest
├── src/
│   ├── preprocess.py              # Q1a: Data download + preprocessing
│   ├── train.py                   # Q1b: Whisper fine-tuning (CPU optimized)
│   ├── evaluate.py                # Q1b-c: WER evaluation on FLEURS Hindi
│   ├── error_analysis.py          # Q1d-g: Error sampling, taxonomy, fixes
│   ├── cleanup.py                 # Q2a: Hindi number normalization
│   ├── english_detect.py          # Q2b: English word detection
│   ├── spelling.py                # Q3: Spelling classification
│   └── lattice.py                 # Q4: Lattice-based WER
├── utils/
│   ├── url_helper.py              # GCS URL transformation
│   ├── hindi_numbers.py           # Hindi number word → digit converter
│   └── text_utils.py              # Shared text processing utilities
├── outputs/                       # All generated results
│   ├── wer_results.csv            # WER comparison table
│   ├── error_taxonomy.csv         # Error categories + examples
│   ├── cleanup_examples.csv       # Number normalization before/after
│   ├── english_tagged.csv         # English-tagged transcripts
│   ├── spelling_results.csv       # Word classification results
│   └── lattice_wer.csv            # Lattice-based WER per model
├── main.py                        # Master pipeline runner
├── requirements.txt               # Dependencies
├── .env                           # HuggingFace token (fill in yourself)
├── .gitignore                     # Ignores large datasets and model weights
└── README.md                      # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your HuggingFace Token

Edit the `.env` file and add your token:

```
HF_TOKEN=hf_your_actual_token_here
```

You can get a token at: https://huggingface.co/settings/tokens

## Running the Pipeline

### Full Pipeline (All Questions)

```bash
python main.py
```

### Individual Steps

```bash
# Q1: Preprocessing → Training → Evaluation → Error Analysis
python main.py --step preprocess
python main.py --step train
python main.py --step evaluate
python main.py --step error_analysis

# Q2: Cleanup Pipeline
python main.py --step cleanup            # Number normalization
python main.py --step english_detect     # English word detection

# Q3: Spelling Classification
python main.py --step spelling

# Q4: Lattice-Based WER
python main.py --step lattice
```

### Run Multiple Steps

```bash
python main.py --steps spelling,lattice
```

### Direct Script Execution

Each script can also be run directly:

```bash
python src/preprocess.py --max-samples 5     # Test with 5 recordings
python src/train.py --epochs 1 --max-steps 50  # Quick training test
python src/evaluate.py --pretrained-only       # Only evaluate pretrained
python src/spelling.py                         # Classify all words
python src/lattice.py                          # Lattice evaluation
```

## CPU Optimization

Training is optimized for CPU execution:
- Batch size: 2 (small to fit in memory)
- Gradient accumulation: 8 steps (effective batch = 16)
- No FP16 (CPU doesn't benefit from half precision)
- Gradient checkpointing enabled (saves memory)

**Expected training time on CPU:** Several hours depending on hardware.

For faster iteration, use `--max-steps` to limit training:

```bash
python src/train.py --max-steps 100    # Quick test (~15 min)
```

## Output Files

After running, check the `outputs/` folder for:

| File | Content |
|------|---------|
| `wer_results.csv` | WER comparison: pretrained vs fine-tuned |
| `error_taxonomy.csv` | Error categories with examples |
| `error_examples.json` | 25+ sampled error utterances |
| `cleanup_examples.csv` | Number normalization before/after |
| `english_tagged.csv` | Transcripts with [EN] tags |
| `spelling_results.csv` | 1,75,000 words classified |
| `lattice_wer.csv` | Standard vs lattice WER per model |

## Libraries Used

- `transformers` — Whisper model loading + fine-tuning
- `datasets` — FLEURS dataset loading
- `torch` — PyTorch backend
- `jiwer` — WER calculation
- `librosa` — Audio processing
- `soundfile` — Audio I/O
- `pandas` — Data handling
- `tqdm` — Progress bars
- `python-dotenv` — Environment variable loading
