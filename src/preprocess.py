"""
preprocess.py - Data Download & Preprocessing (Question 1a)
==========================================================
This script handles:
1. Loading the CSV manifest (FT Data - data.csv)
2. Transforming URLs to the correct GCS endpoints
3. Downloading audio files (WAV) and transcription JSONs
4. Parsing transcriptions and segmenting audio into utterances
5. Building a HuggingFace Dataset for Whisper fine-tuning

Usage:
    python src/preprocess.py                    # Full download + preprocessing
    python src/preprocess.py --max-samples 5    # Only process first 5 recordings
    python src/preprocess.py --skip-download    # Skip download, just build dataset
"""

import os
import sys
import json
import argparse
import csv
import time

import requests
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.url_helper import transform_url, get_recording_id_from_url
from utils.text_utils import normalize_hindi_text

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPTION_DIR = os.path.join(DATA_DIR, "transcriptions")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CSV_PATH = os.path.join(PROJECT_ROOT, "FT Data - data.csv")

# Whisper expects 16kHz audio
TARGET_SAMPLE_RATE = 16000


def create_directories():
    """Create all required data directories."""
    for d in [DATA_DIR, AUDIO_DIR, TRANSCRIPTION_DIR, METADATA_DIR, PROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[✓] Data directories created.")


def load_csv_manifest(csv_path: str, max_samples: int = None) -> list:
    """
    Load the dataset CSV manifest.

    Args:
        csv_path: Path to the FT Data CSV file
        max_samples: If set, only load first N samples

    Returns:
        List of dicts with keys: user_id, recording_id, language, duration,
        rec_url_gcp, transcription_url_gcp, metadata_url_gcp
    """
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            records.append(row)

    print(f"[✓] Loaded {len(records)} records from CSV.")
    return records


def download_file(url: str, save_path: str, retries: int = 3) -> bool:
    """
    Download a file from a URL with retry logic.

    Args:
        url: URL to download from
        save_path: Local path to save the file
        retries: Number of retry attempts

    Returns:
        True if download succeeded, False otherwise
    """
    if os.path.exists(save_path):
        return True  # Already downloaded

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"  [!] HTTP {response.status_code} for {url}")
        except requests.RequestException as e:
            print(f"  [!] Attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(2)

    return False


def download_all_data(records: list) -> dict:
    """
    Download audio, transcription, and metadata files for all records.

    Args:
        records: List of record dicts from CSV

    Returns:
        Dict mapping recording_id -> {audio_path, transcription_path, metadata_path}
    """
    downloaded = {}
    failed = []

    print(f"\n[→] Downloading data for {len(records)} recordings...")

    for record in tqdm(records, desc="Downloading"):
        rec_id = record["recording_id"]

        # Transform URLs to the new format
        audio_url = transform_url(record["rec_url_gcp"])
        trans_url = transform_url(record["transcription_url_gcp"])
        meta_url = transform_url(record["metadata_url_gcp"])

        # Define save paths
        audio_path = os.path.join(AUDIO_DIR, f"{rec_id}_audio.wav")
        trans_path = os.path.join(TRANSCRIPTION_DIR, f"{rec_id}_transcription.json")
        meta_path = os.path.join(METADATA_DIR, f"{rec_id}_metadata.json")

        # Download each file
        audio_ok = download_file(audio_url, audio_path)
        trans_ok = download_file(trans_url, trans_path)
        meta_ok = download_file(meta_url, meta_path)

        if audio_ok and trans_ok:
            downloaded[rec_id] = {
                "audio_path": audio_path,
                "transcription_path": trans_path,
                "metadata_path": meta_path if meta_ok else None,
                "duration": float(record.get("duration", 0)),
                "user_id": record["user_id"],
            }
        else:
            failed.append(rec_id)

    print(f"[✓] Downloaded: {len(downloaded)} | Failed: {len(failed)}")
    if failed:
        print(f"    Failed IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return downloaded


def parse_transcription(trans_path: str) -> list:
    """
    Parse a transcription JSON file to extract text segments.

    The JSON structure may vary, but typically contains:
    - A list of segments with text and timestamps
    - Or a single transcription string

    Args:
        trans_path: Path to the transcription JSON

    Returns:
        List of dicts with keys: text, start_time, end_time
        Each represents one utterance segment
    """
    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  [!] Error reading {trans_path}: {e}")
        return []

    segments = []

    # Handle different JSON formats
    if isinstance(data, list):
        # Format: list of segments
        for item in data:
            if isinstance(item, dict):
                text = item.get("text", item.get("transcription", ""))
                start = item.get("start", item.get("start_time", 0))
                end = item.get("end", item.get("end_time", 0))
                if text.strip():
                    segments.append({
                        "text": text.strip(),
                        "start_time": float(start),
                        "end_time": float(end),
                    })
            elif isinstance(item, str) and item.strip():
                segments.append({
                    "text": item.strip(),
                    "start_time": 0,
                    "end_time": 0,
                })

    elif isinstance(data, dict):
        # Format: dict with segments key or direct transcription
        if "segments" in data:
            for seg in data["segments"]:
                text = seg.get("text", "")
                if text.strip():
                    segments.append({
                        "text": text.strip(),
                        "start_time": float(seg.get("start", 0)),
                        "end_time": float(seg.get("end", 0)),
                    })
        elif "transcription" in data:
            segments.append({
                "text": data["transcription"].strip(),
                "start_time": 0,
                "end_time": float(data.get("duration", 0)),
            })
        elif "text" in data:
            segments.append({
                "text": data["text"].strip(),
                "start_time": 0,
                "end_time": float(data.get("duration", 0)),
            })

    elif isinstance(data, str) and data.strip():
        segments.append({
            "text": data.strip(),
            "start_time": 0,
            "end_time": 0,
        })

    return segments


def segment_audio(audio_path: str, segments: list, rec_id: str) -> list:
    """
    Segment a long audio file into utterance-level chunks based on timestamps.
    If no timestamps are available, use the full audio.

    Whisper works best with segments under 30 seconds.

    Args:
        audio_path: Path to the full audio WAV file
        segments: List of segment dicts with start_time, end_time, text
        rec_id: Recording ID for naming output files

    Returns:
        List of dicts with: audio_path, text, duration
    """
    try:
        audio_data, sr = sf.read(audio_path)
    except Exception as e:
        print(f"  [!] Error reading audio {audio_path}: {e}")
        return []

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        try:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
        except ImportError:
            print("  [!] librosa not available for resampling. Using original sample rate.")

    utterances = []
    max_segment_duration = 30  # seconds (Whisper's max input length)

    for i, seg in enumerate(segments):
        start = seg["start_time"]
        end = seg["end_time"]
        text = seg["text"]

        # If no timestamps, use full audio
        if start == 0 and end == 0:
            # Split long audio into chunks
            total_duration = len(audio_data) / sr
            if total_duration > max_segment_duration:
                # Split into 30-second chunks with overlap
                chunk_size = int(max_segment_duration * sr)
                for j in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[j:j + chunk_size]
                    chunk_path = os.path.join(PROCESSED_DIR, f"{rec_id}_chunk_{j // chunk_size}.wav")
                    sf.write(chunk_path, chunk, sr)
                    utterances.append({
                        "audio_path": chunk_path,
                        "text": text,  # Full text for each chunk (will be handled during training)
                        "duration": len(chunk) / sr,
                    })
            else:
                seg_path = os.path.join(PROCESSED_DIR, f"{rec_id}_full.wav")
                sf.write(seg_path, audio_data, sr)
                utterances.append({
                    "audio_path": seg_path,
                    "text": text,
                    "duration": total_duration,
                })
        else:
            # Use timestamps to extract segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio_data = audio_data[start_sample:end_sample]

            if len(segment_audio_data) == 0:
                continue

            seg_path = os.path.join(PROCESSED_DIR, f"{rec_id}_seg_{i}.wav")
            sf.write(seg_path, segment_audio_data, sr)

            utterances.append({
                "audio_path": seg_path,
                "text": text,
                "duration": len(segment_audio_data) / sr,
            })

    return utterances


def build_dataset(downloaded: dict) -> list:
    """
    Build a dataset from downloaded audio + transcriptions.
    Each entry has: audio_path, text, duration

    Args:
        downloaded: Dict from download_all_data()

    Returns:
        List of utterance dicts ready for training
    """
    print("\n[→] Building dataset from downloaded files...")
    all_utterances = []

    for rec_id, info in tqdm(downloaded.items(), desc="Processing"):
        # Parse transcription
        segments = parse_transcription(info["transcription_path"])
        if not segments:
            continue

        # Segment audio based on timestamps
        utterances = segment_audio(info["audio_path"], segments, rec_id)
        all_utterances.extend(utterances)

    print(f"[✓] Total utterances: {len(all_utterances)}")
    return all_utterances


def save_dataset(utterances: list, output_dir: str):
    """
    Save the processed dataset as a JSON manifest.

    Args:
        utterances: List of utterance dicts
        output_dir: Directory to save the manifest
    """
    manifest_path = os.path.join(output_dir, "manifest.json")

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(utterances, f, ensure_ascii=False, indent=2)

    print(f"[✓] Dataset manifest saved to: {manifest_path}")
    print(f"    Total utterances: {len(utterances)}")

    # Print statistics
    if utterances:
        durations = [u["duration"] for u in utterances]
        total_hours = sum(durations) / 3600
        print(f"    Total audio: {total_hours:.2f} hours")
        print(f"    Avg duration: {sum(durations)/len(durations):.1f}s")
        print(f"    Min duration: {min(durations):.1f}s")
        print(f"    Max duration: {max(durations):.1f}s")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess Hindi ASR training data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only process first N recordings (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step, use existing files")
    parser.add_argument("--csv-path", type=str, default=CSV_PATH,
                        help="Path to the data CSV file")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Step 1: Create directories
    create_directories()

    # Step 2: Load CSV
    records = load_csv_manifest(args.csv_path, args.max_samples)

    # Step 3: Download data
    if args.skip_download:
        print("[→] Skipping download, using existing files...")
        downloaded = {}
        for record in records:
            rec_id = record["recording_id"]
            audio_path = os.path.join(AUDIO_DIR, f"{rec_id}_audio.wav")
            trans_path = os.path.join(TRANSCRIPTION_DIR, f"{rec_id}_transcription.json")
            meta_path = os.path.join(METADATA_DIR, f"{rec_id}_metadata.json")
            if os.path.exists(audio_path) and os.path.exists(trans_path):
                downloaded[rec_id] = {
                    "audio_path": audio_path,
                    "transcription_path": trans_path,
                    "metadata_path": meta_path if os.path.exists(meta_path) else None,
                    "duration": float(record.get("duration", 0)),
                    "user_id": record["user_id"],
                }
    else:
        downloaded = download_all_data(records)

    if not downloaded:
        print("[✗] No data was downloaded/found. Check URLs and network.")
        sys.exit(1)

    # Step 4: Build dataset
    utterances = build_dataset(downloaded)

    # Step 5: Save
    save_dataset(utterances, PROCESSED_DIR)

    print("\n[✓] Preprocessing complete!")
    print(f"    Next step: python src/train.py")


if __name__ == "__main__":
    main()
