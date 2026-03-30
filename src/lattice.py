"""
lattice.py - Lattice-Based WER Evaluation (Question 4)
=====================================================
Constructs a lattice from multiple ASR model outputs + human reference
to compute a fairer WER that accounts for valid transcription alternatives.

Key concepts:
- A lattice replaces a rigid reference with positional bins of valid alternatives
- Each bin contains valid lexical, phonetic, and spelling variations
- WER is computed against the lattice (any valid alternative counts as correct)

Alignment unit: WORD (justified because Hindi is space-delimited and
ASR evaluation tradition uses word-level metrics; subword would add complexity
without clear benefit for this conversational data).

Usage:
    python src/lattice.py
    python src/lattice.py --input "Question 4 - Task.csv"
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict, Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import normalize_hindi_text

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "Question 4 - Task.csv")


def load_q4_data(input_path: str) -> list:
    """
    Load the Q4 dataset: multiple ASR model outputs + human reference.

    CSV columns: segment_url_link, Human, Model H, Model i, Model k, Model l, Model m, Model n

    Args:
        input_path: Path to Q4 CSV file

    Returns:
        List of dicts with: url, human, models (dict of model_name -> transcript)
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Identify model columns (everything except segment_url_link and Human)
        model_cols = [h.strip() for h in headers
                      if h.strip() and h.strip() not in ("segment_url_link", "Human")]

        for row in reader:
            entry = {
                "url": row.get("segment_url_link", "").strip(),
                "human": row.get("Human", "").strip(),
                "models": {}
            }
            for col in model_cols:
                transcript = row.get(col, "").strip()
                if transcript:
                    entry["models"][col.strip()] = transcript

            if entry["human"]:
                data.append(entry)

    print(f"[✓] Loaded {len(data)} segments with {len(model_cols)} models.")
    print(f"    Models: {', '.join(model_cols)}")
    return data, model_cols


def word_align(ref_words: list, hyp_words: list) -> list:
    """
    Align two word sequences using dynamic programming (edit distance alignment).

    Returns alignment operations: 'match', 'substitution', 'insertion', 'deletion'.

    Args:
        ref_words: Reference word list
        hyp_words: Hypothesis word list

    Returns:
        List of (operation, ref_word, hyp_word) tuples
    """
    n = len(ref_words)
    m = len(hyp_words)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        op[i][0] = "D"  # deletion
    for j in range(m + 1):
        dp[0][j] = j
        op[0][j] = "I"  # insertion

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
                op[i][j] = "M"  # match
            else:
                costs = [
                    (dp[i-1][j] + 1, "D"),      # deletion
                    (dp[i][j-1] + 1, "I"),      # insertion
                    (dp[i-1][j-1] + 1, "S"),    # substitution
                ]
                dp[i][j], op[i][j] = min(costs, key=lambda x: x[0])

    # Traceback
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and op[i][j] == "M":
            alignment.append(("match", ref_words[i-1], hyp_words[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and op[i][j] == "S":
            alignment.append(("substitution", ref_words[i-1], hyp_words[j-1]))
            i -= 1; j -= 1
        elif j > 0 and op[i][j] == "I":
            alignment.append(("insertion", "", hyp_words[j-1]))
            j -= 1
        elif i > 0:
            alignment.append(("deletion", ref_words[i-1], ""))
            i -= 1
        else:
            break

    alignment.reverse()
    return alignment


def normalize_word(word: str) -> str:
    """
    Normalize a word for comparison (remove punctuation, lowercase).

    Args:
        word: Input word

    Returns:
        Normalized word
    """
    # Remove common punctuation
    word = word.strip().rstrip("।,!?;:.\"'-–—")
    word = word.lstrip("\"'")
    word = word.lower()
    return word


def build_lattice(human_ref: str, model_outputs: dict) -> list:
    """
    Construct a lattice from human reference and multiple model outputs.

    Strategy:
    1. Start with human reference words as the backbone
    2. Align each model output to the reference
    3. At each position, collect all valid alternatives
    4. Trust model agreement: if 3+ models agree on a word different from
       the reference, include it as a valid alternative
    5. Handle insertions, deletions, and substitutions

    Args:
        human_ref: Human reference transcript
        model_outputs: Dict of model_name -> transcript

    Returns:
        List of sets, where each set contains valid words at that position
    """
    ref_words = human_ref.split()
    num_positions = len(ref_words)

    # Initialize lattice with reference words
    lattice = [set() for _ in range(num_positions)]
    for i, word in enumerate(ref_words):
        lattice[i].add(normalize_word(word))

    # Track model agreement at each position
    position_votes = [Counter() for _ in range(num_positions)]

    # Align each model output to reference
    for model_name, transcript in model_outputs.items():
        model_words = transcript.split()
        alignment = word_align(ref_words, model_words)

        ref_pos = 0
        for op, ref_word, hyp_word in alignment:
            if op == "match":
                norm_hyp = normalize_word(hyp_word)
                lattice[ref_pos].add(norm_hyp)
                position_votes[ref_pos][norm_hyp] += 1
                ref_pos += 1
            elif op == "substitution":
                norm_hyp = normalize_word(hyp_word)
                position_votes[ref_pos][norm_hyp] += 1

                # Check if it's a valid alternative:
                # - Minor spelling variation (e.g., हाँ vs हां)
                # - Number format (चौदह vs 14)
                # - Common variation (सब्ज़ी vs सब्जी)
                norm_ref = normalize_word(ref_word)

                # Accept if character overlap is high (spelling variation)
                ref_chars = set(norm_ref)
                hyp_chars = set(norm_hyp)
                if ref_chars and hyp_chars:
                    overlap = len(ref_chars & hyp_chars) / max(len(ref_chars), len(hyp_chars))
                    if overlap > 0.5:
                        lattice[ref_pos].add(norm_hyp)

                ref_pos += 1
            elif op == "deletion":
                # Reference word was deleted in model output
                ref_pos += 1
            elif op == "insertion":
                # Model inserted an extra word - skip (don't advance ref position)
                pass

    # Trust model agreement: if 3+ models agree on a substitution, add it
    num_models = len(model_outputs)
    agreement_threshold = max(3, num_models // 2)

    for pos in range(num_positions):
        for word, count in position_votes[pos].items():
            if count >= agreement_threshold:
                lattice[pos].add(word)

    return lattice


def compute_lattice_wer(lattice: list, hypothesis: str) -> dict:
    """
    Compute WER for a hypothesis against a lattice reference.

    At each position, if the hypothesis word matches ANY word in the
    lattice bin, it counts as correct.

    Uses dynamic programming alignment where match cost is 0 if the
    hypothesis word appears in the corresponding lattice bin.

    Args:
        lattice: List of sets of valid alternatives at each position
        hypothesis: Model output transcript

    Returns:
        Dict with: wer, errors (substitutions, insertions, deletions)
    """
    hyp_words = [normalize_word(w) for w in hypothesis.split()]
    n = len(lattice)
    m = len(hyp_words)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Match: does hyp word appear in lattice bin?
            if hyp_words[j-1] in lattice[i-1]:
                dp[i][j] = dp[i-1][j-1]  # No cost for correct match
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,       # deletion
                    dp[i][j-1] + 1,       # insertion
                    dp[i-1][j-1] + 1,     # substitution
                )

    total_errors = dp[n][m]
    total_ref = n  # Reference length = number of lattice positions

    wer = total_errors / total_ref if total_ref > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "errors": total_errors,
        "ref_length": total_ref,
        "hyp_length": m,
    }


def compute_standard_wer(reference: str, hypothesis: str) -> float:
    """
    Compute standard (rigid) WER for comparison.

    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript

    Returns:
        WER as float
    """
    ref_words = [normalize_word(w) for w in reference.split()]
    hyp_words = [normalize_word(w) for w in hypothesis.split()]

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)

    return round(dp[n][m] / n, 4) if n > 0 else 0.0


def save_results(data: list, model_cols: list, output_dir: str):
    """
    Save lattice WER results with comparison to standard WER.

    Args:
        data: Processed data with WER results
        model_cols: List of model column names
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Lattice WER CSV =====
    csv_path = os.path.join(output_dir, "lattice_wer.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Segment_URL"]
        for model in model_cols:
            header.extend([f"{model}_Standard_WER", f"{model}_Lattice_WER", f"{model}_Improvement"])
        writer.writerow(header)

        for entry in data:
            row = [entry["url"]]
            for model in model_cols:
                if model in entry.get("results", {}):
                    r = entry["results"][model]
                    std_wer = r.get("standard_wer", 0)
                    lat_wer = r.get("lattice_wer", 0)
                    improvement = round(std_wer - lat_wer, 4)
                    row.extend([std_wer, lat_wer, improvement])
                else:
                    row.extend(["N/A", "N/A", "N/A"])
            writer.writerow(row)

    print(f"[✓] Lattice WER results saved to: {csv_path}")

    # ===== Summary JSON =====
    # Compute average WERs per model
    model_avg = {}
    for model in model_cols:
        std_wers = []
        lat_wers = []
        for entry in data:
            if model in entry.get("results", {}):
                std_wers.append(entry["results"][model]["standard_wer"])
                lat_wers.append(entry["results"][model]["lattice_wer"])
        if std_wers:
            model_avg[model] = {
                "avg_standard_wer": round(sum(std_wers) / len(std_wers), 4),
                "avg_lattice_wer": round(sum(lat_wers) / len(lat_wers), 4),
                "avg_improvement": round(
                    (sum(std_wers) - sum(lat_wers)) / len(std_wers), 4
                ),
                "num_segments": len(std_wers),
            }

    summary = {
        "alignment_unit": "word",
        "alignment_justification": (
            "Word-level alignment is chosen because: "
            "(1) Hindi is space-delimited, making word boundaries clear; "
            "(2) Standard ASR evaluation uses WER (word error rate); "
            "(3) Subword/character alignment would add complexity without "
            "clear benefit for conversational Hindi evaluations."
        ),
        "lattice_construction": (
            "The lattice is built by: "
            "(1) Using human reference as backbone; "
            "(2) Aligning each model output using DP edit-distance; "
            "(3) Adding alternatives at each position from model outputs; "
            "(4) Trusting model agreement: 3+ models agreeing on a word overwrites the reference; "
            "(5) Accepting spelling variations with >50% character overlap."
        ),
        "trust_model_agreement": (
            "If 3 or more models produce the same word at a position, "
            "and the human reference has a different word, the model consensus "
            "is added to the lattice as a valid alternative. This handles cases "
            "where the human reference contains errors."
        ),
        "model_averages": model_avg,
        "total_segments": len(data),
    }

    json_path = os.path.join(output_dir, "lattice_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] Lattice analysis saved to: {json_path}")

    # ===== Print Summary Table =====
    print(f"\n{'=' * 70}")
    print(f"  LATTICE WER COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Standard WER':<15} {'Lattice WER':<15} {'Improvement':<15}")
    print(f"{'-' * 70}")

    for model in model_cols:
        if model in model_avg:
            avg = model_avg[model]
            print(f"{model:<15} {avg['avg_standard_wer']:<15.4f} "
                  f"{avg['avg_lattice_wer']:<15.4f} {avg['avg_improvement']:<15.4f}")

    print(f"{'=' * 70}")
    print(f"\n  Alignment unit: WORD")
    print(f"  Key insight: Lattice-based evaluation reduces WER for models")
    print(f"  that were unfairly penalized by rigid reference matching.")


def main():
    """Main lattice evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Lattice-Based WER Evaluation")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Path to Q4 data CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 8: LATTICE-BASED WER EVALUATION (Q4)")
    print("=" * 60)

    # Check input file
    if not os.path.exists(args.input):
        print(f"[✗] Input file not found: {args.input}")
        sys.exit(1)

    # Step 1: Load data
    data, model_cols = load_q4_data(args.input)

    # Step 2: Process each segment
    print(f"\n[→] Building lattices and computing WER for {len(data)} segments...")

    for entry in data:
        # Build lattice from human + all model outputs
        lattice = build_lattice(entry["human"], entry["models"])

        entry["lattice"] = [list(bin_set) for bin_set in lattice]
        entry["results"] = {}

        # Compute both standard and lattice WER for each model
        for model_name, transcript in entry["models"].items():
            std_wer = compute_standard_wer(entry["human"], transcript)
            lat_result = compute_lattice_wer(lattice, transcript)

            entry["results"][model_name] = {
                "standard_wer": std_wer,
                "lattice_wer": lat_result["wer"],
                "errors": lat_result["errors"],
                "ref_length": lat_result["ref_length"],
            }

    # Step 3: Save results
    save_results(data, model_cols, OUTPUT_DIR)

    print(f"\n[✓] Lattice evaluation complete!")
    print(f"    All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
