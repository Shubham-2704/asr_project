"""
error_analysis.py - Error Sampling, Taxonomy & Fix (Questions 1d-g)
==================================================================
This script:
  d) Systematically samples 25+ error utterances from evaluation
  e) Builds an error taxonomy with categories and examples
  f) Proposes fixes for top 3 error types
  g) Implements at least 1 fix with before/after results

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --predictions-path outputs/evaluation_predictions.json
"""

import os
import sys
import json
import csv
import re
import argparse
from collections import Counter, defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import normalize_hindi_text, is_devanagari_word, is_roman_word

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "evaluation_predictions.json")


# ===== Error Categories =====
# These categories are designed to emerge from Hindi ASR error patterns
ERROR_CATEGORIES = {
    "vowel_matra_error": {
        "name": "Vowel / Matra Error",
        "description": "Incorrect vowel marks (matras) - e.g., ा vs ो, ी vs ि",
        "patterns": [
            (r'ा', r'ो'), (r'ो', r'ा'),  # aa vs o
            (r'ी', r'ि'), (r'ि', r'ी'),  # ii vs i
            (r'ू', r'ु'), (r'ु', r'ू'),  # uu vs u
            (r'े', r'ै'), (r'ै', r'े'),  # e vs ai
        ],
    },
    "consonant_confusion": {
        "name": "Consonant Confusion",
        "description": "Similar-sounding consonants confused - e.g., ब vs व, ड vs ड़",
        "patterns": [
            (r'ब', r'व'), (r'व', r'ब'),
            (r'ड', r'ड़'), (r'ड़', r'ड'),
            (r'श', r'ष'), (r'ष', r'श'),
            (r'न', r'ण'), (r'ण', r'न'),
        ],
    },
    "word_boundary_error": {
        "name": "Word Boundary Error",
        "description": "Words merged or split incorrectly",
    },
    "code_switching_error": {
        "name": "Code-Switching Error (Hindi-English)",
        "description": "English words in Hindi conversation misrecognized or script-switched",
    },
    "homophone_confusion": {
        "name": "Homophone Confusion",
        "description": "Words that sound similar but are spelled differently",
    },
    "insertion_deletion": {
        "name": "Insertion / Deletion",
        "description": "Extra words added or words dropped from transcription",
    },
    "number_transcription": {
        "name": "Number Transcription Error",
        "description": "Numbers written as digits vs words, or wrong number words",
    },
    "punctuation_filler": {
        "name": "Punctuation / Filler Word Error",
        "description": "Filler words (हम्म, अ) or punctuation differences",
    },
    "visarga_anusvara": {
        "name": "Visarga / Anusvara / Chandrabindu Error",
        "description": "Nasal marks (ं, ँ) or visarga (ः) errors",
    },
    "other": {
        "name": "Other / Unclassified",
        "description": "Errors not fitting other categories",
    },
}


def load_predictions(predictions_path: str) -> list:
    """
    Load evaluation predictions from JSON file.

    Args:
        predictions_path: Path to evaluation_predictions.json

    Returns:
        List of prediction dicts (from the best available model)
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prefer fine-tuned predictions; fall back to pretrained
    if "finetuned" in data and data["finetuned"]:
        preds = data["finetuned"]
        model_name = "Fine-tuned Whisper Small"
    elif "pretrained" in data and data["pretrained"]:
        preds = data["pretrained"]
        model_name = "Pretrained Whisper Small"
    else:
        print("[✗] No predictions found in the file.")
        sys.exit(1)

    print(f"[✓] Loaded {len(preds)} predictions from {model_name}.")
    return preds, model_name


def sample_errors(predictions: list, min_samples: int = 25) -> list:
    """
    Systematically sample error utterances.

    Sampling strategy (Q1d):
    1. Filter to only utterances with WER > 0
    2. Stratify by WER severity: low (0-0.3), medium (0.3-0.6), high (0.6+)
    3. Take proportional samples from each stratum
    4. Ensure minimum of 25 samples

    Args:
        predictions: List of prediction dicts with 'wer' field
        min_samples: Minimum number of error samples

    Returns:
        List of sampled error prediction dicts
    """
    # Filter errors only
    errors = [p for p in predictions if p["wer"] > 0]

    if not errors:
        print("[!] No errors found in predictions.")
        return []

    # Stratify by severity
    low = [p for p in errors if p["wer"] <= 0.3]      # Minor errors
    medium = [p for p in errors if 0.3 < p["wer"] <= 0.6]  # Moderate errors
    high = [p for p in errors if p["wer"] > 0.6]       # Severe errors

    print(f"[i] Error distribution: Low={len(low)}, Medium={len(medium)}, High={len(high)}")

    # Proportional sampling
    total = len(errors)
    target = max(min_samples, 25)

    sampled = []
    for stratum, name in [(low, "low"), (medium, "medium"), (high, "high")]:
        if not stratum:
            continue
        n = max(1, int(target * len(stratum) / total))
        step = max(1, len(stratum) // n)
        selected = stratum[::step][:n]
        for s in selected:
            s["severity"] = name
        sampled.extend(selected)

    # Ensure we have at least min_samples
    if len(sampled) < min_samples:
        remaining = [p for p in errors if p not in sampled]
        needed = min_samples - len(sampled)
        step = max(1, len(remaining) // needed)
        extra = remaining[::step][:needed]
        for e in extra:
            e["severity"] = "supplemental"
        sampled.extend(extra)

    print(f"[✓] Sampled {len(sampled)} error utterances (strategy: proportional stratified)")

    return sampled


def classify_error(reference: str, hypothesis: str) -> list:
    """
    Classify an error into one or more taxonomy categories.

    Compares reference and hypothesis word-by-word and character-by-character
    to determine the error type.

    Args:
        reference: Ground truth text
        hypothesis: Model output text

    Returns:
        List of (category_key, evidence) tuples
    """
    categories = []
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Check word boundary errors (different word counts)
    if len(ref_words) != len(hyp_words):
        diff = abs(len(ref_words) - len(hyp_words))
        if len(ref_words) > len(hyp_words):
            categories.append(("insertion_deletion", f"Deletion: {diff} word(s) dropped"))
        else:
            categories.append(("insertion_deletion", f"Insertion: {diff} extra word(s)"))

    # Check for merged/split words
    ref_joined = "".join(ref_words)
    hyp_joined = "".join(hyp_words)
    if ref_joined == hyp_joined and ref_words != hyp_words:
        categories.append(("word_boundary_error", "Words merged or split differently"))

    # Check for English/code-switching issues
    for w in ref_words:
        if is_roman_word(w):
            categories.append(("code_switching_error", f"English word in reference: {w}"))
            break

    # Check character-level differences
    for rw, hw in zip(ref_words, hyp_words):
        if rw == hw:
            continue

        # Check vowel/matra differences
        for ref_pat, hyp_pat in ERROR_CATEGORIES["vowel_matra_error"]["patterns"]:
            if ref_pat in rw and rw.replace(ref_pat, hyp_pat) == hw:
                categories.append(("vowel_matra_error", f"'{rw}' → '{hw}'"))
                break

        # Check consonant confusion
        for ref_pat, hyp_pat in ERROR_CATEGORIES["consonant_confusion"]["patterns"]:
            if ref_pat in rw and rw.replace(ref_pat, hyp_pat) == hw:
                categories.append(("consonant_confusion", f"'{rw}' → '{hw}'"))
                break

        # Check for anusvara/chandrabindu/visarga
        anusvara_chars = ['ं', 'ँ', 'ः']
        for char in anusvara_chars:
            if (char in rw) != (char in hw):
                categories.append(("visarga_anusvara", f"'{rw}' → '{hw}' (nasal mark)"))
                break

    # Check for filler words
    fillers = ['हम्म', 'अ', 'आ', 'उम्म', 'एम']
    for f in fillers:
        if f in reference and f not in hypothesis:
            categories.append(("punctuation_filler", f"Filler '{f}' missed"))
        elif f not in reference and f in hypothesis:
            categories.append(("punctuation_filler", f"Filler '{f}' inserted"))

    # If no specific category found, mark as other
    if not categories:
        categories.append(("other", "Unclassified error"))

    return categories


def build_taxonomy(sampled_errors: list) -> dict:
    """
    Build an error taxonomy from sampled errors (Q1e).

    For each category, collect examples showing:
    - Reference transcript
    - Model output
    - Reasoning about the cause

    Args:
        sampled_errors: List of sampled error dicts

    Returns:
        Dict mapping category -> {count, examples}
    """
    taxonomy = defaultdict(lambda: {"count": 0, "examples": []})

    for error in sampled_errors:
        ref = error.get("reference_normalized", error.get("reference", ""))
        hyp = error.get("hypothesis_normalized", error.get("hypothesis", ""))

        classifications = classify_error(ref, hyp)

        for category, evidence in classifications:
            taxonomy[category]["count"] += 1

            # Keep 3-5 examples per category
            if len(taxonomy[category]["examples"]) < 5:
                taxonomy[category]["examples"].append({
                    "reference": error.get("reference", ref),
                    "hypothesis": error.get("hypothesis", hyp),
                    "wer": error.get("wer", 0),
                    "severity": error.get("severity", ""),
                    "evidence": evidence,
                    "reasoning": f"Category: {ERROR_CATEGORIES.get(category, {}).get('name', category)}. "
                                 f"Evidence: {evidence}.",
                })

    return dict(taxonomy)


def propose_fixes(taxonomy: dict) -> list:
    """
    Propose specific, actionable fixes for top 3 error types (Q1f).

    Args:
        taxonomy: Error taxonomy dict

    Returns:
        List of fix proposals
    """
    # Sort categories by frequency
    sorted_cats = sorted(taxonomy.items(), key=lambda x: x[1]["count"], reverse=True)
    top3 = sorted_cats[:3]

    fixes = []
    fix_proposals = {
        "vowel_matra_error": {
            "fix": "Post-processing substitution map for common matra confusions. "
                   "Build a confusion matrix from training data and apply rule-based corrections "
                   "for high-confidence substitutions (e.g., when context disambiguates ा/ो).",
            "actionable": True,
        },
        "consonant_confusion": {
            "fix": "Add data augmentation with acoustically similar consonant pairs. "
                   "Use a phoneme-aware language model to rescore n-best hypotheses.",
            "actionable": False,
        },
        "word_boundary_error": {
            "fix": "Post-processing word segmentation using a Hindi dictionary lookup. "
                   "When two words are merged, try splitting at all valid positions and "
                   "check if both halves are valid Hindi words.",
            "actionable": True,
        },
        "code_switching_error": {
            "fix": "Fine-tune with code-switching augmented data. Add English loan words "
                   "commonly used in Hindi (interview, computer, phone) to the training vocabulary.",
            "actionable": False,
        },
        "insertion_deletion": {
            "fix": "Post-processing confidence-based filtering. Remove low-confidence tokens "
                   "and add commonly missed filler words based on acoustic patterns.",
            "actionable": True,
        },
        "homophone_confusion": {
            "fix": "Context-aware post-processing using a Hindi language model to select "
                   "the most likely candidate among homophones.",
            "actionable": False,
        },
        "visarga_anusvara": {
            "fix": "Rule-based post-processing to add/remove anusvara (ं) based on "
                   "Hindi grammar rules (e.g., before प/ब/म consonants, nasal mark is expected).",
            "actionable": True,
        },
        "punctuation_filler": {
            "fix": "Rule-based filler word normalization. Remove common ASR-inserted fillers "
                   "and standardize punctuation.",
            "actionable": True,
        },
        "number_transcription": {
            "fix": "Post-processing number normalization (already implemented in cleanup.py). "
                   "Apply Hindi number word ↔ digit conversion rules.",
            "actionable": True,
        },
    }

    for cat_key, cat_data in top3:
        proposal = fix_proposals.get(cat_key, {
            "fix": "Investigate further with more data to determine root cause.",
            "actionable": False,
        })
        fixes.append({
            "rank": len(fixes) + 1,
            "category": ERROR_CATEGORIES.get(cat_key, {}).get("name", cat_key),
            "category_key": cat_key,
            "count": cat_data["count"],
            "proposed_fix": proposal["fix"],
            "actionable": proposal["actionable"],
        })

    return fixes


def implement_fix(sampled_errors: list, fixes: list) -> list:
    """
    Implement at least one proposed fix and show before/after results (Q1g).

    Implements: Post-processing text normalization
    - Common substitution fixes
    - Filler word cleanup
    - Anusvara correction

    Args:
        sampled_errors: Error samples to apply fix on
        fixes: Proposed fixes list

    Returns:
        List of before/after results
    """
    print("\n[→] Implementing fix: Post-processing text normalization...")

    # ===== Common substitution rules =====
    substitution_rules = [
        # Anusvara corrections (before labial consonants)
        (r'([कखगघ])([^ं])([पबमभ])', r'\1ं\2\3'),  # Add anusvara before labials

        # Common matra fixes
        (r'हे', r'है'),    # Common Whisper error

        # Filler word cleanup
        (r'\bअ\s', ' '),    # Remove lone 'अ' filler
        (r'\bहम्म\b', ''),  # Remove hmm filler

        # Word boundary fixes
        (r'(\w)(\s+)(\w)', lambda m: m.group(0)),  # Keep normal spacing
    ]

    before_after = []

    for error in sampled_errors[:15]:  # Apply to targeted subset
        original_hyp = error.get("hypothesis", "")
        fixed_hyp = original_hyp

        # Apply substitution rules
        for pattern, replacement in substitution_rules:
            if callable(replacement):
                fixed_hyp = re.sub(pattern, replacement, fixed_hyp)
            else:
                fixed_hyp = re.sub(pattern, replacement, fixed_hyp)

        # Clean up extra spaces
        fixed_hyp = re.sub(r'\s+', ' ', fixed_hyp).strip()

        # Calculate improvement
        ref = error.get("reference_normalized", error.get("reference", ""))
        old_hyp_norm = normalize_hindi_text(original_hyp)
        new_hyp_norm = normalize_hindi_text(fixed_hyp)

        try:
            from jiwer import wer as compute_wer
            old_wer = compute_wer(ref, old_hyp_norm) if ref else 0.0
            new_wer = compute_wer(ref, new_hyp_norm) if ref else 0.0
        except Exception:
            old_wer = error.get("wer", 0)
            new_wer = old_wer

        before_after.append({
            "reference": error.get("reference", ref),
            "before": original_hyp,
            "after": fixed_hyp,
            "wer_before": round(old_wer, 4),
            "wer_after": round(new_wer, 4),
            "improved": new_wer < old_wer,
        })

    # Summary
    improved = sum(1 for ba in before_after if ba["improved"])
    print(f"[✓] Fix applied to {len(before_after)} samples.")
    print(f"    Improved: {improved}/{len(before_after)}")

    return before_after


def save_results(taxonomy: dict, fixes: list, before_after: list,
                 sampled_errors: list, output_dir: str):
    """
    Save all error analysis results.

    Args:
        taxonomy: Error taxonomy dict
        fixes: Proposed fixes
        before_after: Fix implementation results
        sampled_errors: Sampled error data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Error Taxonomy CSV =====
    tax_path = os.path.join(output_dir, "error_taxonomy.csv")
    with open(tax_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Count", "Description", "Example_Reference",
                         "Example_Hypothesis", "Evidence"])
        for cat_key, cat_data in sorted(taxonomy.items(), key=lambda x: x[1]["count"], reverse=True):
            cat_info = ERROR_CATEGORIES.get(cat_key, {})
            for ex in cat_data["examples"][:3]:
                writer.writerow([
                    cat_info.get("name", cat_key),
                    cat_data["count"],
                    cat_info.get("description", ""),
                    ex.get("reference", ""),
                    ex.get("hypothesis", ""),
                    ex.get("evidence", ""),
                ])
    print(f"[✓] Error taxonomy saved to: {tax_path}")

    # ===== Error Examples JSON (all 25+ samples) =====
    examples_path = os.path.join(output_dir, "error_examples.json")
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_strategy": "Proportional stratified sampling by WER severity "
                                 "(low: 0-0.3, medium: 0.3-0.6, high: 0.6+). "
                                 "Every Nth error from each stratum.",
            "total_sampled": len(sampled_errors),
            "samples": sampled_errors,
            "taxonomy": {k: {"count": v["count"], "examples": v["examples"]}
                         for k, v in taxonomy.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Error examples saved to: {examples_path}")

    # ===== Proposed Fixes =====
    fixes_path = os.path.join(output_dir, "proposed_fixes.json")
    with open(fixes_path, "w", encoding="utf-8") as f:
        json.dump(fixes, f, ensure_ascii=False, indent=2)
    print(f"[✓] Proposed fixes saved to: {fixes_path}")

    # ===== Before/After Results =====
    ba_path = os.path.join(output_dir, "fix_before_after.csv")
    with open(ba_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Reference", "Before_Fix", "After_Fix",
                         "WER_Before", "WER_After", "Improved"])
        for ba in before_after:
            writer.writerow([
                ba["reference"], ba["before"], ba["after"],
                ba["wer_before"], ba["wer_after"], ba["improved"],
            ])
    print(f"[✓] Before/after results saved to: {ba_path}")


def main():
    """Main error analysis pipeline."""
    parser = argparse.ArgumentParser(description="Error analysis for Hindi ASR")
    parser.add_argument("--predictions-path", type=str, default=PREDICTIONS_PATH)
    parser.add_argument("--min-samples", type=int, default=25)
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 4: ERROR ANALYSIS & TAXONOMY")
    print("=" * 60)

    # Check if predictions exist
    if not os.path.exists(args.predictions_path):
        print(f"[✗] Predictions file not found: {args.predictions_path}")
        print("[i] Run 'python src/evaluate.py' first to generate predictions.")
        sys.exit(1)

    # Step 1: Load predictions
    predictions, model_name = load_predictions(args.predictions_path)

    # Step 2: Sample errors (Q1d)
    sampled = sample_errors(predictions, args.min_samples)

    if not sampled:
        print("[✗] No errors to analyze.")
        sys.exit(1)

    # Step 3: Build taxonomy (Q1e)
    taxonomy = build_taxonomy(sampled)

    print("\n[→] Error Taxonomy Summary:")
    for cat_key, cat_data in sorted(taxonomy.items(), key=lambda x: x[1]["count"], reverse=True):
        cat_name = ERROR_CATEGORIES.get(cat_key, {}).get("name", cat_key)
        print(f"    {cat_name}: {cat_data['count']} occurrences")

    # Step 4: Propose fixes (Q1f)
    fixes = propose_fixes(taxonomy)

    print("\n[→] Top 3 Fix Proposals:")
    for fix in fixes:
        print(f"    #{fix['rank']} {fix['category']} ({fix['count']} errors)")
        print(f"        Fix: {fix['proposed_fix'][:80]}...")

    # Step 5: Implement fix (Q1g)
    before_after = implement_fix(sampled, fixes)

    # Step 6: Save everything
    save_results(taxonomy, fixes, before_after, sampled, OUTPUT_DIR)

    print(f"\n[✓] Error analysis complete!")
    print(f"    Next step: python src/cleanup.py")


if __name__ == "__main__":
    main()
