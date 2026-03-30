"""
spelling.py - Hindi Spelling Classification (Question 3)
=======================================================
Classifies ~1,75,000 unique Hindi words as correctly or incorrectly spelled.

Approach:
1. Dictionary-based: Check against known Hindi word lists
2. Frequency analysis: Very common words are likely correct
3. Pattern analysis: Check valid Devanagari character sequences
4. Morphological rules: Validate Hindi word structure patterns

Outputs:
- Correct/Incorrect classification for each word
- Confidence score (high/medium/low) with reasoning
- Analysis of low-confidence words

Usage:
    python src/spelling.py
    python src/spelling.py --words-file "Unique Words Data - Sheet1.csv"
"""

import os
import sys
import csv
import json
import re
import argparse
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import is_devanagari_word, is_devanagari

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_WORDS_FILE = os.path.join(PROJECT_ROOT, "Unique Words Data - Sheet1.csv")


# ===== Hindi Spelling Validation Rules =====

# Valid Devanagari consonants (व्यंजन)
CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")

# Valid Devanagari vowels (स्वर)
VOWELS = set("अआइईउऊऋएऐओऔ")

# Valid Devanagari vowel signs (मात्राएँ)
MATRAS = set("ािीुूृेैोौंँःॅॉ")

# Virama / Halant
HALANT = "्"

# Nukta (for borrowed sounds)
NUKTA = "़"

# Valid combinations that can follow halant
VALID_CONJUNCTS = {
    "क्ष", "त्र", "ज्ञ", "श्र", "द्व", "द्ध", "न्न", "ल्ल",
    "क्क", "प्प", "त्त", "द्द", "ब्ब", "म्म", "स्स",
    "स्त", "स्थ", "स्न", "स्प", "स्म", "स्व",
    "न्द", "न्त", "न्ध", "म्ब", "म्प",
    "प्र", "क्र", "ग्र", "त्र", "द्र", "भ्र",
    "ष्ट", "ष्ठ", "द्य", "ध्य", "न्य", "द्भ",
}

# Common valid Hindi suffixes
VALID_SUFFIXES = [
    "ों", "ें", "ाँ", "ाएँ", "ाओं", "ियों", "ाना", "ाने",
    "ता", "ती", "ते", "ना", "ने", "नी",
    "कर", "कार", "वाला", "वाली", "वाले",
    "पन", "पना", "दार", "हार", "गार",
]

# Common invalid patterns (likely misspellings)
INVALID_PATTERNS = [
    r'(.)\1{3,}',           # 4+ repeated characters → typo
    r'[ािीुूृेैोौ]{3,}',   # 3+ consecutive matras → invalid
    r'्{2,}',               # Multiple halants
    r'^[ािीुूृेैोौ]',      # Starting with a matra (invalid)
    r'[ंँ]{2,}',            # Multiple anusvara/chandrabindu
]


def load_words(words_file: str) -> list:
    """
    Load the unique words list from CSV.

    Args:
        words_file: Path to the CSV with one column 'word'

    Returns:
        List of unique words
    """
    words = []
    with open(words_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "").strip()
            if word:
                words.append(word)

    print(f"[✓] Loaded {len(words)} unique words.")
    return words


def check_devanagari_validity(word: str) -> tuple:
    """
    Check if a word follows valid Devanagari character sequences.

    Args:
        word: Word to validate

    Returns:
        (is_valid, issues) tuple
    """
    issues = []

    # Check for invalid patterns
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, word):
            issues.append(f"Invalid pattern: {pattern}")

    # Check character-level validity
    chars = list(word)
    for i, char in enumerate(chars):
        # Check if char is valid Devanagari
        if not (is_devanagari(char) or char in MATRAS or char == HALANT
                or char == NUKTA or char.isdigit() or char in '.-'):
            if not char.isascii():  # Ignore ASCII chars (handled elsewhere)
                issues.append(f"Non-Devanagari char: '{char}' (U+{ord(char):04X})")

    # Check for matra without preceding consonant (at start)
    if word and word[0] in MATRAS:
        issues.append("Starts with a matra (vowel sign)")

    return len(issues) == 0, issues


def is_english_in_devanagari(word: str) -> bool:
    """
    Check if a word is an English word transliterated to Devanagari.

    Per assignment guidelines: English words transcribed in Devanagari
    are considered CORRECT spelling.

    Args:
        word: Devanagari word to check

    Returns:
        True if likely an English word in Devanagari
    """
    # Import the loanword dictionary
    from src.english_detect import ENGLISH_LOANWORDS_DEVANAGARI

    # Direct lookup
    if word in ENGLISH_LOANWORDS_DEVANAGARI:
        return True

    # Check common English suffixes in Devanagari
    english_suffixes = ["शन", "मेंट", "नेस", "िंग", "टर", "ली", "फुल", "लेस"]
    for suffix in english_suffixes:
        if word.endswith(suffix) and len(word) > 4:
            return True

    return False


def classify_word(word: str) -> dict:
    """
    Classify a single word as correct or incorrect spelling.

    Uses multiple signals:
    1. Devanagari structural validity
    2. English loanword check (Devanagari English = correct)
    3. Length and character analysis
    4. Common pattern matching

    Args:
        word: Word to classify

    Returns:
        Dict with: classification, confidence, reason
    """
    # ===== Pre-checks =====

    # Empty or whitespace
    if not word or not word.strip():
        return {
            "word": word,
            "classification": "incorrect",
            "confidence": "high",
            "reason": "Empty or whitespace-only word",
        }

    # Pure digits or punctuation
    if word.isdigit() or all(c in '.,;:!?-' for c in word):
        return {
            "word": word,
            "classification": "correct",
            "confidence": "high",
            "reason": "Number or punctuation",
        }

    # Single character (valid Hindi characters are valid words)
    if len(word) == 1:
        if is_devanagari(word):
            return {
                "word": word,
                "classification": "correct",
                "confidence": "medium",
                "reason": "Single Devanagari character (could be abbreviation or particle)",
            }
        elif word.isascii() and word.isalpha():
            return {
                "word": word,
                "classification": "correct",
                "confidence": "medium",
                "reason": "Single Latin character (abbreviation)",
            }

    # ===== Check if it's Roman script (English word) =====
    if all(c.isascii() for c in word if c.isalpha()):
        clean = re.sub(r'[^a-zA-Z]', '', word)
        if clean:
            return {
                "word": word,
                "classification": "correct",
                "confidence": "medium",
                "reason": "Roman script word (English in conversation - per guidelines, Devanagari form is expected)",
            }

    # ===== Check if non-Devanagari script (Urdu, etc.) =====
    if not is_devanagari_word(word):
        devanagari_count = sum(1 for c in word if is_devanagari(c))
        if devanagari_count == 0:
            return {
                "word": word,
                "classification": "incorrect",
                "confidence": "high",
                "reason": "Non-Devanagari script (potentially Urdu/Arabic script in Devanagari context)",
            }

    # ===== English loanword in Devanagari (CORRECT per guidelines) =====
    try:
        if is_english_in_devanagari(word):
            return {
                "word": word,
                "classification": "correct",
                "confidence": "high",
                "reason": "English word in Devanagari (correct per transcription guidelines)",
            }
    except ImportError:
        pass  # Skip if english_detect not available

    # ===== Structural validity check =====
    is_valid, issues = check_devanagari_validity(word)

    if not is_valid:
        return {
            "word": word,
            "classification": "incorrect",
            "confidence": "high",
            "reason": f"Invalid Devanagari structure: {'; '.join(issues)}",
        }

    # ===== Common Hindi words (high frequency = likely correct) =====
    # Very common Hindi particles, postpositions, conjunctions
    COMMON_WORDS = {
        "है", "हैं", "था", "थी", "थे", "हो", "हुआ", "हुई",
        "का", "की", "के", "को", "से", "में", "पर", "ने",
        "और", "या", "कि", "जो", "तो", "भी", "ही", "सा",
        "यह", "वह", "ये", "वो", "मैं", "हम", "तुम", "आप",
        "जी", "नहीं", "हाँ", "हां", "ना", "मत", "बस",
        "कर", "करना", "करें", "किया", "होना", "जाना", "आना",
        "अच्छा", "बहुत", "कुछ", "सब", "कैसे", "क्या", "कब",
        "अब", "तब", "जब", "पहले", "बाद", "साथ", "लिए",
        "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ",
        "वाला", "वाली", "वाले", "कोई", "किसी", "अपना", "अपनी",
    }

    if word in COMMON_WORDS:
        return {
            "word": word,
            "classification": "correct",
            "confidence": "high",
            "reason": "Common Hindi word (high-frequency)",
        }

    # ===== Check word length and structure =====

    # Very long words without halant/conjunct (possible concatenation error)
    if len(word) > 20:
        halant_count = word.count(HALANT)
        if halant_count == 0:
            return {
                "word": word,
                "classification": "incorrect",
                "confidence": "medium",
                "reason": "Unusually long word without conjuncts (possible word concatenation)",
            }

    # ===== Valid suffix check =====
    has_valid_suffix = any(word.endswith(suffix) for suffix in VALID_SUFFIXES)

    # ===== Default: assume correct with medium confidence =====
    # For words that pass all checks but aren't in our dictionary,
    # we give medium confidence as correct
    return {
        "word": word,
        "classification": "correct",
        "confidence": "medium" if has_valid_suffix else "low",
        "reason": "Passes structural checks" + (
            " with valid Hindi suffix" if has_valid_suffix
            else "; not in reference dictionary (low confidence)"
        ),
    }


def classify_all_words(words: list) -> list:
    """
    Classify all words in the list.

    Args:
        words: List of unique words

    Returns:
        List of classification dicts
    """
    print(f"\n[→] Classifying {len(words)} words...")

    results = []
    for i, word in enumerate(words):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i:,}/{len(words):,}...")

        result = classify_word(word)
        results.append(result)

    # Summary
    correct = sum(1 for r in results if r["classification"] == "correct")
    incorrect = sum(1 for r in results if r["classification"] == "incorrect")
    high_conf = sum(1 for r in results if r["confidence"] == "high")
    med_conf = sum(1 for r in results if r["confidence"] == "medium")
    low_conf = sum(1 for r in results if r["confidence"] == "low")

    print(f"\n[✓] Classification complete!")
    print(f"    Correct: {correct:,} ({correct/len(results)*100:.1f}%)")
    print(f"    Incorrect: {incorrect:,} ({incorrect/len(results)*100:.1f}%)")
    print(f"    Confidence: High={high_conf:,}, Medium={med_conf:,}, Low={low_conf:,}")

    return results


def review_low_confidence(results: list, n: int = 50) -> list:
    """
    Review low-confidence classifications (Q3c).

    Examines 40-50 words from the low confidence bucket
    and provides detailed analysis.

    Args:
        results: All classification results
        n: Number of low-confidence words to review

    Returns:
        List of review dicts
    """
    low_conf = [r for r in results if r["confidence"] == "low"]

    if not low_conf:
        print("[i] No low-confidence words found.")
        return []

    # Take evenly spaced samples
    step = max(1, len(low_conf) // n)
    sample = low_conf[::step][:n]

    print(f"\n[→] Reviewing {len(sample)} low-confidence words...")

    reviews = []
    for r in sample:
        word = r["word"]

        # Additional analysis for review
        analysis = {
            "word": word,
            "original_classification": r["classification"],
            "original_reason": r["reason"],
            "char_count": len(word),
            "has_halant": HALANT in word,
            "has_nukta": NUKTA in word,
            "has_anusvara": "ं" in word or "ँ" in word,
            "is_devanagari": is_devanagari_word(word),
        }

        reviews.append(analysis)

    return reviews


def identify_unreliable_categories(results: list) -> list:
    """
    Identify word categories where the system is unreliable (Q3d).

    Args:
        results: All classification results

    Returns:
        List of unreliable category descriptions
    """
    unreliable = []

    # Category 1: Words with nukta (borrowed sounds)
    nukta_words = [r for r in results if NUKTA in r["word"]]
    if nukta_words:
        low_conf_nukta = sum(1 for r in nukta_words if r["confidence"] == "low")
        unreliable.append({
            "category": "Words with nukta (़)",
            "total": len(nukta_words),
            "low_confidence": low_conf_nukta,
            "explanation": "Words containing nukta (़) represent borrowed sounds (ज़, फ़, etc.). "
                           "These are common in Hindi-Urdu vocabulary but our system cannot "
                           "reliably determine if the nukta placement is correct without an "
                           "extensive Urdu-Hindi dictionary. Examples: ज़रूर, फ़िल्म, क़रीब.",
        })

    # Category 2: Proper nouns / named entities
    # These often have unusual character patterns
    long_uncommon = [r for r in results
                     if r["confidence"] == "low"
                     and len(r["word"]) > 5
                     and not any(r["word"].endswith(s) for s in VALID_SUFFIXES)]
    if long_uncommon:
        unreliable.append({
            "category": "Proper nouns and named entities",
            "total": len(long_uncommon),
            "low_confidence": len(long_uncommon),
            "explanation": "Proper nouns (names of people, places, brands) often have "
                           "spelling patterns that don't match standard Hindi vocabulary. "
                           "Without a named entity dictionary, our system cannot distinguish "
                           "between a misspelled common word and a correctly spelled proper noun.",
        })

    return unreliable


def save_results(results: list, reviews: list, unreliable: list, output_dir: str):
    """
    Save spelling classification results.

    Args:
        results: Classification results for all words
        reviews: Low-confidence review results
        unreliable: Unreliable category analysis
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Main Spelling Results CSV (for Google Sheet) =====
    csv_path = os.path.join(output_dir, "spelling_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Classification", "Confidence", "Reason"])
        for r in results:
            writer.writerow([
                r["word"],
                "correct spelling" if r["classification"] == "correct" else "incorrect spelling",
                r["confidence"],
                r["reason"],
            ])

    print(f"[✓] Spelling results saved to: {csv_path}")

    # ===== Summary Statistics =====
    correct = sum(1 for r in results if r["classification"] == "correct")
    incorrect = sum(1 for r in results if r["classification"] == "incorrect")

    summary = {
        "total_unique_words": len(results),
        "correctly_spelled": correct,
        "incorrectly_spelled": incorrect,
        "confidence_distribution": {
            "high": sum(1 for r in results if r["confidence"] == "high"),
            "medium": sum(1 for r in results if r["confidence"] == "medium"),
            "low": sum(1 for r in results if r["confidence"] == "low"),
        },
        "low_confidence_review": reviews,
        "unreliable_categories": unreliable,
    }

    json_path = os.path.join(output_dir, "spelling_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] Spelling analysis saved to: {json_path}")

    # ===== Print Summary =====
    print(f"\n{'=' * 50}")
    print(f"  SPELLING CLASSIFICATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total unique words: {len(results):,}")
    print(f"  Correctly spelled:  {correct:,} ({correct/len(results)*100:.1f}%)")
    print(f"  Incorrectly spelled: {incorrect:,} ({incorrect/len(results)*100:.1f}%)")
    print(f"{'=' * 50}")

    if reviews:
        print(f"\n  Low-confidence review ({len(reviews)} words):")
        for rev in reviews[:10]:
            print(f"    '{rev['word']}' - {rev['original_classification']} ({rev['original_reason'][:60]})")

    if unreliable:
        print(f"\n  Unreliable categories:")
        for u in unreliable:
            print(f"    • {u['category']}: {u['total']} words, {u['low_confidence']} low confidence")
            print(f"      {u['explanation'][:100]}...")


def main():
    """Main spelling classification pipeline."""
    parser = argparse.ArgumentParser(description="Hindi Spelling Classification")
    parser.add_argument("--words-file", type=str, default=DEFAULT_WORDS_FILE,
                        help="Path to the unique words CSV file")
    parser.add_argument("--review-count", type=int, default=50,
                        help="Number of low-confidence words to review")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 7: SPELLING CLASSIFICATION (Q3)")
    print("=" * 60)

    # Check words file
    if not os.path.exists(args.words_file):
        print(f"[✗] Words file not found: {args.words_file}")
        print("[i] Provide the 'Unique Words Data - Sheet1.csv' file.")
        sys.exit(1)

    # Step 1: Load words
    words = load_words(args.words_file)

    # Step 2: Classify all words
    results = classify_all_words(words)

    # Step 3: Review low-confidence words (Q3c)
    reviews = review_low_confidence(results, args.review_count)

    # Step 4: Identify unreliable categories (Q3d)
    unreliable = identify_unreliable_categories(results)

    # Step 5: Save results
    save_results(results, reviews, unreliable, OUTPUT_DIR)

    print(f"\n[✓] Spelling classification complete!")
    print(f"    Next step: python src/lattice.py")


if __name__ == "__main__":
    main()
