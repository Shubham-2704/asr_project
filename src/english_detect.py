"""
english_detect.py - English Word Detection in Hindi Text (Question 2b)
=====================================================================
Detects English words in Hindi transcripts (written in Devanagari)
and tags them with [EN]...[/EN] markers.

Strategy:
1. Dictionary-based: Common English loanwords transliterated to Devanagari
2. Script analysis: Detect Roman script tokens directly
3. Phonetic pattern matching: Identify Devanagari words that are English

Output format:
  Input:  "मेरा इंटरव्यू बहुत अच्छा गया"
  Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया"

Usage:
    python src/english_detect.py
    python src/english_detect.py --input outputs/raw_asr_pairs.json
"""

import os
import sys
import json
import csv
import re
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_utils import is_devanagari_word, is_roman_word

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ===== Configuration =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ===== English Loanwords in Devanagari =====
# Common English words frequently used in Hindi conversations
# These are transliterated to Devanagari as per common usage
ENGLISH_LOANWORDS_DEVANAGARI = {
    # Technology
    "कंप्यूटर", "कम्प्यूटर", "लैपटॉप", "मोबाइल", "फोन",
    "इंटरनेट", "वेबसाइट", "ऐप", "सॉफ्टवेयर", "हार्डवेयर",
    "डेस्कटॉप", "टैब", "टेबलेट", "वाईफाई", "ब्लूटूथ",
    "पेनड्राइव", "माउस", "कीबोर्ड", "स्क्रीन", "डिस्प्ले",
    "प्रोग्राम", "प्रोग्रामिंग", "कोड", "कोडिंग", "डेटा",
    "सर्वर", "डेटाबेस", "नेटवर्क", "ऑनलाइन", "ऑफलाइन",

    # Work / Career
    "इंटरव्यू", "जॉब", "ऑफिस", "कंपनी", "बॉस",
    "मीटिंग", "प्रोजेक्ट", "टीम", "मैनेजर", "रिपोर्ट",
    "सैलरी", "प्रमोशन", "रिज्यूमे", "स्किल", "ट्रेनिंग",
    "वर्कशॉप", "प्रेजेंटेशन", "डेडलाइन", "टास्क", "शिफ्ट",

    # Education
    "स्कूल", "कॉलेज", "यूनिवर्सिटी", "क्लास", "एग्जाम",
    "रिजल्ट", "टेस्ट", "स्टूडेंट", "टीचर", "प्रोफेसर",
    "डिग्री", "सर्टिफिकेट", "कोर्स", "सब्जेक्ट", "टॉपिक",
    "लाइब्रेरी", "लैब", "होमवर्क", "असाइनमेंट", "नोट्स",

    # Food
    "पिज्जा", "बर्गर", "चॉकलेट", "कॉफी", "टी",
    "बिस्किट", "केक", "आइसक्रीम", "सैंडविच", "ब्रेड",
    "सॉस", "चीज़", "क्रीम", "जूस", "कोल्डड्रिंक",

    # Common words
    "प्रॉब्लम", "सॉल्व", "टाइम", "टाइमिंग", "पार्किंग",
    "ट्रैफिक", "रोड", "फ्लोर", "लिफ्ट", "एलिवेटर",
    "शॉप", "शॉपिंग", "मार्केट", "बजट", "डिस्काउंट",
    "ड्राइवर", "कार", "बस", "ट्रेन", "फ्लाइट",
    "हॉस्पिटल", "डॉक्टर", "नर्स", "मेडिसिन", "टैबलेट",
    "पेशेंट", "ऑपरेशन", "रिपोर्ट", "चेकअप", "ट्रीटमेंट",

    # Lifestyle
    "फैशन", "ब्रांड", "स्टाइल", "डिज़ाइन", "कलर",
    "म्यूजिक", "मूवी", "फिल्म", "सीरीज", "शो",
    "गेम", "स्पोर्ट्स", "क्रिकेट", "फुटबॉल", "टेनिस",
    "ट्रेडिशनल", "मॉडर्न", "फैमिली", "फ्रेंड", "पार्टी",

    # Social media / modern
    "फेसबुक", "ट्विटर", "इंस्टाग्राम", "यूट्यूब", "गूगल",
    "वीडियो", "फोटो", "सेल्फी", "लाइक", "शेयर",
    "पोस्ट", "कमेंट", "फॉलो", "ट्रेंडिंग", "वायरल",

    # Business
    "बिज़नेस", "कस्टमर", "क्लाइंट", "प्रोडक्ट", "सर्विस",
    "मार्केटिंग", "इन्वेस्टमेंट", "प्रॉफिट", "लॉस", "फंड",

    # Emotions / Adjectives
    "हैप्पी", "सैड", "गुड", "बैड", "नाइस",
    "परफेक्ट", "इंपॉर्टेंट", "सिंपल", "ईज़ी", "हार्ड",
    "क्लियर", "कन्फ्यूज़", "टेंशन", "स्ट्रेस", "रिलैक्स",

    # Actions
    "स्टार्ट", "स्टॉप", "चेक", "ट्राई", "यूज़",
    "प्रैक्टिस", "डिस्कस", "एक्सप्लेन", "एक्सप्लोर", "एक्चुअली",
    "बेसिकली", "डायरेक्ट", "अप्लाई",

    # Mixed usage (from Q4 data)
    "पैशन", "डांसिंग", "इंफॉर्मेशन", "फीडबैक", "लाइक",
    "हार्ट", "प्योर", "बिहेव", "गिफ्टेड", "लैंड",
    "हीट", "वेव्स", "वेब्स", "इजी", "स्ट्रगल",
}


def detect_english_words(text: str) -> list:
    """
    Detect English words in a Hindi transcript.

    Uses three methods:
    1. Direct Roman script detection
    2. Devanagari loanword dictionary lookup
    3. Pattern matching for common English word endings in Devanagari

    Args:
        text: Hindi transcript text

    Returns:
        List of (word, start_idx, end_idx, method) tuples
    """
    detected = []
    words = text.split()

    for i, word in enumerate(words):
        # Clean word of punctuation for comparison
        clean_word = re.sub(r'[,\.\!\?\;\:\"\'\(\)\[\]]', '', word)

        if not clean_word:
            continue

        method = None

        # Method 1: Roman script detection
        if is_roman_word(clean_word):
            method = "roman_script"

        # Method 2: Dictionary lookup (Devanagari loanwords)
        elif clean_word in ENGLISH_LOANWORDS_DEVANAGARI:
            method = "dictionary"

        # Method 3: Check common English suffixes in Devanagari
        # -tion → -शन, -ing → -िंग, -ment → -मेंट, -ness → -नेस
        elif any(clean_word.endswith(suffix) for suffix in
                 ["शन", "िंग", "मेंट", "नेस", "टर", "ली", "फुल"]):
            # Additional check: make sure it's a known pattern
            if len(clean_word) > 3:  # Avoid false positives on short words
                method = "suffix_pattern"

        if method:
            detected.append({
                "word": clean_word,
                "original": word,
                "position": i,
                "method": method,
            })

    return detected


def tag_english_words(text: str) -> str:
    """
    Tag English words in text with [EN]...[/EN] markers.

    Example:
        Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"
        Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मिल गई"

    Args:
        text: Hindi transcript text

    Returns:
        Tagged text with [EN]...[/EN] markers
    """
    detected = detect_english_words(text)

    if not detected:
        return text

    # Build set of positions to tag
    tag_positions = {d["position"] for d in detected}

    words = text.split()
    tagged_words = []

    for i, word in enumerate(words):
        if i in tag_positions:
            tagged_words.append(f"[EN]{word}[/EN]")
        else:
            tagged_words.append(word)

    return " ".join(tagged_words)


def process_transcripts(pairs: list) -> list:
    """
    Process a list of transcript pairs to detect and tag English words.

    Args:
        pairs: List of dicts with 'raw_asr' or 'text' keys

    Returns:
        List of processed results
    """
    results = []

    for pair in pairs:
        text = pair.get("raw_asr", pair.get("text", pair.get("human_reference", "")))

        if not text:
            continue

        detected = detect_english_words(text)
        tagged = tag_english_words(text)

        results.append({
            "original": text,
            "tagged": tagged,
            "english_words": [d["word"] for d in detected],
            "detection_methods": [d["method"] for d in detected],
            "english_word_count": len(detected),
            "total_word_count": len(text.split()),
        })

    return results


def save_results(results: list, output_dir: str):
    """
    Save English detection results to CSV and JSON.

    Args:
        results: List of processed transcript results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== Tagged Transcripts CSV =====
    csv_path = os.path.join(output_dir, "english_tagged.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Original Transcript", "Tagged Transcript",
                         "English Words Found", "Count", "Detection Methods"])

        for r in results:
            writer.writerow([
                r["original"],
                r["tagged"],
                ", ".join(r["english_words"]) if r["english_words"] else "None",
                r["english_word_count"],
                ", ".join(set(r["detection_methods"])) if r["detection_methods"] else "N/A",
            ])

    print(f"[✓] Tagged transcripts saved to: {csv_path}")

    # ===== Full Results JSON =====
    json_path = os.path.join(output_dir, "english_detection_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_transcripts": len(results),
            "transcripts_with_english": sum(1 for r in results if r["english_word_count"] > 0),
            "total_english_words_found": sum(r["english_word_count"] for r in results),
            "results": results[:100],  # First 100 for file size
        }, f, ensure_ascii=False, indent=2)

    print(f"[✓] Full detection results saved to: {json_path}")

    # ===== Print Summary =====
    total = len(results)
    with_english = sum(1 for r in results if r["english_word_count"] > 0)
    all_english = []
    for r in results:
        all_english.extend(r["english_words"])

    print(f"\n{'=' * 50}")
    print(f"  ENGLISH WORD DETECTION SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total transcripts: {total}")
    print(f"  With English words: {with_english} ({with_english/total*100:.1f}%)" if total else "")
    print(f"  Total English words: {len(all_english)}")
    if all_english:
        from collections import Counter
        top10 = Counter(all_english).most_common(10)
        print(f"  Top 10 English words:")
        for word, count in top10:
            print(f"    {word}: {count}")
    print(f"{'=' * 50}")


def main():
    """Main English detection pipeline."""
    parser = argparse.ArgumentParser(description="English Word Detection in Hindi Transcripts")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input JSON with transcript pairs")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 6: ENGLISH WORD DETECTION (Q2b)")
    print("=" * 60)

    # Try to load existing ASR pairs, or use demo data
    pairs = []

    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"[✓] Loaded {len(pairs)} transcripts from {args.input}")
    else:
        # Check for raw ASR pairs from cleanup step
        pairs_path = os.path.join(OUTPUT_DIR, "raw_asr_pairs.json")
        if os.path.exists(pairs_path):
            with open(pairs_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            print(f"[✓] Loaded {len(pairs)} ASR pairs from cleanup step.")
        else:
            # Use demo data
            print("[→] Using demo transcripts for English detection...")
            pairs = [
                {"raw_asr": "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई"},
                {"raw_asr": "ये प्रॉब्लम सॉल्व नहीं हो रहा"},
                {"raw_asr": "मुझे म्यूजिक सुनना पसंद है गाना भी पसंद है एक्चुअली और डांसिंग तो पैशन है मेरा"},
                {"raw_asr": "हाँ नहीं तो पहले डेस्कटॉप था ना बाद में लैपटॉप आया"},
                {"raw_asr": "अच्छा चलिए अच्छा हुआ आपने ये सारी इंफॉर्मेशन मुझे बता दी"},
                {"raw_asr": "तो भारत तो एक खुद से ही गिफ्टेड लैंड है तो हमें इसको एक्सप्लोर करना चाहिए"},
                {"raw_asr": "जी फीडबैक मिलने पर सुधार करना"},
                {"raw_asr": "बहुत प्योर हार्ट रहता है सबका"},
                {"raw_asr": "हम उनके साथ कैसा बिहेव किए उस समय"},
                {"raw_asr": "लाइक रविवार को कॉलेज की भी छुट्टी रहती है"},
            ]

    # Process transcripts
    results = process_transcripts(pairs)

    # Save results
    save_results(results, OUTPUT_DIR)

    print(f"\n[✓] English word detection complete!")
    print(f"    Next step: python src/spelling.py")


if __name__ == "__main__":
    main()
