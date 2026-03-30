"""
text_utils.py - Common Text Processing Utilities
================================================
Shared text normalization and Devanagari script utilities
used across multiple modules in the pipeline.
"""

import re
import unicodedata


def normalize_hindi_text(text: str) -> str:
    """
    Normalize Hindi text for ASR evaluation.

    Steps:
    1. Convert to NFC Unicode normalization
    2. Remove extra whitespace
    3. Remove common punctuation (keep Devanagari danda)
    4. Lowercase any English characters
    5. Strip leading/trailing whitespace

    Args:
        text: Raw Hindi text

    Returns:
        Normalized text ready for WER comparison
    """
    if not text:
        return ""

    # Unicode NFC normalization (important for Devanagari)
    text = unicodedata.normalize("NFC", text)

    # Remove common punctuation but keep Devanagari danda (।) and visarga
    text = re.sub(r'[,\.\!\?\;\:\"\'\(\)\[\]\{\}\-\–\—\…]', ' ', text)

    # Lowercase English characters (Whisper sometimes outputs mixed case)
    text = text.lower()

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)

    # Strip
    text = text.strip()

    return text


def is_devanagari(char: str) -> bool:
    """
    Check if a character is in the Devanagari Unicode block.

    Devanagari range: U+0900 to U+097F
    Devanagari Extended: U+A8E0 to U+A8FF

    Args:
        char: Single character to check

    Returns:
        True if the character is Devanagari
    """
    code = ord(char)
    return (0x0900 <= code <= 0x097F) or (0xA8E0 <= code <= 0xA8FF)


def is_devanagari_word(word: str) -> bool:
    """
    Check if a word is primarily written in Devanagari script.

    Args:
        word: Word to check

    Returns:
        True if more than 50% of alphabetic chars are Devanagari
    """
    if not word:
        return False

    # Count Devanagari vs non-Devanagari alphabetic characters
    devanagari_count = sum(1 for c in word if is_devanagari(c))
    alpha_count = sum(1 for c in word if c.isalpha())

    if alpha_count == 0:
        return False

    return devanagari_count / alpha_count > 0.5


def is_roman_word(word: str) -> bool:
    """
    Check if a word is written in Roman/Latin script.

    Args:
        word: Word to check

    Returns:
        True if the word contains only ASCII letters
    """
    cleaned = re.sub(r'[^a-zA-Z]', '', word)
    return len(cleaned) > 0 and cleaned.isascii()


def split_into_sentences(text: str) -> list:
    """
    Split Hindi text into sentences.
    Uses Devanagari danda (।), period, and question mark as delimiters.

    Args:
        text: Hindi text

    Returns:
        List of sentence strings
    """
    # Split on Devanagari danda, period, question mark, exclamation
    sentences = re.split(r'[।\.\?\!]+', text)
    # Clean and filter empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def word_tokenize(text: str) -> list:
    """
    Simple word tokenizer for Hindi text.
    Splits on whitespace and removes empty tokens.

    Args:
        text: Text to tokenize

    Returns:
        List of word tokens
    """
    return [w for w in text.split() if w.strip()]


def calculate_char_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    Uses simple edit distance.

    Args:
        reference: Ground truth text
        hypothesis: Model output text

    Returns:
        CER as a float between 0 and 1
    """
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Dynamic programming edit distance
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


if __name__ == "__main__":
    # Quick self-test
    print("=== Text Utils Test ===")

    test = "  हैलो ,  दोस्तों!  कैसे हो?  "
    print(f"Original:   '{test}'")
    print(f"Normalized: '{normalize_hindi_text(test)}'")

    print(f"\n'क' is Devanagari: {is_devanagari('क')}")
    print(f"'A' is Devanagari: {is_devanagari('A')}")

    print(f"\n'हैलो' is Devanagari word: {is_devanagari_word('हैलो')}")
    print(f"'hello' is Roman word: {is_roman_word('hello')}")

    ref = "मेरा नाम राम है"
    hyp = "मेरा नम राम है"
    print(f"\nCER('{ref}', '{hyp}'): {calculate_char_error_rate(ref, hyp):.4f}")
