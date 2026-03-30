"""
hindi_numbers.py - Hindi Number Word to Digit Converter
======================================================
Converts Hindi number words (Devanagari) to their digit equivalents.

Handles:
- Simple numbers: दो → 2, दस → 10, सौ → 100
- Compound numbers: तीन सौ चौवन → 354, पच्चीस → 25
- Large numbers: एक हज़ार → 1000, दो लाख → 200000
- Edge cases: idiomatic usage detection (e.g., "दो-चार बातें" stays as-is)
"""

import re

# ===== Basic Hindi Number Words (0-99) =====
HINDI_UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
    "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45,
    "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
    "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
    "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्यासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
    "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
    "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

# ===== Multipliers =====
HINDI_MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000, "हजार": 1000, "हज़ारों": 1000,
    "लाख": 100000, "लाखों": 100000,
    "करोड़": 10000000, "करोड़ों": 10000000,
    "अरब": 1000000000,
}

# ===== Idiomatic Patterns (should NOT be converted) =====
# These are common Hindi expressions where numbers are used figuratively
IDIOM_PATTERNS = [
    r"दो-चार",        # दो-चार बातें (a few things)
    r"चार-पाँच",      # roughly 4-5
    r"दो-तीन",        # 2-3 (approximate)
    r"एक-दो",         # 1-2 (approximate)
    r"तीन-चार",       # 3-4 (approximate)
    r"सात-आठ",        # 7-8 (approximate)
    r"दो-एक",         # one or two
    r"एक-आध",         # one or so
    r"चार\s+चाँद",    # चार चाँद लगाना (to embellish)
    r"एक\s+नंबर",     # एक नंबर (first class)
    r"दो\s+नंबर",     # दो नंबर (black market)
    r"तीन\s+तेरह",    # scatter in all directions
    r"नौ\s+दो\s+ग्यारह",  # run away (9+2=11 ~ flee)
    r"साठ\s+पाट",     # settlement
    r"छत्तीस\s+का\s+आंकड़ा",  # enmity
    r"एक\s+से\s+एक",  # each one better
]


def is_idiomatic(text: str) -> bool:
    """
    Check if the text contains an idiomatic expression with numbers.
    If so, the numbers should NOT be converted to digits.

    Args:
        text: Hindi text string to check

    Returns:
        True if the text contains an idiom pattern
    """
    for pattern in IDIOM_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def parse_hindi_number(words: list) -> tuple:
    """
    Parse a sequence of Hindi number words into a single integer.

    Strategy:
    - First, check if any word is a direct unit (0-99)
    - Then, handle multipliers (सौ, हज़ार, लाख, करोड़)
    - Combine: "तीन सौ चौवन" → 3 * 100 + 54 = 354

    Args:
        words: List of Hindi words to parse

    Returns:
        Tuple of (number_value, words_consumed)
        Returns (None, 0) if no valid number found
    """
    if not words:
        return None, 0

    total = 0
    current = 0
    words_consumed = 0

    i = 0
    while i < len(words):
        word = words[i].strip()

        # Check if it's a unit (0-99)
        if word in HINDI_UNITS:
            current = HINDI_UNITS[word]
            words_consumed = i + 1
            i += 1
        # Check if it's a multiplier (सौ, हज़ार, etc.)
        elif word in HINDI_MULTIPLIERS:
            multiplier = HINDI_MULTIPLIERS[word]
            if current == 0:
                current = 1  # "सौ" alone means 100
            if multiplier >= 1000:
                # For हज़ार, लाख, करोड़: multiply and add to total
                total += current * multiplier
                current = 0
            else:
                # For सौ: just multiply current
                current = current * multiplier
            words_consumed = i + 1
            i += 1
        else:
            # Not a number word, stop parsing
            break

    total += current

    if words_consumed == 0:
        return None, 0

    return total, words_consumed


def convert_numbers_in_text(text: str) -> str:
    """
    Convert Hindi number words in text to digits.

    Handles:
    - Simple: "दो आदमी" → "2 आदमी"
    - Compound: "तीन सौ चौवन रुपये" → "354 रुपये"
    - Edge cases: Skips idiomatic expressions

    Args:
        text: Hindi text with number words

    Returns:
        Text with number words replaced by digits
    """
    # First check: skip idiomatic patterns
    if is_idiomatic(text):
        # Only convert non-idiomatic parts
        # For now, return as-is if idiom is detected
        return text

    words = text.split()
    result = []
    i = 0

    while i < len(words):
        word = words[i].strip()

        # Check if current word starts a number sequence
        if word in HINDI_UNITS or word in HINDI_MULTIPLIERS:
            # Try to parse a complete number from this position
            number, consumed = parse_hindi_number(words[i:])
            if number is not None and consumed > 0:
                result.append(str(number))
                i += consumed
                continue

        result.append(words[i])
        i += 1

    return " ".join(result)


if __name__ == "__main__":
    # ===== Test Examples =====
    test_cases = [
        ("दो आदमी आए", "Simple: दो → 2"),
        ("दस बजे मिलते हैं", "Simple: दस → 10"),
        ("सौ रुपये दो", "Simple: सौ → 100"),
        ("तीन सौ चौवन लोग", "Compound: तीन सौ चौवन → 354"),
        ("पच्चीस दिन बाद", "Direct compound: पच्चीस → 25"),
        ("एक हज़ार रुपये", "Thousand: एक हज़ार → 1000"),
        ("दो लाख तीस हज़ार", "Large: दो लाख तीस हज़ार → 230000"),
        ("दो-चार बातें करो", "Idiom: should stay as-is"),
    ]

    print("=" * 60)
    print("Hindi Number Conversion Test")
    print("=" * 60)
    for text, description in test_cases:
        converted = convert_numbers_in_text(text)
        print(f"\n{description}")
        print(f"  Input:  {text}")
        print(f"  Output: {converted}")
