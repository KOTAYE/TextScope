# -*- coding: utf-8 -*-
"""
Language detection analyzer.
Identifies if the text is English, Polish, or Ukrainian based on character sets and word frequencies.
"""

import re
from typing import Dict, List, Union

# Common words dictionaries for language confidence scoring
COMMON_WORDS: Dict[str, List[str]] = {
    'pl': [
        'i', 'w', 'na', 'z', 'do', 'nie', 'to', 'jest', 'że', 'się',
        'co', 'jak', 'ale', 'za', 'od', 'po', 'tak', 'ten', 'o', 'czy',
        'tego', 'tej', 'tym', 'był', 'była', 'było', 'dla', 'ze', 'tylko',
        'bardzo', 'już', 'może', 'jeszcze', 'tu', 'jego', 'jej', 'ich',
        'kiedy', 'teraz', 'więc', 'gdzie', 'też', 'kto', 'nic', 'przez'
    ],
    'en': [
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
        'in', 'with', 'he', 'she', 'it', 'was', 'for', 'are', 'were', 'be',
        'has', 'have', 'had', 'not', 'this', 'that', 'from', 'they', 'we',
        'been', 'will', 'would', 'could', 'should', 'their', 'there', 'what',
        'when', 'where', 'how', 'who', 'did', 'does', 'do', 'if', 'my', 'your'
    ],
    'uk': [
        'і', 'в', 'на', 'з', 'до', 'не', 'це', 'є', 'що', 'як',
        'але', 'за', 'від', 'по', 'так', 'цей', 'ця', 'о', 'чи',
        'він', 'вона', 'воно', 'вони', 'ми', 'ви', 'та', 'або',
        'був', 'була', 'було', 'були', 'тут', 'де', 'коли', 'його',
        'її', 'їх', 'dla', 'тільки', 'дуже', 'ще', 'може', 'вже'
    ]
}

# Unique characters indicating specific languages
LANG_SPECIFIC_CHARS: Dict[str, str] = {
    'pl': 'ąęćłńóśźżĄĘĆŁŃÓŚŹŻ',
    'uk': 'іїєґІЇЄҐ'
}

# Display names map for detected languages
LANG_NAMES: Dict[str, str] = {
    'pl': 'Polski',
    'en': 'Angielski',
    'uk': 'Ukraiński',
    'unknown': 'Nieznany'
}

def detect_language(text: str) -> Dict[str, Union[str, int]]:
    """
    Detects if the given text is English, Polish, or Ukrainian.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        Dict[str, Union[str, int]]: Dictionary with 'code', 'name', and 'confidence' percentage.
    """
    if not text or len(text.strip()) < 5:
        return {'code': 'unknown', 'name': LANG_NAMES['unknown'], 'confidence': 0}

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = max(len(words), 1)

    char_scores = {}
    for lang, chars in LANG_SPECIFIC_CHARS.items():
        count = sum(1 for ch in text if ch in chars)
        char_scores[lang] = count

    for lang, count in char_scores.items():
        if count >= 1:
            return {
                'code': lang,
                'name': LANG_NAMES[lang],
                'confidence': min(95, 65 + count * 5)
            }

    word_scores = {}
    for lang, common in COMMON_WORDS.items():
        matches = sum(1 for w in words if w in common)
        word_scores[lang] = matches / total_words * 100

    cyrillic_count = sum(1 for ch in text if '\u0400' <= ch <= '\u04FF')
    latin_count = sum(1 for ch in text if ch.isalpha() and ch.isascii())

    if latin_count > cyrillic_count:
        word_scores.pop('uk', None)
    else:
        word_scores.pop('en', None)
        word_scores.pop('pl', None)

    if not word_scores:
        return {'code': 'unknown', 'name': LANG_NAMES['unknown'], 'confidence': 0}

    best_lang = max(word_scores, key=word_scores.get)
    best_score = word_scores[best_lang]

    if best_score < 3:
        if cyrillic_count > latin_count:
            best_lang = 'uk'
        else:
            best_lang = 'en'
        confidence = 30
    else:
        confidence = min(95, int(best_score * 3 + 30))

    return {
        'code': best_lang,
        'name': LANG_NAMES.get(best_lang, 'Nieznany'),
        'confidence': confidence
    }
