# -*- coding: utf-8 -*-
"""
Text statistics analyzer.
Computes readability scores, word counts, unique words ratios, and lists top words.
"""

import re
from collections import Counter
from typing import Dict, List, Union

def count_syllables(word: str) -> int:
    """
    Counts the number of syllables in a given word for both Latin and Cyrillic character sets.
    
    Args:
        word (str): The word to inspect.
        
    Returns:
        int: Total syllable count (minimum of 1).
    """
    word = word.lower().strip()
    if not word:
        return 1
    count = 0
    vowels = 'aeiouyаеєиіїоуюя'
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    return max(count, 1)

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into independent sentences based on end punctuation.
    
    Args:
        text (str): The input text.
        
    Returns:
        List[str]: List of sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]

def get_text_stats(text: str) -> Dict[str, Union[int, float, str, List[Dict[str, Union[str, int]]]]]:
    """
    Compiles detailed structural statistics about the text.
    
    Args:
        text (str): The input text.
        
    Returns:
        Dict[str, Union[int, float, str, List[Dict[str, Union[str, int]]]]]: Text metrics including word count,
        sentence count, readability index, vocabulary richness, and frequent words.
    """
    words = re.findall(r'\b\w+\b', text)
    sentences = split_into_sentences(text)
    syllables = sum(count_syllables(w) for w in words)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    char_count = len(text)
    char_no_spaces = len(text.replace(' ', ''))

    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)
    avg_sentence_len = round(word_count / sentence_count, 1)

    unique_words = set(w.lower() for w in words)
    ttr = round(len(unique_words) / max(word_count, 1) * 100, 1)

    if word_count > 0 and sentence_count > 0:
        asl = word_count / sentence_count
        asw = syllables / word_count
        readability = 206.835 - (1.015 * asl) - (84.6 * asw)
        readability = max(0, min(100, round(readability, 1)))
    else:
        readability = 0.0

    if readability >= 80:
        difficulty = "Łatwy"
        diff_color = "#4ade80"
    elif readability >= 60:
        difficulty = "Średni"
        diff_color = "#facc15"
    elif readability >= 40:
        difficulty = "Trudny"
        diff_color = "#fb923c"
    else:
        difficulty = "Bardzo trudny"
        diff_color = "#f87171"

    word_freq = Counter(w.lower() for w in words if len(w) > 3)
    top_words = [{'word': w, 'count': c} for w, c in word_freq.most_common(5)]

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'char_count': char_count,
        'char_no_spaces': char_no_spaces,
        'avg_word_len': avg_word_len,
        'avg_sentence_len': avg_sentence_len,
        'unique_words': len(unique_words),
        'vocabulary_richness': ttr,
        'readability': readability,
        'difficulty': difficulty,
        'difficulty_color': diff_color,
        'syllable_count': syllables,
        'top_words': top_words
    }
