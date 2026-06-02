# -*- coding: utf-8 -*-
"""
Tests for the text statistics module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.stats import (
    count_syllables,
    split_into_sentences,
    get_text_stats
)

def test_count_syllables():
    assert count_syllables("cat") == 1
    assert count_syllables("apple") == 2
    assert count_syllables("książka") == 2
    assert count_syllables("книга") == 2

def test_split_into_sentences():
    text = "To jest pierwsze zdanie. A to jest drugie zdanie! Czy to jest trzecie?"
    s = split_into_sentences(text)
    assert len(s) == 3
    assert s[0] == "To jest pierwsze zdanie."
    assert s[1] == "A to jest drugie zdanie!"
    assert s[2] == "Czy to jest trzecie?"

def test_get_text_stats():
    text = "To jest przykładowy tekst, który służy do przetestowania statystyk tekstu."
    stats = get_text_stats(text)
    assert stats['word_count'] > 5
    assert stats['sentence_count'] >= 1
    assert stats['char_count'] == len(text)
    assert 0.0 <= stats['readability'] <= 100.0
    assert len(stats['top_words']) <= 5
