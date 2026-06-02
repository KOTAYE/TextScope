# -*- coding: utf-8 -*-
"""
Tests for the sentiment analysis module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.sentiment import (
    normalize_text_for_lexicon,
    analyze_multilingual_sentiment,
    analyze_sentiment,
    get_word_sentiments,
    analyze_sentences_sentiment
)

def test_normalize_text():
    assert normalize_text_for_lexicon("Ładny") == "ladny"
    assert normalize_text_for_lexicon("Zgniłe") == "zgnile"

def test_analyze_multilingual_sentiment_pl():
    # positive pl stem 'dobr'
    score, pos, neg = analyze_multilingual_sentiment("To jest dobre danie.", "pl")
    assert pos > 0
    assert neg == 0
    assert score == 1.0

    # negative pl stem 'kiepsk'
    score, pos, neg = analyze_multilingual_sentiment("Kiepski hotel.", "pl")
    assert pos == 0
    assert neg > 0
    assert score == -1.0

def test_analyze_multilingual_sentiment_uk():
    # positive uk stem 'чудов'
    score, pos, neg = analyze_multilingual_sentiment("Це чудове місце.", "uk")
    assert pos > 0
    assert neg == 0
    assert score == 1.0

def test_analyze_sentiment_categories():
    res_pos = analyze_sentiment("Wspaniały dzień! Bardzo polecam.")
    assert "pozytywny" in res_pos['category'].lower() or res_pos['combined_score'] > 0
    assert res_pos['positive_pct'] > 50

    res_neg = analyze_sentiment("Okropne jedzenie i beznadziejna obsługa.")
    assert "negatywny" in res_neg['category'].lower() or res_neg['combined_score'] < 0
    assert res_neg['negative_pct'] > 50

def test_get_word_sentiments():
    words = get_word_sentiments("Wspaniały i okropny", lang_code="pl")
    types = [w['type'] for w in words if w['type'] != 'space']
    assert 'positive' in types
    assert 'negative' in types

def test_analyze_sentences_sentiment():
    text = "Wspaniały dzień. Okropna pogoda."
    results = analyze_sentences_sentiment(text)
    assert len(results) >= 2
    assert results[0]['sentiment_score'] > 0
    assert results[1]['sentiment_score'] < 0
