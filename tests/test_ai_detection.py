# -*- coding: utf-8 -*-
"""
Tests for the AI text detection module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.ai_detection import (
    calculate_sentence_uniformity,
    calculate_vocabulary_richness,
    calculate_formality_score,
    calculate_repetition_patterns,
    calculate_punctuation_score,
    calculate_connector_density,
    detect_ai_text
)

def test_detect_ai_too_short():
    res = detect_ai_text("Krótki tekst.")
    assert res['error'] is True
    assert res['ai_probability'] == 0.0
    assert res['human_probability'] == 100.0

def test_detect_ai_normal_text():
    # Long enough text to analyze
    text = (
        "Furthermore, it should be noted that artificial intelligence has become a standard tool in modern times. "
        "Consequently, many students utilize these advanced methods to write their assignments. "
        "However, we must understand the implications of such automated academic work."
    )
    res = detect_ai_text(text)
    assert res['error'] is False
    assert 0.0 <= res['ai_probability'] <= 100.0
    assert res['ai_probability'] + res['human_probability'] == 100.0
    assert 'details' in res

def test_stylometric_functions():
    sentences = ["This is sentence one.", "This is sentence two.", "And this is the final sentence."]
    uniformity = calculate_sentence_uniformity(sentences)
    assert 0.0 <= uniformity <= 1.0

    text = "Furthermore, it should be noted that we need to analyze this however and moreover."
    formality = calculate_formality_score(text)
    assert 0.0 <= formality <= 1.0

    connectors = calculate_connector_density(text)
    assert 0.0 <= connectors <= 1.0
