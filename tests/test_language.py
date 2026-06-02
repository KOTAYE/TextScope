# -*- coding: utf-8 -*-
"""
Tests for the language detection module.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.language import detect_language

def test_detect_language_polish():
    res = detect_language("To jest wspaniały i bardzo ciekawy polski tekst.")
    assert res['code'] == 'pl'
    assert res['name'] == 'Polski'
    assert res['confidence'] > 50

def test_detect_language_english():
    res = detect_language("This is a simple english text containing common words like the is at.")
    assert res['code'] == 'en'
    assert res['name'] == 'Angielski'
    assert res['confidence'] > 50

def test_detect_language_ukrainian():
    res = detect_language("Це гарний український текст з особливими літерами.")
    assert res['code'] == 'uk'
    assert res['name'] == 'Ukraiński'
    assert res['confidence'] > 50

def test_detect_language_unknown():
    res = detect_language("abc")
    assert res['code'] == 'unknown'
    assert res['confidence'] == 0
