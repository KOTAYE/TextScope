# -*- coding: utf-8 -*-
"""
Tests for the LLM semantic analysis module.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.llm_analysis import analyze_text_with_llm

def test_llm_without_api_key():
    # Force empty API key env
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        res = analyze_text_with_llm("Wspaniały produkt.")
        assert res is None

def test_llm_with_api_key_mocked():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": '{"sentiment_score": 0.9, "category": "Pozytywny", "subjectivity": 30.0, "explanation": "Świetna opinia.", "emotions": {"joy": 95.0, "anger": 0.0, "sadness": 0.0, "fear": 0.0, "surprise": 5.0, "disgust": 0.0}, "topics": {"obsługa": {"keywords": ["kelner"], "relevance": 80.0}}, "recommendations": ["Dobra robota."]}'
                }]
            }
        }]
    }

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key_123"}):
        with patch("requests.post", return_value=mock_response):
            res = analyze_text_with_llm("Wspaniały produkt.")
            assert res is not None
            assert res["sentiment_score"] == 0.9
            assert res["category"] == "Pozytywny"
            assert res["subjectivity"] == 30.0
            assert res["explanation"] == "Świetna opinia."
            assert res["emotions"]["joy"] == 95.0
            assert res["topics"]["obsługa"]["relevance"] == 80.0
            assert res["recommendations"] == ["Dobra robota."]
