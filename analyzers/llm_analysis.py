# -*- coding: utf-8 -*-
"""
LLM Semantic Analysis Module.
Connects to Gemini API to perform deep semantic, contextual, and sarcasm analysis of text.
"""

import json
import os
import requests
from typing import Dict, Any, Optional

def analyze_text_with_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    Sends the text to Google's Gemini API for advanced semantic and sarcasm analysis.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary from LLM, or None if disabled/failed.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None  # Gracefully skip if no API key is configured
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    prompt = f"""Przeanalizuj poniższy tekst pod kątem nastroju (sentymentu), subiektywności, emocji, tematów oraz wygeneruj 2 spersonalizowane rekomendacje w języku polskim.
Tekst do analizy: "{text}"

Wydobądź informacje dla kategorii tematycznych: "obsługa", "cena", "jakość", "lokalizacja", "atmosfera", "inne". Określ ich trafność (relevance od 0.0 do 100.0) oraz pasujące słowa kluczowe.

Zwróć odpowiedź WYŁĄCZNIE w formacie JSON o poniższej strukturze (bez żadnego formatowania markdown ```json, tylko czysty ciąg JSON):
{{
  "sentiment_score": 0.8,
  "category": "Pozytywny",
  "subjectivity": 45.0,
  "explanation": "szczegółowa, błyskotliwa i precyzyjna analiza semantyczna i kontekstowa w języku polskim (2-3 zdania) wyjaśniająca ukryty sens, odczucia i czy występuje ironia/sarkazm.",
  "emotions": {{
    "joy": 85.0,
    "anger": 0.0,
    "sadness": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "disgust": 0.0
  }},
  "topics": {{
    "obsługa": {{ "keywords": [], "relevance": 0.0 }},
    "cena": {{ "keywords": [], "relevance": 0.0 }},
    "jakość": {{ "keywords": [], "relevance": 0.0 }},
    "lokalizacja": {{ "keywords": [], "relevance": 0.0 }},
    "atmosfera": {{ "keywords": [], "relevance": 0.0 }},
    "inne": {{ "keywords": [], "relevance": 0.0 }}
  }},
  "recommendations": [
    "Pierwsza dynamiczna rekomendacja po polsku...",
    "Druga dynamiczna rekomendacja po polsku..."
  ]
}}"""

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=12)
        if response.status_code == 200:
            res_data = response.json()
            # Extract generated text from Gemini structure
            candidates = res_data.get("candidates", [])
            if candidates:
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text_content:
                    parsed = json.loads(text_content)
                    return {
                        "sentiment_score": float(parsed.get("sentiment_score", 0.0)),
                        "category": str(parsed.get("category", "Neutralny")),
                        "subjectivity": float(parsed.get("subjectivity", 0.0)),
                        "explanation": str(parsed.get("explanation", "")),
                        "emotions": parsed.get("emotions", None),
                        "topics": parsed.get("topics", None),
                        "recommendations": parsed.get("recommendations", None)
                    }
    except Exception:
        pass  # Fail silently to guarantee application stability
        
    return None
