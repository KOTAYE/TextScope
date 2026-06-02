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

def explain_ai_detection_with_llm(text: str, ai_details: dict, ai_probability: float) -> Optional[str]:
    """
    Calls Google Gemini 2.0 Flash API to write a 2-3 sentence Polish explanation
    of the calculated AI probability and stylometric metrics.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Format metrics for the prompt
    metrics_str = "\n".join([f"- {k}: {v}%" for k, v in ai_details.items()])
    
    prompt = f"""Przeanalizuj poniższy tekst pod kątem autorstwa człowieka lub sztucznej inteligencji.
Oto wyniki analizy stylistycznej (heurystycznej):
- Prawdopodobieństwo AI: {ai_probability}%
Szczegóły metryk:
{metrics_str}

Tekst do analizy:
"{text}"

Napisz 2-3 zwięzłe zdania wyjaśnienia po polsku (przystępnym i profesjonalnym językiem), DLACZEGO ten tekst uzyskał taki wynik. Odnieś się krótko do powyższych metryk (np. jednorodność zdań, słownictwo, interpunkcja) oraz ogólnego charakteru tekstu (np. naturalność, powtarzalność, monotonność).
Zwróć TYLKO czysty tekst wyjaśnienia po polsku, bez żadnego formatowania markdown."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            candidates = res_data.get("candidates", [])
            if candidates:
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text_content:
                    return text_content
    except Exception:
        pass
    return None

def detect_fake_review_fallback(text: str) -> Dict[str, Any]:
    words = text.split()
    word_count = len(words)
    
    if word_count < 6:
        return {
            "verdict": "suspicious",
            "verdict_label": "Podejrzana",
            "confidence": 60.0,
            "reasoning": "Tekst jest zbyt krótki (mniej niż 6 słów), aby wiarygodnie ocenić jego autentyczność jako prawdziwą opinię klienta."
        }
    
    text_lower = text.lower()
    suspicious_patterns = [
        "super!!!", "najlepszy na świecie", "polecam każdemu", 
        "oszuści", "tragedia!!!", "odradzam", "omijajcie z daleka"
    ]
    matches = sum(1 for p in suspicious_patterns if p in text_lower)
    if matches >= 2 or (matches >= 1 and word_count < 12):
        return {
            "verdict": "suspicious",
            "verdict_label": "Podejrzana",
            "confidence": 65.0,
            "reasoning": "Tekst zawiera silnie nacechowane, szablonowe zwroty emocjonalne przy stosunkowo niskiej szczegółowości opisu."
        }
        
    return {
        "verdict": "genuine",
        "verdict_label": "Prawdziwa",
        "confidence": 75.0,
        "reasoning": "Opinia nie wykazuje oczywistych cech manipulacji, szablonowości ani powtarzalności charakterystycznej dla fałszywych recenzji."
    }

def detect_fake_review_with_llm(text: str) -> Dict[str, Any]:
    """
    Analyzes review text using Google Gemini 2.0 Flash to detect if it's likely genuine,
    suspicious, or a fake/manipulated review.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return detect_fake_review_fallback(text)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    prompt = f"""Przeanalizuj poniższą recenzję pod kątem autentyczności (czy jest to prawdziwa opinia klienta opisująca rzeczywiste doświadczenia, czy też fake/manipulacja napisana na zlecenie lub wygenerowana przez AI).
Weź pod uwagę:
- Brak konkretnych detali (brak nazw dań, specyficznych sytuacji, konkretnych wad/zalet, brak imion pracowników czy opisów pokoi/miejsca).
- Nadmierną, nienaturalną ekscytację i przesadne pochwały (lub skrajną, bezpodstawną, ogólnikową krytykę).
- Szablonową strukturę zdań i nadmiernie poprawną gramatykę bez potocznych zwrotów typowych dla ludzi.
- Użycie marketingowych frazesów.

Tekst opinii do analizy:
"{text}"

Zwróć odpowiedź WYŁĄCZNIE w formacie JSON o poniższej strukturze (bez żadnego formatowania markdown ```json, tylko czysty ciąg JSON):
{{
  "verdict": "genuine" | "suspicious" | "fake",
  "verdict_label": "Prawdziwa" | "Podejrzana" | "Prawdopodobnie fake",
  "confidence": 85.0,
  "reasoning": "Wyjaśnienie w języku polskim (2-3 zdania), dlaczego opinia uzyskała taki werdykt, odnosząc się do cech tego konkretnego tekstu."
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
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            res_data = response.json()
            candidates = res_data.get("candidates", [])
            if candidates:
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text_content:
                    parsed = json.loads(text_content)
                    return {
                        "verdict": str(parsed.get("verdict", "genuine")),
                        "verdict_label": str(parsed.get("verdict_label", "Prawdziwa")),
                        "confidence": float(parsed.get("confidence", 75.0)),
                        "reasoning": str(parsed.get("reasoning", ""))
                    }
    except Exception:
        pass
        
    return detect_fake_review_fallback(text)

def chat_with_reviews_with_llm(message: str, reviews: list[str], history: list[dict[str, str]]) -> str:
    """
    Sends the user query and analyzed CSV reviews context along with history to Gemini 2.0 Flash.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Błąd: Klucz API Gemini nie jest skonfigurowany. Konwersacja z plikiem CSV wymaga aktywnego klucza API."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Format and limit reviews context (limit to top 100 reviews or approx 20k chars)
    limit = 100
    limited_reviews = reviews[:limit]
    reviews_block = "\n".join([f"- {r.strip()}" for r in limited_reviews if r.strip()])
    
    system_instruction = (
        "Jesteś inteligentnym i analitycznym asystentem biznesowym TextScope.\n"
        "Twoim zadaniem jest odpowiadanie na pytania użytkownika dotyczące przesłanych recenzji klientów (pochodzących z pliku CSV).\n"
        "Poniżej znajduje się lista tych opinii:\n"
        f"{reviews_block}\n\n"
        "Wymagania dotyczące odpowiedzi:\n"
        "- Odpowiadaj wyłącznie w języku polskim.\n"
        "- Bądź obiektywny, wyciągaj konstruktywne wnioski i analizuj trendy (np. słabe strony, mocne strony, powtarzające się problemy).\n"
        "- Odnoś się do konkretnych recenzji z listy, jeśli to możliwe.\n"
        "- Jeśli dane w recenzjach nie pozwalają odpowiedzieć na pytanie, poinformuj o tym użytkownika wprost.\n"
        "- Formatuj odpowiedzi za pomocą Markdown (nagłówki, listy wypunktowane, pogrubienia), aby były bardzo czytelne."
    )
    
    gemini_contents = []
    
    for msg in history:
        role = "user" if msg.get("role") == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "").strip()}]
        })
        
    gemini_contents.append({
        "role": "user",
        "parts": [{"text": message.strip()}]
    })
    
    payload = {
        "contents": gemini_contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            res_data = response.json()
            candidates = res_data.get("candidates", [])
            if candidates:
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text_content:
                    return text_content
            return "Błąd: Nie udało się uzyskać odpowiedzi od modelu AI."
        else:
            return f"Błąd API Gemini: Status {response.status_code}"
    except Exception as e:
        return f"Błąd połączenia z API Gemini: {str(e)}"

def generate_timeline_annotations(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Calls Google Gemini 2.0 Flash API to detect emotional turning points in a sequence of sentences.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Prepare sentence data representation for LLM prompt
    sentences_str = ""
    for idx, s in enumerate(sentences):
        sentences_str += f"Index: {idx} | Score: {s.get('sentiment_score', 0.0)} | Tekst: \"{s.get('sentence', '')}\"\n"
        
    prompt = f"""Przeanalizuj poniższą sekwencję zdań z tekstu pod kątem dynamiki emocjonalnej.
Wskaż od 1 do maksymalnie 3 kluczowych punktów zwrotnych (turning points), gdzie następuje nagła zmiana tonu (np. z pochwały na krytykę, nagłe rozczarowanie, sarkazm, zmiana tematu lub podsumowanie).
Zdania do analizy (indeks, lokalny wynik sentymentu i tekst):
{sentences_str}

Dla każdego punktu zwrotnego podaj:
1. Indeks zdania (0-indexed).
2. Krótki, chwytliwy tytuł po polsku (np. "Nagłe rozczarowanie", "Ironia", "Podsumowanie", "Zmiana tonu").
3. Krótkie wyjaśnienie po polsku (1 zdanie), dlaczego ten moment jest kluczowy.

Zwróć odpowiedź WYŁĄCZNIE w formacie JSON o poniższej strukturze (bez żadnego formatowania markdown ```json, tylko czysty ciąg JSON):
{{
  "turning_points": [
    {{ "index": 2, "title": "Tytuł", "reason": "Wyjaśnienie..." }}
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
            candidates = res_data.get("candidates", [])
            if candidates:
                text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                if text_content:
                    parsed = json.loads(text_content)
                    return parsed.get("turning_points", [])
    except Exception:
        pass
        
    return []
