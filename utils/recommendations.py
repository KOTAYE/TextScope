# -*- coding: utf-8 -*-
"""
Recommendations helper module.
Generates structural tips and actionable advice in Polish for improving text style, sentiment balance, or AI stylometry.
"""

from typing import Dict, List, Optional, Union

def generate_sentiment_recommendations(
    sentiment_result: Dict[str, Union[str, float, Dict[str, float]]],
    emotions_result: Optional[Dict[str, Union[Dict[str, float], Optional[str]]]] = None
) -> List[str]:
    """
    Formulates sentiment advice based on polarity ratings and detected emotional states.
    
    Args:
        sentiment_result (Dict): Polarity score and category output.
        emotions_result (Optional[Dict]): Emotional state intensity outputs.
        
    Returns:
        List[str]: Polish recommendation strings.
    """
    recommendations = []
    score = float(sentiment_result.get('combined_score', 0.0))

    if score >= 0.3:
        recommendations.append("✅ Tekst ma wyraźnie pozytywny wydźwięk. Dobrze nadaje się jako pozytywna recenzja.")
    elif score >= 0.1:
        recommendations.append("🙂 Tekst jest raczej pozytywny, ale mógłby być bardziej entuzjastyczny.")
    elif score <= -0.3:
        recommendations.append("⚠️ Tekst ma wyraźnie negatywny wydźwięk. Rozważ bardziej konstruktywną krytykę.")
        recommendations.append("💡 Spróbuj dodać też pozytywne aspekty, aby recenzja była bardziej wyważona.")
    elif score <= -0.1:
        recommendations.append("😕 Tekst jest raczej negatywny. Możesz go złagodzić dodając konkretne sugestie.")
    else:
        recommendations.append("😐 Tekst jest neutralny. Dodaj więcej emocji lub opinii, aby był bardziej angażujący.")

    textblob_data = sentiment_result.get('textblob', {})
    if isinstance(textblob_data, dict):
        subjectivity = float(textblob_data.get('subjectivity', 50.0))
        if subjectivity > 80:
            recommendations.append("📝 Tekst jest bardzo subiektywny. Dodaj fakty lub konkretne przykłady.")
        elif subjectivity < 20:
            recommendations.append("📊 Tekst jest bardzo obiektywny. Dodaj osobiste odczucia, jeśli to recenzja.")

    if emotions_result:
        dominant = emotions_result.get('dominant')
        if dominant == 'anger':
            recommendations.append("😡 Wykryto dużo złości. Spróbuj wyrazić niezadowolenie spokojniej.")
        elif dominant == 'sadness':
            recommendations.append("😢 Tekst wyraża smutek. Dodaj elementy nadziei lub konstruktywne wnioski.")
        elif dominant == 'fear':
            recommendations.append("😨 Tekst wyraża strach/obawy. Spróbuj przedstawić rozwiązania.")

    return recommendations

def generate_ai_recommendations(ai_result: Dict[str, Union[float, str, Dict[str, float], bool]]) -> List[str]:
    """
    Formulates actionable tips to make text sound more human and avoid predictable structures.
    
    Args:
        ai_result (Dict): AI detector metrics output.
        
    Returns:
        List[str]: Actions list in Polish.
    """
    recommendations = []
    ai_prob = float(ai_result.get('ai_probability', 0.0))
    details = ai_result.get('details', {})

    if ai_result.get('error'):
        recommendations.append("ℹ️ Tekst jest zbyt krótki. Napisz co najmniej 2-3 zdania dla dokładnej analizy.")
        return recommendations

    if ai_prob >= 75:
        recommendations.append("🤖 Tekst wygląda na wygenerowany przez AI. Oto jak go poprawić:")
        recommendations.append("💡 Dodaj osobiste doświadczenia i konkretne szczegóły.")
        recommendations.append("💡 Użyj bardziej potocznego języka i krótszych zdań.")
        recommendations.append("💡 Dodaj emocje, humor lub osobiste opinie.")
    elif ai_prob >= 55:
        recommendations.append("🔍 Tekst ma pewne cechy tekstu AI. Rozważ poprawki:")
        recommendations.append("💡 Urozmaić długość zdań - niech będą różnej długości.")
        recommendations.append("💡 Unikaj nadmiernie formalnych zwrotów.")
    elif ai_prob >= 40:
        recommendations.append("❓ Nie można jednoznacznie określić autora tekstu.")
    else:
        recommendations.append("✅ Tekst wygląda na napisany przez człowieka.")

    if isinstance(details, dict):
        if float(details.get('Jednorodność zdań', 0.0)) > 70:
            recommendations.append("📏 Zdania mają zbyt podobną długość. Zmieniaj ich długość.")
        if float(details.get('Formalność stylu', 0.0)) > 70:
            recommendations.append("🎩 Styl jest zbyt formalny. Użyj bardziej naturalnego języka.")
        if float(details.get('Gęstość konektorów', 0.0)) > 70:
            recommendations.append("🔗 Za dużo łączników (moreover, however...). Uprość strukturę zdań.")

    return recommendations
