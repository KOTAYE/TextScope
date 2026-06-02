# -*- coding: utf-8 -*-
"""
Emotion analysis module.
Detects intensity scores for primary emotions: Joy, Anger, Sadness, Fear, Surprise, and Disgust.
"""

import re
from typing import Dict, Optional, Union

# Keywords dictionary mapping emotions to relevant words
EMOTION_WORDS: Dict[str, list[str]] = {
    'joy': [
        'happy', 'joy', 'love', 'wonderful', 'amazing', 'great', 'excellent',
        'fantastic', 'awesome', 'beautiful', 'perfect', 'delighted', 'pleased',
        'glad', 'cheerful', 'fun', 'enjoy', 'smile', 'laugh', 'best', 'brilliant',
        'superb', 'outstanding', 'magnificent', 'thrilled', 'excited', 'paradise',
        'heaven', 'bliss', 'ecstatic', 'lovely', 'adore', 'charming', 'bright',
        'good', 'nice', 'liked', 'like', 'recommend', 'favorite', 'favourite',
        'worth', 'friendly', 'helpful', 'welcoming', 'comfortable', 'cozy'
    ],
    'anger': [
        'angry', 'hate', 'terrible', 'horrible', 'worst', 'awful', 'furious',
        'annoyed', 'frustrated', 'rage', 'disgusting', 'outrageous', 'rude',
        'unacceptable', 'pathetic', 'infuriating', 'irritating', 'mad', 'pissed',
        'hostile', 'aggressive', 'violent', 'cruel', 'brutal', 'ruthless',
        'disrespectful', 'insulting', 'offensive', 'abusive', 'toxic'
    ],
    'sadness': [
        'sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'sorry',
        'regret', 'unfortunate', 'gloomy', 'heartbroken', 'cry', 'tears',
        'lonely', 'lost', 'grief', 'mourn', 'sorrow', 'melancholy', 'hopeless',
        'devastated', 'tragic', 'painful', 'suffering', 'despair', 'empty',
        'miss', 'missing', 'gone', 'abandoned', 'forgotten'
    ],
    'fear': [
        'scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous',
        'panic', 'horror', 'dread', 'frightened', 'alarmed', 'concerned',
        'uneasy', 'threatened', 'danger', 'dangerous', 'risk', 'risky',
        'creepy', 'spooky', 'eerie', 'haunted', 'nightmare', 'phobia',
        'terror', 'intimidating', 'suspicious', 'paranoid'
    ],
    'surprise': [
        'surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
        'wow', 'incredible', 'unbelievable', 'stunning', 'remarkable',
        'speechless', 'mind-blowing', 'jaw-dropping', 'extraordinary',
        'phenomenal', 'startled', 'bewildered', 'awestruck', 'whoa',
        'omg', 'unreal', 'insane', 'crazy'
    ],
    'disgust': [
        'disgusting', 'gross', 'nasty', 'repulsive', 'revolting', 'sickening',
        'vile', 'yuck', 'eww', 'filthy', 'dirty', 'contaminated', 'rotten',
        'stinky', 'smelly', 'moldy', 'spoiled', 'foul', 'putrid', 'nauseating',
        'repugnant', 'hideous', 'grotesque', 'trash', 'garbage', 'waste'
    ]
}

def detect_emotions(text: str) -> Dict[str, Union[Dict[str, float], Optional[str]]]:
    """
    Analyzes the input text for emotional keyword counts and determines the dominant emotion.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        Dict[str, Union[Dict[str, float], Optional[str]]]: Dictionary containing individual emotion scores,
        the dominant emotion code, its Polish name, and representing emoji.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    total = max(len(words), 1)

    scores = {}
    for emotion, keywords in EMOTION_WORDS.items():
        count = sum(1 for w in words if w in keywords)
        raw_score = (count / total) * 100 * 8
        scores[emotion] = round(min(raw_score, 100), 1)

    dominant = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else None

    emoji_map = {
        'joy': '😄', 'anger': '😡', 'sadness': '😢',
        'fear': '😨', 'surprise': '😲', 'disgust': '🤢'
    }
    name_map = {
        'joy': 'Radość', 'anger': 'Złość', 'sadness': 'Smutek',
        'fear': 'Strach', 'surprise': 'Zaskoczenie', 'disgust': 'Obrzydzenie'
    }

    return {
        'scores': scores,
        'dominant': dominant,
        'dominant_name': name_map.get(dominant, '—'),
        'dominant_emoji': emoji_map.get(dominant, '😐')
    }
