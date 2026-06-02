# -*- coding: utf-8 -*-
"""
Topic extraction analyzer.
Identifies relevant topics (e.g. food, service, price, ambiance) and extracts top keywords.
"""

import re
from collections import Counter
from typing import Dict, List, Union

# Categorized keyword lists for topic detection
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    'jedzenie': [
        'food', 'meal', 'dish', 'menu', 'taste', 'delicious', 'tasty',
        'cook', 'cooked', 'fresh', 'flavor', 'flavour', 'recipe', 'eat',
        'breakfast', 'lunch', 'dinner', 'pizza', 'burger', 'salad', 'soup',
        'steak', 'sushi', 'dessert', 'cake', 'bread', 'cheese', 'sauce',
        'spicy', 'sweet', 'salty', 'bland', 'overcooked', 'undercooked',
        'portion', 'portions', 'ingredients', 'appetizer', 'entree',
        'jedzenie', 'danie', 'smak', 'posiłek', 'kuchnia', 'smaczne',
        'pyszne', 'świeże', 'gotowane', 'porcja', 'porcje', 'składniki'
    ],
    'obsługa': [
        'service', 'staff', 'waiter', 'waitress', 'server', 'manager',
        'friendly', 'polite', 'rude', 'slow', 'fast', 'quick', 'attentive',
        'helpful', 'professional', 'unprofessional', 'efficient', 'waiting',
        'waited', 'response', 'attitude', 'hospitable', 'courteous',
        'obsługa', 'kelner', 'kelnerka', 'personel', 'pracownik', 'miły',
        'uprzejmy', 'niegrzeczny', 'wolny', 'szybki', 'profesjonalny'
    ],
    'cena': [
        'price', 'prices', 'expensive', 'cheap', 'affordable', 'cost',
        'value', 'money', 'worth', 'overpriced', 'underpriced', 'budget',
        'bill', 'tip', 'charge', 'charged', 'pay', 'paid', 'discount',
        'deal', 'bargain', 'reasonable', 'pricey', 'costly',
        'cena', 'ceny', 'drogi', 'tani', 'kosztuje', 'pieniądze',
        'rachunek', 'wartość', 'opłata', 'zniżka', 'promocja'
    ],
    'atmosfera': [
        'atmosphere', 'ambiance', 'ambience', 'decor', 'decoration',
        'interior', 'design', 'cozy', 'warm', 'cold', 'noisy', 'quiet',
        'loud', 'music', 'lighting', 'clean', 'dirty', 'comfortable',
        'uncomfortable', 'spacious', 'crowded', 'romantic', 'modern',
        'classic', 'elegant', 'view', 'location', 'setting',
        'atmosfera', 'wystrój', 'wnętrze', 'przytulny', 'ciepły',
        'hałaśliwy', 'cichy', 'czysty', 'brudny', 'wygodny', 'przestronny'
    ],
    'jakość': [
        'quality', 'standard', 'level', 'good', 'bad', 'great', 'poor',
        'excellent', 'terrible', 'mediocre', 'average', 'superior',
        'inferior', 'premium', 'luxury', 'basic', 'consistent',
        'jakość', 'standard', 'poziom', 'dobry', 'zły', 'świetny',
        'kiepski', 'doskonały', 'okropny', 'przeciętny', 'średni'
    ],
    'dostawa': [
        'delivery', 'deliver', 'delivered', 'shipping', 'ship', 'shipped',
        'package', 'packaging', 'box', 'arrived', 'arrival', 'tracking',
        'courier', 'late', 'early', 'on time', 'delayed', 'damage',
        'damaged', 'broken', 'intact', 'order', 'ordered',
        'dostawa', 'dostarczony', 'paczka', 'opakowanie', 'kurier',
        'przesyłka', 'zamówienie', 'spóźniony', 'uszkodzony'
    ]
}

def extract_topics(text: str) -> Dict[str, Union[Dict[str, Dict[str, Union[List[str], int, float]]], List[Dict[str, Union[str, int]]], int]]:
    """
    Extracts high-relevance topics and lists top meaningful word frequencies from text.
    
    Args:
        text (str): The input text.
        
    Returns:
        Dict[str, Union[Dict[str, Dict[str, Union[List[str], int, float]]], List[Dict[str, Union[str, int]]], int]]: Map containing
        classified category details, top words frequency lists, and total topic counts.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_set = set(words)
    total_words = max(len(words), 1)

    found_topics = {}
    for category, keywords in TOPIC_KEYWORDS.items():
        matched = [w for w in keywords if w in word_set]
        if matched:
            count = sum(words.count(w) for w in matched)
            relevance = round(count / total_words * 100, 1)
            found_topics[category] = {
                'keywords': matched[:5],
                'count': count,
                'relevance': relevance
            }

    sorted_topics = dict(
        sorted(found_topics.items(), key=lambda x: x[1]['relevance'], reverse=True)
    )

    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
                  'but', 'in', 'with', 'for', 'to', 'of', 'was', 'were',
                  'i', 'w', 'na', 'z', 'do', 'nie', 'to', 'jest', 'się'}
    meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
    word_freq = Counter(meaningful_words)
    top_keywords = [{'word': w, 'count': c} for w, c in word_freq.most_common(8)]

    return {
        'categories': sorted_topics,
        'top_keywords': top_keywords,
        'total_topics_found': len(sorted_topics)
    }
