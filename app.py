# -*- coding: utf-8 -*-

import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from flask import Flask, render_template, request, jsonify, Response
import nltk
from textblob import TextBlob
import re
import math
import string
import csv
import io
import json
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

sid = SentimentIntensityAnalyzer()

def count_syllables(word):

    word = word.lower().strip()
    if not word:
        return 1
    count = 0
    vowels = 'aeiouyаеєиіїоуюя'
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    return max(count, 1)
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]
COMMON_WORDS = {
    'pl': [
        'i', 'w', 'na', 'z', 'do', 'nie', 'to', 'jest', 'że', 'się',
        'co', 'jak', 'ale', 'za', 'od', 'po', 'tak', 'ten', 'o', 'czy',
        'tego', 'tej', 'tym', 'był', 'była', 'było', 'dla', 'ze', 'tylko',
        'bardzo', 'już', 'może', 'jeszcze', 'tu', 'jego', 'jej', 'ich',
        'kiedy', 'teraz', 'więc', 'gdzie', 'też', 'kto', 'nic', 'przez'
    ],
    'en': [
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
        'in', 'with', 'he', 'she', 'it', 'was', 'for', 'are', 'were', 'be',
        'has', 'have', 'had', 'not', 'this', 'that', 'from', 'they', 'we',
        'been', 'will', 'would', 'could', 'should', 'their', 'there', 'what',
        'when', 'where', 'how', 'who', 'did', 'does', 'do', 'if', 'my', 'your'
    ],
    'uk': [
        'і', 'в', 'на', 'з', 'до', 'не', 'це', 'є', 'що', 'як',
        'але', 'за', 'від', 'по', 'так', 'цей', 'ця', 'о', 'чи',
        'він', 'вона', 'воно', 'вони', 'ми', 'ви', 'та', 'або',
        'був', 'була', 'було', 'були', 'тут', 'де', 'коли', 'його',
        'її', 'їх', 'dla', 'тільки', 'дуже', 'ще', 'може', 'вже'
    ]
}

LANG_SPECIFIC_CHARS = {
    'pl': 'ąęćłńóśźżĄĘĆŁŃÓŚŹŻ',
    'uk': 'іїєґІЇЄҐ'
}

LANG_NAMES = {
    'pl': 'Polski',
    'en': 'Angielski',
    'uk': 'Ukraiński',
    'unknown': 'Nieznany'
}

def detect_language(text):
    if not text or len(text.strip()) < 5:
        return {'code': 'unknown', 'name': LANG_NAMES['unknown'], 'confidence': 0}

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = max(len(words), 1)

    char_scores = {}
    for lang, chars in LANG_SPECIFIC_CHARS.items():
        count = sum(1 for ch in text if ch in chars)
        char_scores[lang] = count

    for lang, count in char_scores.items():
        if count >= 3:
            return {
                'code': lang,
                'name': LANG_NAMES[lang],
                'confidence': min(95, 70 + count * 3)
            }

    word_scores = {}
    for lang, common in COMMON_WORDS.items():
        matches = sum(1 for w in words if w in common)
        word_scores[lang] = matches / total_words * 100

    cyrillic_count = sum(1 for ch in text if '\u0400' <= ch <= '\u04FF')
    latin_count = sum(1 for ch in text if ch.isalpha() and ch.isascii())

    if latin_count > cyrillic_count:
        word_scores.pop('uk', None)
    else:
        word_scores.pop('en', None)
        word_scores.pop('pl', None)

    if not word_scores:
        return {'code': 'unknown', 'name': LANG_NAMES['unknown'], 'confidence': 0}

    best_lang = max(word_scores, key=word_scores.get)
    best_score = word_scores[best_lang]

    if best_score < 3:
        if cyrillic_count > latin_count:
            best_lang = 'uk'
        else:
            best_lang = 'en'
        confidence = 30
    else:
        confidence = min(95, int(best_score * 3 + 30))

    return {
        'code': best_lang,
        'name': LANG_NAMES.get(best_lang, 'Nieznany'),
        'confidence': confidence
    }

TOPIC_KEYWORDS = {
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

def extract_topics(text):

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

def generate_sentiment_recommendations(sentiment_result, emotions_result=None):

    recommendations = []
    score = sentiment_result.get('combined_score', 0)
    category = sentiment_result.get('category', '')

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

    subjectivity = sentiment_result.get('textblob', {}).get('subjectivity', 50)
    if subjectivity > 80:
        recommendations.append("📝 Tekst jest bardzo subiektywny. Dodaj fakty lub konkretne przykłady.")
    elif subjectivity < 20:
        recommendations.append("📊 Tekst jest bardzo obiektywny. Dodaj osobiste odczucia, jeśli to recenzja.")

    if emotions_result:
        dominant = emotions_result.get('dominant', '')
        if dominant == 'anger':
            recommendations.append("😡 Wykryto dużo złości. Spróbuj wyrazić niezadowolenie spokojniej.")
        elif dominant == 'sadness':
            recommendations.append("😢 Tekst wyraża smutek. Dodaj elementy nadziei lub konstruktywne wnioski.")
        elif dominant == 'fear':
            recommendations.append("😨 Tekst wyraża strach/obawy. Spróbuj przedstawić rozwiązania.")

    return recommendations

def generate_ai_recommendations(ai_result):

    recommendations = []
    ai_prob = ai_result.get('ai_probability', 0)
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

    if details.get('Jednorodność zdań', 0) > 70:
        recommendations.append("📏 Zdania mają zbyt podobną długość. Zmieniaj ich długość.")
    if details.get('Formalność stylu', 0) > 70:
        recommendations.append("🎩 Styl jest zbyt formalny. Użyj bardziej naturalnego języka.")
    if details.get('Gęstość konektorów', 0) > 70:
        recommendations.append("🔗 Za dużo łączników (moreover, however...). Uprość strukturę zdań.")

    return recommendations

POLISH_LEXICON = {
    'positive': [
        'wspanial', 'swietn', 'dobr', 'polecam', 'super', 'pyszn', 'smaczn', 'mil',
        'profesjonal', 'rewelac', 'koch', 'ladn', 'czyst', 'bardzo', 'przyjemn',
        'ideal', 'szybko', 'zadowol', 'rewelacja', 'ekstra', 'klasa', 'pomocn',
        'podob', 'zachwyc', 'fajn'
    ],
    'negative': [
        'zly', 'kiepsk', 'okropn', 'niedobr', 'drog', 'brudn', 'niepolecam', 'odradzam',
        'traged', 'dramat', 'glup', 'beznadziej', 'fatal', 'woln', 'niemil', 'brud',
        'oszust', 'porazka', 'niesmaczn', 'czar', 'koszmar', 'najgorsz', 'nudn',
        'wstretn'
    ]
}

UKRAINIAN_LEXICON = {
    'positive': [
        'чудов', 'класн', 'добр', 'рекоменду', 'супер', 'смачн', 'мил', 'професійн',
        'коха', 'любл', 'гарн', 'чист', 'дуже', 'приємн', 'ідеал', 'швидк', 'задовол',
        'клас', 'допомог', 'кращ', 'файн', 'смачно', 'подоб', 'захоп'
    ],
    'negative': [
        'поган', 'гірш', 'жахлив', 'недоподоб', 'дорог', 'брудн', 'нерекоменду',
        'жаль', 'трагед', 'драм', 'дурн', 'безнадій', 'фатал', 'повільн', 'немил',
        'неприємн', 'жах', 'кошмар', 'найгірш', 'обман', 'шахрай', 'нудн'
    ]
}

def normalize_text_for_lexicon(text):
    text = text.lower()
    accents = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z'
    }
    for char, repl in accents.items():
        text = text.replace(char, repl)
    return text

def analyze_multilingual_sentiment(text, lang_code):
    text_norm = normalize_text_for_lexicon(text)
    words = re.findall(r'\b\w+\b', text_norm)
    if lang_code == 'pl':
        lex = POLISH_LEXICON
    elif lang_code == 'uk':
        lex = UKRAINIAN_LEXICON
    else:
        return 0.0, 0, 0
    pos_count = 0
    neg_count = 0
    for word in words:
        is_pos = any(word.startswith(stem) or stem in word for stem in lex['positive'])
        is_neg = any(word.startswith(stem) or stem in word for stem in lex['negative'])
        if is_pos:
            pos_count += 1
        if is_neg:
            neg_count += 1
    total = pos_count + neg_count
    if total == 0:
        return 0.0, 0, 0
    score = (pos_count - neg_count) / total
    return score, pos_count, neg_count

def analyze_sentiment(text):
    lang_info = detect_language(text)
    lang_code = lang_info['code']
    vader_scores = sid.polarity_scores(text)
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity
    if lang_code in ['pl', 'uk']:
        custom_score, pos_cnt, neg_cnt = analyze_multilingual_sentiment(text, lang_code)
        if pos_cnt + neg_cnt > 0:
            combined = custom_score * 0.8 + (vader_scores['compound'] * 0.05) + (tb_polarity * 0.15)
        else:
            combined = (vader_scores['compound'] * 0.6) + (tb_polarity * 0.4)
    else:
        combined = (vader_scores['compound'] * 0.6) + (tb_polarity * 0.4)
    if combined >= 0.3:
        category = "Pozytywny"
        emoji = "😊"
    elif combined <= -0.3:
        category = "Negatywny"
        emoji = "😠"
    elif combined >= 0.1:
        category = "Raczej pozytywny"
        emoji = "🙂"
    elif combined <= -0.1:
        category = "Raczej negatywny"
        emoji = "😕"
    else:
        category = "Neutralny"
        emoji = "😐"
    positive_pct = round((combined + 1) / 2 * 100, 1)
    negative_pct = round(100 - positive_pct, 1)
    return {
        'category': category,
        'emoji': emoji,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'combined_score': round(combined, 3),
        'vader': {
            'positive': round(vader_scores['pos'] * 100, 1),
            'negative': round(vader_scores['neg'] * 100, 1),
            'neutral': round(vader_scores['neu'] * 100, 1),
            'compound': round(vader_scores['compound'], 3)
        },
        'textblob': {
            'polarity': round(tb_polarity, 3),
            'subjectivity': round(tb_subjectivity * 100, 1)
        }
    }

def get_word_sentiments(text, lang_code=None):
    if not lang_code:
        lang_code = detect_language(text)['code']
    tokens = re.findall(r'\S+|\s+', text)
    result = []
    for token in tokens:
        if token.strip() == '':
            result.append({'text': token, 'score': 0, 'type': 'space'})
            continue
        clean = re.sub(r'[^\w\'-]', '', token)
        clean_lower = clean.lower()
        score = 0.0
        word_type = 'neutral'
        if clean and len(clean) > 1:
            if lang_code == 'en':
                score = sid.polarity_scores(clean)['compound']
                if score >= 0.3:
                    word_type = 'positive'
                elif score <= -0.3:
                    word_type = 'negative'
                else:
                    word_type = 'neutral'
            else:
                clean_norm = normalize_text_for_lexicon(clean_lower)
                if lang_code == 'pl':
                    lex = POLISH_LEXICON
                elif lang_code == 'uk':
                    lex = UKRAINIAN_LEXICON
                else:
                    lex = None
                if lex:
                    is_pos = any(clean_norm.startswith(stem) or stem in clean_norm for stem in lex['positive'])
                    is_neg = any(clean_norm.startswith(stem) or stem in clean_norm for stem in lex['negative'])
                    if is_pos:
                        score = 0.8
                        word_type = 'positive'
                    elif is_neg:
                        score = -0.8
                        word_type = 'negative'
                if word_type == 'neutral' and len(clean) > 3:
                    score = sid.polarity_scores(clean)['compound']
                    if score >= 0.3:
                        word_type = 'positive'
                    elif score <= -0.3:
                        word_type = 'negative'
        result.append({
            'text': token,
            'score': round(score, 3),
            'type': word_type
        })
    return result

EMOTION_WORDS = {
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

def detect_emotions(text):

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

def get_text_stats(text):

    words = re.findall(r'\b\w+\b', text)
    sentences = split_into_sentences(text)
    syllables = sum(count_syllables(w) for w in words)

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    char_count = len(text)
    char_no_spaces = len(text.replace(' ', ''))

    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)
    avg_sentence_len = round(word_count / sentence_count, 1)

    unique_words = set(w.lower() for w in words)
    ttr = round(len(unique_words) / max(word_count, 1) * 100, 1)

    if word_count > 0 and sentence_count > 0:
        asl = word_count / sentence_count
        asw = syllables / word_count
        readability = 206.835 - (1.015 * asl) - (84.6 * asw)
        readability = max(0, min(100, round(readability, 1)))
    else:
        readability = 0

    if readability >= 80:
        difficulty = "Łatwy"
        diff_color = "#4ade80"
    elif readability >= 60:
        difficulty = "Średni"
        diff_color = "#facc15"
    elif readability >= 40:
        difficulty = "Trudny"
        diff_color = "#fb923c"
    else:
        difficulty = "Bardzo trudny"
        diff_color = "#f87171"

    word_freq = Counter(w.lower() for w in words if len(w) > 3)
    top_words = [{'word': w, 'count': c} for w, c in word_freq.most_common(5)]

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'char_count': char_count,
        'char_no_spaces': char_no_spaces,
        'avg_word_len': avg_word_len,
        'avg_sentence_len': avg_sentence_len,
        'unique_words': len(unique_words),
        'vocabulary_richness': ttr,
        'readability': readability,
        'difficulty': difficulty,
        'difficulty_color': diff_color,
        'syllable_count': syllables,
        'top_words': top_words
    }

def calculate_sentence_uniformity(sentences):

    if len(sentences) < 2:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    if avg_len == 0:
        return 0.5

    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / avg_len

    if cv < 0.15:
        return 0.9
    elif cv < 0.3:
        return 0.7
    elif cv < 0.5:
        return 0.5
    elif cv < 0.8:
        return 0.3
    else:
        return 0.15

def calculate_vocabulary_richness(text):

    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5:
        return 0.5

    ttr = len(set(words)) / len(words)

    if ttr > 0.75:
        return 0.7
    elif ttr > 0.6:
        return 0.55
    elif ttr > 0.45:
        return 0.4
    else:
        return 0.3

def calculate_formality_score(text):

    text_lower = text.lower()

    informal_markers = [
        r'\b(lol|haha|хаха|ахах|омг|wtf|btw|imho|кста|чел|норм|ок|окей|tbh|ngl|fr|imo|smh|idk)\b',
        r'\.{3,}',
        r'!{2,}',
        r'\?{2,}',
    ]

    informal_count = 0
    for pattern in informal_markers:
        informal_count += len(re.findall(pattern, text_lower))

    formal_markers = [
        r'\b(furthermore|moreover|additionally|consequently|nevertheless|however|therefore|thus|hence|accordingly)\b',
        r'\b(крім того|більш того|отже|таким чином|відповідно|зокрема|водночас|безперечно)\b',
        r'\b(it is important to note|it should be noted|it is worth mentioning)\b',
        r'\b(варто зазначити|слід зауважити|необхідно підкреслити)\b',
        r'\b(in conclusion|to summarize|in summary|overall)\b',
    ]

    formal_count = 0
    for pattern in formal_markers:
        formal_count += len(re.findall(pattern, text_lower))

    words_count = max(len(text.split()), 1)
    informal_ratio = informal_count / words_count
    formal_ratio = formal_count / words_count

    if formal_ratio > 0.03:
        score = 0.85
    elif formal_ratio > 0.015:
        score = 0.7
    elif formal_ratio > 0.005:
        score = 0.6
    elif informal_ratio > 0.05:
        score = 0.15
    elif informal_ratio > 0.02:
        score = 0.25
    else:
        score = 0.45

    return score

def calculate_repetition_patterns(text):

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 3:
        return 0.5

    starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
    start_counter = Counter(starts)
    most_common = start_counter.most_common(1)[0][1] if start_counter else 0
    start_rep = most_common / len(sentences)

    words = re.findall(r'\b\w+\b', text)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

    score = 0.5
    if avg_word_len > 6:
        score += 0.15
    if start_rep > 0.4:
        score += 0.1

    return min(score, 1.0)

def calculate_punctuation_score(text):

    if len(text) < 20:
        return 0.5

    commas = text.count(',')
    periods = text.count('.')
    excl = text.count('!')
    quest = text.count('?')
    semi = text.count(';')
    colons = text.count(':')

    total_punct = commas + periods + excl + quest + semi + colons
    words_count = max(len(text.split()), 1)
    punct_ratio = total_punct / words_count

    if 0.12 < punct_ratio < 0.28:
        score = 0.65
    else:
        score = 0.35

    if semi > 0 or colons > 1:
        score += 0.1
    if excl > 2:
        score -= 0.15

    return max(0, min(score, 1.0))

def calculate_connector_density(text):

    text_lower = text.lower()
    connectors = [
        r'\b(however|moreover|furthermore|additionally|consequently|therefore|nevertheless)\b',
        r'\b(in addition|as a result|on the other hand|for instance|for example|in contrast)\b',
        r'\b(significantly|particularly|specifically|essentially|fundamentally)\b',
        r'\b(крім того|більш того|зокрема|таким чином|відповідно|натомість|водночас)\b',
    ]

    connector_count = 0
    for pattern in connectors:
        connector_count += len(re.findall(pattern, text_lower))

    sentences = split_into_sentences(text)
    if not sentences:
        return 0.5

    density = connector_count / len(sentences)

    if density > 0.5:
        return 0.9
    elif density > 0.3:
        return 0.7
    elif density > 0.15:
        return 0.55
    else:
        return 0.3

def detect_ai_text(text):

    if len(text.strip()) < 30:
        return {
            'ai_probability': 0,
            'verdict': 'Tekst jest zbyt krótki do analizy',
            'details': {},
            'error': True
        }

    sentences = split_into_sentences(text)

    uniformity = calculate_sentence_uniformity(sentences)
    vocabulary = calculate_vocabulary_richness(text)
    formality = calculate_formality_score(text)
    repetition = calculate_repetition_patterns(text)
    punctuation = calculate_punctuation_score(text)
    connectors = calculate_connector_density(text)

    weights = {
        'uniformity': 0.20,
        'vocabulary': 0.12,
        'formality': 0.28,
        'repetition': 0.12,
        'punctuation': 0.12,
        'connectors': 0.16
    }

    ai_score = (
        uniformity * weights['uniformity'] +
        vocabulary * weights['vocabulary'] +
        formality * weights['formality'] +
        repetition * weights['repetition'] +
        punctuation * weights['punctuation'] +
        connectors * weights['connectors']
    )

    ai_probability = round(ai_score * 100, 1)
    ai_probability = max(5, min(95, ai_probability))

    if ai_probability >= 75:
        verdict = "Bardzo prawdopodobne, że napisane przez AI"
        verdict_level = "high"
    elif ai_probability >= 55:
        verdict = "Możliwe, że napisane przez AI"
        verdict_level = "medium"
    elif ai_probability >= 40:
        verdict = "Nieokreślone"
        verdict_level = "uncertain"
    elif ai_probability >= 25:
        verdict = "Raczej napisane przez człowieka"
        verdict_level = "low"
    else:
        verdict = "Prawdopodobnie napisane przez człowieka"
        verdict_level = "very_low"

    return {
        'ai_probability': ai_probability,
        'human_probability': round(100 - ai_probability, 1),
        'verdict': verdict,
        'verdict_level': verdict_level,
        'details': {
            'Jednorodność zdań': round(uniformity * 100, 1),
            'Bogactwo leksykalne': round(vocabulary * 100, 1),
            'Formalność stylu': round(formality * 100, 1),
            'Wzorce powtórzeń': round(repetition * 100, 1),
            'Interpunkcja': round(punctuation * 100, 1),
            'Gęstość konektorów': round(connectors * 100, 1)
        },
        'error': False
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    analysis_type = data.get('type', 'sentiment')

    if not text:
        return jsonify({'error': 'Wprowadź tekst do analizy'})

    if analysis_type == 'sentiment':
        sentiment = analyze_sentiment(text)
        words = get_word_sentiments(text)
        emotions = detect_emotions(text)
        stats = get_text_stats(text)
        topics = extract_topics(text)
        language = detect_language(text)
        recommendations = generate_sentiment_recommendations(sentiment, emotions)

        return jsonify({
            'sentiment': sentiment,
            'words': words,
            'emotions': emotions,
            'stats': stats,
            'topics': topics,
            'language': language,
            'recommendations': recommendations
        })

    elif analysis_type == 'ai_detect':
        ai = detect_ai_text(text)
        stats = get_text_stats(text)
        language = detect_language(text)
        recommendations = generate_ai_recommendations(ai)

        return jsonify({
            'ai': ai,
            'stats': stats,
            'language': language,
            'recommendations': recommendations
        })

    else:
        return jsonify({'error': 'Nieznany typ analizy'})

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():

    if 'file' not in request.files:
        return jsonify({'error': 'Nie przesłano pliku CSV'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nie wybrano pliku'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Plik musi mieć rozszerzenie .csv'}), 400

    try:

        stream = io.StringIO(file.stream.read().decode('utf-8'))
        reader = csv.reader(stream)
        header = next(reader, None)

        results = []
        all_scores = []
        emotion_totals = {}
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        row_number = 0

        for row in reader:
            if not row or not row[0].strip():
                continue

            row_number += 1
            review_text = row[0].strip()

            sentiment = analyze_sentiment(review_text)
            emotions = detect_emotions(review_text)
            language = detect_language(review_text)

            all_scores.append(sentiment['combined_score'])

            if sentiment['combined_score'] >= 0.1:
                positive_count += 1
            elif sentiment['combined_score'] <= -0.1:
                negative_count += 1
            else:
                neutral_count += 1

            for emo, score in emotions['scores'].items():
                emotion_totals[emo] = emotion_totals.get(emo, 0) + score

            results.append({
                'row': row_number,
                'text': review_text[:100] + ('...' if len(review_text) > 100 else ''),
                'sentiment': sentiment,
                'dominant_emotion': emotions['dominant_name'],
                'language': language
            })

        if not results:
            return jsonify({'error': 'Plik CSV jest pusty lub nie zawiera recenzji'}), 400

        total = len(results)
        avg_score = round(sum(all_scores) / total, 3) if all_scores else 0

        avg_emotions = {}
        for emo, total_score in emotion_totals.items():
            avg_emotions[emo] = round(total_score / total, 1)

        summary = {
            'total_reviews': total,
            'average_sentiment': avg_score,
            'distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'average_emotions': avg_emotions
        }

        return jsonify({
            'summary': summary,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': f'Błąd podczas przetwarzania pliku: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_texts():

    data = request.get_json()
    text1 = data.get('text1', '').strip()
    text2 = data.get('text2', '').strip()

    if not text1 or not text2:
        return jsonify({'error': 'Wprowadź oba teksty do porównania'}), 400

    result1 = {
        'sentiment': analyze_sentiment(text1),
        'emotions': detect_emotions(text1),
        'ai': detect_ai_text(text1),
        'stats': get_text_stats(text1),
        'topics': extract_topics(text1),
        'language': detect_language(text1)
    }
    result1['recommendations'] = generate_sentiment_recommendations(
        result1['sentiment'], result1['emotions']
    )

    result2 = {
        'sentiment': analyze_sentiment(text2),
        'emotions': detect_emotions(text2),
        'ai': detect_ai_text(text2),
        'stats': get_text_stats(text2),
        'topics': extract_topics(text2),
        'language': detect_language(text2)
    }
    result2['recommendations'] = generate_sentiment_recommendations(
        result2['sentiment'], result2['emotions']
    )

    comparison = {
        'sentiment_diff': round(
            result1['sentiment']['combined_score'] - result2['sentiment']['combined_score'], 3
        ),
        'ai_diff': round(
            result1['ai']['ai_probability'] - result2['ai']['ai_probability'], 1
        ),
        'word_count_diff': result1['stats']['word_count'] - result2['stats']['word_count'],
        'same_language': result1['language']['code'] == result2['language']['code']
    }

    return jsonify({
        'text1': result1,
        'text2': result2,
        'comparison': comparison
    })

@app.route('/export_report', methods=['POST'])
def export_report():

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Brak danych do wygenerowania raportu'}), 400

    text = data.get('text', 'Brak tekstu')
    sentiment = data.get('sentiment', {})
    emotions = data.get('emotions', {})
    stats = data.get('stats', {})
    ai = data.get('ai', {})
    topics = data.get('topics', {})
    language = data.get('language', {})
    recommendations = data.get('recommendations', [])

    html = f"""
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <title>Raport analizy tekstu</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 30px;
                color: #333;
                background: #fff;
            }}
            h1 {{
                color: #6366f1;
                border-bottom: 3px solid #6366f1;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #4f46e5;
                margin-top: 30px;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 5px;
            }}
            .section {{
                margin-bottom: 25px;
                padding: 15px;
                background: #f9fafb;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }}
            .text-box {{
                background: #eef2ff;
                padding: 15px;
                border-radius: 8px;
                font-style: italic;
                border-left: 4px solid #6366f1;
                margin: 10px 0;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}
            th, td {{
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #e5e7eb;
            }}
            th {{
                background: #6366f1;
                color: white;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-weight: bold;
                color: white;
            }}
            .positive {{ background: #22c55e; }}
            .negative {{ background: #ef4444; }}
            .neutral {{ background: #9ca3af; }}
            .recommendation {{
                padding: 8px 12px;
                margin: 5px 0;
                background: #fffbeb;
                border-left: 3px solid #f59e0b;
                border-radius: 4px;
            }}
            .footer {{
                margin-top: 40px;
                text-align: center;
                color: #9ca3af;
                font-size: 12px;
            }}
            @media print {{
                body {{ padding: 10px; }}
                .section {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <h1>📊 Raport analizy tekstu</h1>
        <p><strong>Data:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <h2>📄 Analizowany tekst</h2>
        <div class="text-box">{text[:500]}{'...' if len(text) > 500 else ''}</div>
    """

    if language:
        html += f"""
        <h2>🌍 Wykryty język</h2>
        <div class="section">
            <p><strong>Język:</strong> {language.get('name', '—')}
            (pewność: {language.get('confidence', 0)}%)</p>
        </div>
        """

    if sentiment:
        cat = sentiment.get('category', '—')
        css_class = 'positive' if 'ozytywny' in cat else ('negative' if 'egatywny' in cat else 'neutral')
        html += f"""
        <h2>😊 Analiza sentymentu</h2>
        <div class="section">
            <p><strong>Ocena:</strong>
                <span class="badge {css_class}">
                    {sentiment.get('emoji', '')} {cat}
                </span>
            </p>
            <table>
                <tr><th>Wskaźnik</th><th>Wartość</th></tr>
                <tr><td>Wynik łączony</td><td>{sentiment.get('combined_score', '—')}</td></tr>
                <tr><td>Pozytywność</td><td>{sentiment.get('positive_pct', '—')}%</td></tr>
                <tr><td>Negatywność</td><td>{sentiment.get('negative_pct', '—')}%</td></tr>
                <tr><td>Subiektywność</td><td>{sentiment.get('textblob', {}).get('subjectivity', '—')}%</td></tr>
            </table>
        </div>
        """

    if emotions and emotions.get('scores'):
        emotion_names = {
            'joy': 'Radość', 'anger': 'Złość', 'sadness': 'Smutek',
            'fear': 'Strach', 'surprise': 'Zaskoczenie', 'disgust': 'Obrzydzenie'
        }
        rows = ''
        for emo, score in emotions.get('scores', {}).items():
            rows += f'<tr><td>{emotion_names.get(emo, emo)}</td><td>{score}%</td></tr>'
        html += f"""
        <h2>{emotions.get('dominant_emoji', '😐')} Analiza emocji</h2>
        <div class="section">
            <p><strong>Dominująca emocja:</strong> {emotions.get('dominant_name', '—')}</p>
            <table>
                <tr><th>Emocja</th><th>Intensywność</th></tr>
                {rows}
            </table>
        </div>
        """

    if ai and not ai.get('error'):
        html += f"""
        <h2>🤖 Wykrywanie AI</h2>
        <div class="section">
            <p><strong>Prawdopodobieństwo AI:</strong> {ai.get('ai_probability', '—')}%</p>
            <p><strong>Werdykt:</strong> {ai.get('verdict', '—')}</p>
        </div>
        """

    if stats:
        html += f"""
        <h2>📈 Statystyki tekstu</h2>
        <div class="section">
            <table>
                <tr><th>Parametr</th><th>Wartość</th></tr>
                <tr><td>Liczba słów</td><td>{stats.get('word_count', '—')}</td></tr>
                <tr><td>Liczba zdań</td><td>{stats.get('sentence_count', '—')}</td></tr>
                <tr><td>Liczba znaków</td><td>{stats.get('char_count', '—')}</td></tr>
                <tr><td>Unikalne słowa</td><td>{stats.get('unique_words', '—')}</td></tr>
                <tr><td>Bogactwo słownictwa</td><td>{stats.get('vocabulary_richness', '—')}%</td></tr>
                <tr><td>Czytelność</td><td>{stats.get('readability', '—')}</td></tr>
                <tr><td>Poziom trudności</td><td>{stats.get('difficulty', '—')}</td></tr>
            </table>
        </div>
        """

    if topics and topics.get('categories'):
        topic_rows = ''
        for cat_name, cat_data in topics['categories'].items():
            keywords_str = ', '.join(cat_data.get('keywords', []))
            topic_rows += f'<tr><td>{cat_name.capitalize()}</td><td>{keywords_str}</td><td>{cat_data.get("relevance", 0)}%</td></tr>'
        html += f"""
        <h2>🏷️ Wykryte tematy</h2>
        <div class="section">
            <table>
                <tr><th>Kategoria</th><th>Słowa kluczowe</th><th>Trafność</th></tr>
                {topic_rows}
            </table>
        </div>
        """

    if recommendations:
        recs_html = ''
        for rec in recommendations:
            recs_html += f'<div class="recommendation">{rec}</div>'
        html += f"""
        <h2>💡 Rekomendacje</h2>
        <div class="section">
            {recs_html}
        </div>
        """

    html += """
        <div class="footer">
            <p>Raport wygenerowany przez Analizator Tekstu — Praca kursowa</p>
            <p>Aby zapisać jako PDF: Ctrl+P → "Zapisz jako PDF"</p>
        </div>
    </body>
    </html>
    """

    return Response(
        html,
        mimetype='text/html',
        headers={'Content-Disposition': 'inline; filename=raport_analizy.html'}
    )

if __name__ == '__main__':
    print("=" * 50)
    print("  Text Analyzer is running!")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)