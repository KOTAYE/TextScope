# -*- coding: utf-8 -*-
"""
Sentiment analysis module.
Handles text polarity calculations, multilingual lexicon-based analysis, and word-by-word sentiment coloring.
"""

import re
from typing import Dict, List, Union, Optional
from textblob import TextBlob
from config import sid
from lexicons import POLISH_LEXICON, UKRAINIAN_LEXICON
from analyzers.language import detect_language

def normalize_text_for_lexicon(text: str) -> str:
    """
    Normalizes Polish/Ukrainian text accents and converts to lowercase for exact stem matching.
    
    Args:
        text (str): The input text to normalize.
        
    Returns:
        str: Normalized lowercase text.
    """
    text = text.lower()
    accents = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z'
    }
    for char, repl in accents.items():
        text = text.replace(char, repl)
    return text

def analyze_multilingual_sentiment(text: str, lang_code: str) -> tuple[float, int, int]:
    """
    Performs dictionary-based sentiment scoring on Polish or Ukrainian texts.
    
    Args:
        text (str): The input text.
        lang_code (str): 'pl' for Polish, 'uk' for Ukrainian.
        
    Returns:
        tuple[float, int, int]: Score (between -1.0 and 1.0), count of positive stems, count of negative stems.
    """
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

def analyze_sentiment(text: str, lang_code: Optional[str] = None) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Calculates combined sentiment score blending custom lexicons, TextBlob, and VADER.
    
    Args:
        text (str): The text to analyze.
        lang_code (Optional[str]): Pre-detected language code.
        
    Returns:
        Dict[str, Union[str, float, Dict[str, float]]]: Combined analysis metrics including categories and polarity counts.
    """
    if not lang_code:
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

def get_word_sentiments(text: str, lang_code: Optional[str] = None) -> List[Dict[str, Union[str, float]]]:
    """
    Segments the text into words and labels each word with its sentiment score and type.
    
    Args:
        text (str): The input text.
        lang_code (Optional[str]): Language code of the text. If None, it will be auto-detected.
        
    Returns:
        List[Dict[str, Union[str, float]]]: List of dictionaries containing word text, score, and sentiment type.
    """
    if not lang_code:
        lang_code = str(detect_language(text)['code'])
        
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

def analyze_sentences_sentiment(text: str) -> List[Dict[str, Union[str, float]]]:
    """
    Splits text into sentences and computes sentiment score and category for each.
    
    Args:
        text (str): The input text.
        
    Returns:
        List[Dict[str, Union[str, float]]]: List of sentences with their scores and categories.
    """
    from analyzers.stats import split_into_sentences
    parent_lang = str(detect_language(text)['code'])
    sentences = split_into_sentences(text)
    results = []
    for s in sentences:
        s_sentiment = analyze_sentiment(s, lang_code=parent_lang)
        results.append({
            'sentence': s,
            'sentiment_score': s_sentiment['combined_score'],
            'category': s_sentiment['category']
        })
    return results
