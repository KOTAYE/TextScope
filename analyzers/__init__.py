# -*- coding: utf-8 -*-
"""
Analyzers package initialization.
Exposes modular text analysis components: language detection, sentiment, stats, emotions, topics, and AI probability.
"""

from .language import detect_language
from .sentiment import analyze_sentiment, get_word_sentiments, analyze_multilingual_sentiment, analyze_sentences_sentiment
from .emotions import detect_emotions
from .stats import get_text_stats, split_into_sentences
from .topics import extract_topics
from .ai_detection import detect_ai_text
from .summarizer import summarize_text
from .clustering import cluster_reviews
from .grammar import check_grammar
from .llm_analysis import analyze_text_with_llm

__all__ = [
    'analyze_text_with_llm',
    'detect_language',
    'analyze_sentiment',
    'get_word_sentiments',
    'analyze_multilingual_sentiment',
    'analyze_sentences_sentiment',
    'detect_emotions',
    'get_text_stats',
    'split_into_sentences',
    'extract_topics',
    'detect_ai_text',
    'summarize_text',
    'cluster_reviews',
    'check_grammar'
]
