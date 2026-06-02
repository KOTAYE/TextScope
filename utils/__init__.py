# -*- coding: utf-8 -*-
"""
Utilities package initialization.
Exposes recommendation and styling advisors.
"""

from .recommendations import generate_sentiment_recommendations, generate_ai_recommendations

__all__ = [
    'generate_sentiment_recommendations',
    'generate_ai_recommendations'
]
