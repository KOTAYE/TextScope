# -*- coding: utf-8 -*-
"""
Configuration module for the TextScope application.
Contains environment loading, Flask application setup, and NLTK resource downloading.
"""

import os
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Force UTF-8 encoding for Windows compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Download NLTK dependencies required for tokenization and VADER analysis
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Global configuration variables
SECRET_KEY: str = os.getenv("SECRET_KEY", "textscope-super-secret-key-2026")

# Global analyzer instance
sid: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
