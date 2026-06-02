# -*- coding: utf-8 -*-
"""
Text Summarization module.
Uses the sumy library and LsaSummarizer to extract key sentences from text.
"""

from typing import List
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def summarize_text(text: str, sentence_count: int = 3, language: str = "english") -> List[str]:
    """
    Extracts the key sentences from the text using LSA Summarization.
    
    Args:
        text (str): The input text to summarize.
        sentence_count (int): Number of sentences in the summary.
        language (str): Language for stop words and stemmer ('english', 'polish', etc.).
        
    Returns:
        List[str]: Extracted summary sentences.
    """
    # Mapping our language codes to sumy language names
    lang_map = {
        'en': 'english',
        'pl': 'polish',
        'uk': 'ukrainian'  # Fallbacks are handled gracefully by sumy
    }
    
    sumy_lang = lang_map.get(language, 'english')
    
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(sumy_lang))
        stemmer = Stemmer(sumy_lang)
        summarizer = LsaSummarizer(stemmer)
        try:
            summarizer.stop_words = get_stop_words(sumy_lang)
        except Exception:
            summarizer.stop_words = get_stop_words('english')
            
        summary = summarizer(parser.document, sentence_count)
        return [str(sentence) for sentence in summary]
    except Exception:
        # Fallback to English tokenizer/stemmer if the language resources are not available
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        stemmer = Stemmer('english')
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words('english')
        summary = summarizer(parser.document, sentence_count)
        return [str(sentence) for sentence in summary]
