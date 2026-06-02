# -*- coding: utf-8 -*-
"""
Polish sentiment lexicon stems.
Contains lists of positive and negative stems for Polish text analysis.
"""

from typing import Dict, List

POLISH_LEXICON: Dict[str, List[str]] = {
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
