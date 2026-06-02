# -*- coding: utf-8 -*-
"""
HTTP route tests for the TextScope Flask application.
"""

import io
import sys
import os
import pytest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b"TextScope" in res.data or b"textscope" in res.data.lower()

def test_analyze_sentiment_route(client):
    res = client.post('/analyze', json={
        'text': 'To jest fantastyczny hotel, polecam każdemu!',
        'type': 'sentiment'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'sentiment' in data
    assert 'words' in data
    assert 'emotions' in data
    assert 'stats' in data
    assert 'topics' in data
    assert 'language' in data
    assert 'recommendations' in data

def test_analyze_ai_detect_route(client):
    res = client.post('/analyze', json={
        'text': (
            "Furthermore, it should be noted that artificial intelligence has become a standard tool in modern times. "
            "Consequently, many students utilize these advanced methods to write their assignments."
        ),
        'type': 'ai_detect'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'ai' in data
    assert 'stats' in data
    assert 'language' in data
    assert 'recommendations' in data

def test_analyze_sentences_route(client):
    res = client.post('/analyze_sentences', json={
        'text': 'Wspaniały hotel. Okropne jedzenie.'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'sentences' in data
    assert len(data['sentences']) >= 2

def test_summarize_route(client):
    res = client.post('/summarize', json={
        'text': (
            "Kraków to jedno z najstarszych i najpiękniejszych miast w Polsce. "
            "Ma bogatą historię i mnóstwo niesamowitych zabytków. "
            "Każdego roku odwiedzają je miliony turystów z całego świata."
        )
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'summary' in data
    assert 'language' in data

def test_grammar_check_route(client):
    with patch('app.check_grammar') as mock_check:
        mock_check.return_value = [{'offset': 0, 'length': 4, 'message': 'Błąd', 'ruleId': 'TEST', 'bad': 'test', 'suggestions': []}]
        res = client.post('/grammar_check', json={
            'text': 'test'
        })
        assert res.status_code == 200
        data = res.get_json()
        assert 'errors' in data
        assert len(data['errors']) == 1

def test_compare_route(client):
    res = client.post('/compare', json={
        'text1': 'To jest świetny produkt.',
        'text2': 'To jest tragiczny produkt.'
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'text1' in data
    assert 'text2' in data
    assert 'comparison' in data
    assert data['comparison']['sentiment_diff'] != 0

def test_export_report_route(client):
    res = client.post('/export_report', json={
        'text': 'Świetny hotel.',
        'sentiment': {'category': 'Pozytywny', 'combined_score': 0.8, 'positive_pct': 90.0, 'negative_pct': 10.0},
        'emotions': {'dominant_name': 'Radość', 'dominant_emoji': '😄', 'scores': {'joy': 80.0}},
        'stats': {'word_count': 2, 'sentence_count': 1, 'char_count': 14},
        'ai': {'ai_probability': 10.0, 'verdict': 'Człowiek'},
        'topics': {'categories': {}},
        'language': {'name': 'Polski'},
        'recommendations': ['Super!']
    })
    assert res.status_code == 200
    assert b"Raport analizy tekstu" in res.data
    assert "pl" in res.headers.get('Content-Disposition', '') or b"raport_analizy.html" in res.data or "html" in res.headers.get('Content-Type', '')

def test_analyze_batch_route(client):
    csv_data = "recenzje\nTo jest świetna recenzja!\nTo jest okropna recenzja.\n"
    res = client.post('/analyze_batch', data={
        'file': (io.BytesIO(csv_data.encode('utf-8')), 'test.csv')
    })
    assert res.status_code == 200
    data = res.get_json()
    assert 'summary' in data
    assert 'results' in data
    assert data['summary']['total_reviews'] == 2
