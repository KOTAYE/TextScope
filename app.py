# -*- coding: utf-8 -*-
"""
Main Flask Application for TextScope.
Defines HTTP routes and handles client requests by delegating to specialized NLP analyzer modules.
"""

import csv
import io
import os
import sys
from typing import Dict, Union, Any, List

from flask import Flask, render_template, request, jsonify, Response

# Ensure absolute paths for modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SECRET_KEY
from analyzers import (
    detect_language,
    analyze_sentiment,
    get_word_sentiments,
    detect_emotions,
    get_text_stats,
    extract_topics,
    detect_ai_text,
    analyze_sentences_sentiment,
    summarize_text,
    cluster_reviews,
    check_grammar
)
from utils import (
    generate_sentiment_recommendations,
    generate_ai_recommendations
)

app = Flask(__name__)
app.secret_key = SECRET_KEY

@app.route('/')
def index() -> str:
    """
    Renders the main dashboard page.
    
    Returns:
        str: HTML template text.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze() -> Response:
    """
    Analyzes sentiment or AI probability for a single input text.
    
    Returns:
        Response: Flask JSON response.
    """
    data: Dict[str, Any] = request.get_json() or {}
    text: str = data.get('text', '').strip()
    analysis_type: str = data.get('type', 'sentiment')

    if not text:
        return jsonify({'error': 'Wprowadź tekst do analizy'}), 400

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

    return jsonify({'error': 'Nieznany typ analizy'}), 400

@app.route('/analyze_sentences', methods=['POST'])
def analyze_sentences() -> Response:
    """
    Analyses sentiment polarity on a sentence-by-sentence basis.
    
    Returns:
        Response: Flask JSON response list of sentences with scores.
    """
    data: Dict[str, Any] = request.get_json() or {}
    text: str = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Wprowadź tekst do analizy'}), 400

    results = analyze_sentences_sentiment(text)
    return jsonify({'sentences': results})

@app.route('/summarize', methods=['POST'])
def summarize() -> Response:
    """
    Generates a 3-sentence summary of the input text using LSA.
    
    Returns:
        Response: Flask JSON response containing the list of sentences.
    """
    data: Dict[str, Any] = request.get_json() or {}
    text: str = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Wprowadź tekst do podsumowania'}), 400

    lang_info = detect_language(text)
    lang_code = str(lang_info['code'])

    summary_sentences = summarize_text(text, sentence_count=3, language=lang_code)
    return jsonify({
        'summary': summary_sentences,
        'language': lang_info
    })

@app.route('/grammar_check', methods=['POST'])
def grammar_check() -> Response:
    """
    Spells and grammar checks the input text.
    
    Returns:
        Response: Flask JSON response containing lists of errors.
    """
    data: Dict[str, Any] = request.get_json() or {}
    text: str = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Wprowadź tekst do weryfikacji'}), 400

    grammar_errors = check_grammar(text)
    return jsonify({'errors': grammar_errors})

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch() -> Response:
    """
    Processes multiple reviews uploaded as a CSV file and clusters them.
    
    Returns:
        Response: Flask JSON response summary and Member stats.
    """
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
        next(reader, None)  # Skip header row

        results = []
        raw_texts = []
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
            raw_texts.append(review_text)

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
                emotion_totals[emo] = emotion_totals.get(emo, 0.0) + score

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
        avg_score = round(sum(all_scores) / total, 3) if all_scores else 0.0

        avg_emotions = {}
        for emo, total_score in emotion_totals.items():
            avg_emotions[emo] = round(total_score / total, 1)

        # Run similarity clustering using scikit-learn
        similarity_clusters = cluster_reviews(raw_texts, threshold=0.7)

        summary = {
            'total_reviews': total,
            'average_sentiment': avg_score,
            'distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'average_emotions': avg_emotions,
            'similarity_clusters': similarity_clusters
        }

        return jsonify({
            'summary': summary,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': f'Błąd podczas przetwarzania pliku: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_texts() -> Response:
    """
    Compares two input texts side-by-side.
    
    Returns:
        Response: Flask JSON response.
    """
    data: Dict[str, Any] = request.get_json() or {}
    text1: str = data.get('text1', '').strip()
    text2: str = data.get('text2', '').strip()

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
def export_report() -> Response:
    """
    Generates a printable, styled HTML report of single text analysis.
    
    Returns:
        Response: Styled printable HTML document.
    """
    data: Dict[str, Any] = request.get_json() or {}
    if not data:
        return jsonify({'error': 'Brak danych do wygenerowania raportu'}), 400

    text: str = data.get('text', 'Brak tekstu')
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
            topic_rows += f'<tr><td>{cat_name.capitalize()}</td><td>{keywords_str}</td><td>{cat_data.get("relevance", 0.0)}%</td></tr>'
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
    print("  TextScope is running!")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)