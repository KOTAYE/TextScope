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
    check_grammar,
    analyze_text_with_llm
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
        stats = get_text_stats(text)
        topics = extract_topics(text)
        language = detect_language(text)
        
        # Call Google Gemini API if configured
        llm_analysis = analyze_text_with_llm(text)
        
        # Initialize emotions using local keyword analysis
        emotions = detect_emotions(text)
        
        # If Gemini AI returned valid analysis, override local sentiment, emotions, topics, and recommendations!
        if llm_analysis:
            # 1. Overriding emotions
            if isinstance(llm_analysis.get('emotions'), dict):
                ai_emotions = llm_analysis['emotions']
                ai_emotions_lower = {str(k).lower().strip(): v for k, v in ai_emotions.items()}
                required_keys = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
                if all(k in ai_emotions_lower for k in required_keys):
                    emoji_map = {
                        'joy': '😄', 'anger': '😡', 'sadness': '😢',
                        'fear': '😨', 'surprise': '😲', 'disgust': '🤢'
                    }
                    name_map = {
                        'joy': 'Radość', 'anger': 'Złość', 'sadness': 'Smutek',
                        'fear': 'Strach', 'surprise': 'Zaskoczenie', 'disgust': 'Obrzydzenie'
                    }
                    try:
                        scores = {k: round(float(ai_emotions_lower[k]), 1) for k in required_keys}
                        dominant = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else None
                        emotions = {
                            'scores': scores,
                            'dominant': dominant,
                            'dominant_name': name_map.get(dominant, '—'),
                            'dominant_emoji': emoji_map.get(dominant, '😐')
                        }
                    except (ValueError, TypeError):
                        pass

            # 2. Overriding sentiment
            ai_score = llm_analysis.get('sentiment_score')
            ai_category = llm_analysis.get('category')
            ai_subjectivity = llm_analysis.get('subjectivity', 0.0)
            if ai_score is not None and ai_category is not None:
                try:
                    score = round(float(ai_score), 3)
                    pos_pct = round(max(0.0, score * 100.0), 1)
                    neg_pct = round(max(0.0, -score * 100.0), 1)
                    neu_pct = round(100.0 - pos_pct - neg_pct, 1)
                    
                    emoji_val = "😐"
                    if score >= 0.1:
                        emoji_val = "😊"
                    elif score <= -0.1:
                        emoji_val = "😢"
                        
                    sentiment = {
                        'score': score,
                        'combined_score': score,
                        'category': str(ai_category),
                        'emoji': emoji_val,
                        'positive_pct': pos_pct,
                        'negative_pct': neg_pct,
                        'neutral_pct': neu_pct,
                        'vader': {
                            'positive': pos_pct,
                            'negative': neg_pct,
                            'neutral': neu_pct,
                            'compound': score
                        },
                        'textblob': {
                            'polarity': score,
                            'subjectivity': round(float(ai_subjectivity), 1)
                        }
                    }
                except (ValueError, TypeError):
                    pass

            # 3. Overriding topics
            ai_topics = llm_analysis.get('topics')
            if isinstance(ai_topics, dict):
                try:
                    topics = {
                        'categories': {
                            k: {
                                'keywords': [str(x) for x in v.get('keywords', [])],
                                'relevance': round(float(v.get('relevance', 0.0)), 1)
                            }
                            for k, v in ai_topics.items()
                        }
                    }
                except (ValueError, TypeError, AttributeError):
                    pass

            # 4. Overriding recommendations
            ai_recs = llm_analysis.get('recommendations')
            if isinstance(ai_recs, list) and len(ai_recs) > 0:
                recommendations = [str(r) for r in ai_recs]
            else:
                recommendations = generate_sentiment_recommendations(sentiment, emotions)
        else:
            recommendations = generate_sentiment_recommendations(sentiment, emotions)

        return jsonify({
            'sentiment': sentiment,
            'words': words,
            'emotions': emotions,
            'stats': stats,
            'topics': topics,
            'language': language,
            'recommendations': recommendations,
            'llm_analysis': llm_analysis
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

@app.route('/generate_example', methods=['POST'])
def generate_example_route() -> Response:
    """
    Generates a realistic text example dynamically via Google Gemini API
    based on requested language and category type.
    """
    data = request.get_json() or {}
    lang = str(data.get('language', 'pl')).strip().lower()
    example_type = str(data.get('type', 'pos')).strip().lower()

    # Pre-defined prompts for Gemini
    prompts_map = {
        'pl': {
            'pos': "Wygeneruj realistyczną, naturalną, krótką (2-4 zdania) opinię/recenzję po polsku, która jest bardzo pozytywna (np. o restauracji, hotelu lub produkcie). Zwróć tylko wygenerowany tekst i nic więcej.",
            'neg': "Wygeneruj realistyczną, naturalną, krótką (2-4 zdania) opinię/recenzję po polsku, która jest bardzo negatywna, rozczarowana i krytyczna. Zwróć tylko wygenerowany tekst i nic więcej.",
            'mixed': "Wygeneruj krótką (2-4 zdania) opinię po polsku, która ma charakter mieszany (częściowo chwali, a częściowo krytykuje produkt lub usługę). Zwróć tylko wygenerowany tekst i nic więcej.",
            'neutral': "Wygeneruj w pełni neutralny, krótki (2-4 zdania) tekst informacyjny po polsku (np. godziny otwarcia biura, fakty naukowe, opis lokalizacji). Zwróć tylko wygenerowany tekst i nic więcej.",
            'ai1': "Wygeneruj krótki (3-5 zdań) tekst o zaawansowanej technologii lub nauce po polsku, napisany w sposób bardzo formalny, bezosobowy, z użyciem słów takich jak 'ponadto', 'warto zauważyć', 'podsumowując' - typowy dla generowania przez ChatGPT. Zwróć tylko wygenerowany tekst i nic więcej.",
            'ai2': "Wygeneruj krótki (3-5 zdań) tekst naukowy lub biznesowy po polsku, o wysokim stopniu formalności i monotonnej strukturze zdań typowej dla modeli językowych AI. Zwróć tylko wygenerowany tekst i nic więcej.",
            'human1': "Wygeneruj krótki (2-4 zdania) komentarz po polsku napisany przez prawdziwego człowieka w sposób bardzo nieformalny, potoczny, ze skrótami (np. lol, w sumie, spoko, nwm) i brakiem wielkich liter. Zwróć tylko wygenerowany tekst.",
            'human2': "Wygeneruj krótki (2-4 zdania) emocjonalny wpis na bloga lub social media po polsku, z typowo ludzkim, ekspresyjnym stylem pisania, slangiem i wykrzyknikami. Zwróć tylko wygenerowany tekst."
        },
        'en': {
            'pos': "Generate a realistic, natural, short (2-4 sentences) online review in English that is extremely positive (e.g. about a restaurant, hotel or product). Return only the generated text.",
            'neg': "Generate a realistic, natural, short (2-4 sentences) online review in English that is extremely negative, disappointed and critical. Return only the generated text.",
            'mixed': "Generate a short (2-4 sentences) review in English that has mixed feelings (partially praises and partially criticizes). Return only the generated text.",
            'neutral': "Generate a completely neutral, short (2-4 sentences) factual text in English (e.g. business hours, scientific facts, address description). Return only the generated text.",
            'ai1': "Generate a short (3-5 sentences) text about technology in English, written in a very formal, passive voice, utilizing connectors like 'furthermore', 'it should be noted', 'consequently' - typical of ChatGPT style. Return only the generated text.",
            'ai2': "Generate a short (3-5 sentences) business/scientific paragraph in English with highly uniform sentence lengths typical of AI assistants. Return only the generated text.",
            'human1': "Generate a short (2-4 sentences) comment in English written by a real human in a very informal, colloquial style, with abbreviations like lol, tbh, ngl, vibes, and casual punctuation. Return only the generated text.",
            'human2': "Generate a short (2-4 sentences) expressive social media post in English with typical human emotion, exclamation marks, slang and casual tone. Return only the generated text."
        }
    }

    # Default fallback examples (in case Gemini API is not active or fails)
    fallbacks = {
        'pl': {
            'pos': "To miejsce jest absolutnie niesamowite! Jedzenie było pyszne, a obsługa niezwykle miła i pomocna. Na pewno wrócimy tu za tydzień. Bardzo polecam!",
            'neg': "Tragedia, najgorsze miejsce w jakim byłem. Obsługa była niemiła, jedzenie zimne i niedobre, a czystość pozostawiała wiele do życzenia. Omijajcie szerokim łukiem!",
            'mixed': "Jedzenie było całkiem smaczne i ładnie podane, ale czas oczekiwania wyniósł ponad godzinę. Obsługa wydawała się znudzona. Ceny są w porządku, ale mogło być lepiej.",
            'neutral': "Restauracja znajduje się przy ulicy Głównej 12, obok poczty. Menu zawiera dania kuchni włoskiej. Czynne od poniedziałku do soboty od 11:00 do 22:00.",
            'ai1': "Sztuczna inteligencja stanowi jeden z najbardziej kluczowych kierunków rozwoju nowoczesnej inżynierii oprogramowania. Ponadto, wdrożenie tych metod pozwala na znaczące usprawnienie procesów biznesowych. Warto zauważyć, że algorytmy uczenia maszynowego wykazują wysoką skuteczność. Podsumowując, technologia ta redefiniuje współczesne standardy.",
            'ai2': "Zrównoważone źródła energii elektrycznej stanowią fundament współczesnej walki ze zmianami klimatycznymi. Dodatkowo, nowoczesne turbiny wiatrowe charakteryzują się optymalną sprawnością. Należy podkreślić, że inwestycje w infrastrukturę ekologiczną przynoszą wymierne korzyści ekonomiczne.",
            'human1': "byłem wczoraj w tej nowej kawiarni i szczerze?? bez rewelacji lol. kawa jak kawa, a czekałem chyba ze 20 minut co mnie mega wkurzyło ngl. wnętrze ładne i spoko muzyka grała ale wifi było tak wolne, że nie dało się pracować...",
            'human2': "właśnie skończyłam oglądać ten film i O MÓJ BOŻE!!! nie wiem jak zacząć, początek nudnawy ale ta końcówka to jakiś kosmos totalny!! nie spodziewałam się takiego obrotu spraw!! gra aktorska super, poryczałam się chyba z trzy razy :("
        },
        'en': {
            'pos': "This restaurant is absolutely amazing! The food was incredibly delicious, and the staff was so friendly and welcoming. Highly recommend to everyone!",
            'neg': "Terrible experience, worst restaurant ever. The food was cold and tasteless, and the waiter was extremely rude. Never coming back, save your money!",
            'mixed': "The food was quite good and presentation was nice, but the service was slow. Prices are reasonable, so I might come back for a quick lunch.",
            'neutral': "The restaurant is located at 42 Main Street. They serve Italian and Mediterranean cuisine. Opening hours are Monday to Saturday, 11am to 10pm.",
            'ai1': "Artificial intelligence has emerged as a transformative technology, reshaping various sectors. Furthermore, the integration of machine learning algorithms enables organizations to optimize efficiency. It is important to note that these systems require careful oversight to maintain accuracy.",
            'ai2': "The implementation of sustainable energy systems represents a critical imperative for global development. Wind power technologies have experienced significant advancements, resulting in lower costs. Consequently, demand for renewable options continues to rise.",
            'human1': "ok so i went to this new place and honestly?? it was kinda mid lol. latte was fine i guess but nothing special... waited like 15 mins which was annoying. vibes were cool tho ngl.",
            'human2': "just finished watching that movie and WOW. ok where do i even start. first hour was slow tbh but then it picks up and holy crap the plot twist!! did NOT see that coming at all!!"
        }
    }

    # Fetch prompt
    lang_prompts = prompts_map.get(lang, prompts_map['pl'])
    prompt = lang_prompts.get(example_type, lang_prompts['pos'])

    # Try Gemini
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=8)
            if response.status_code == 200:
                res_data = response.json()
                candidates = res_data.get("candidates", [])
                if candidates:
                    text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    if text_content:
                        import urllib.parse
                        # Clean potential wrapping quotes
                        if text_content.startswith('"') and text_content.endswith('"'):
                            text_content = text_content[1:-1]
                        return jsonify({'text': text_content, 'source': 'gemini_ai'})
        except Exception:
            pass

    # Fallback to local examples
    lang_fallbacks = fallbacks.get(lang, fallbacks['pl'])
    text_fallback = lang_fallbacks.get(example_type, lang_fallbacks['pos'])
    return jsonify({'text': text_fallback, 'source': 'local_fallback'})

if __name__ == '__main__':
    print("=" * 50)
    print("  TextScope is running!")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)