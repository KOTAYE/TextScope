# TextScope API Documentation

Opis wszystkich dostępnych punktów końcowych (endpoints) w aplikacji TextScope. Wszystkie zapytania `POST` przyjmują i zwracają dane w formacie JSON (z wyjątkiem specyficznych przypadków eksportu i przesyłania plików).

---

## 1. Strona główna

### `GET /`
Renderuje i zwraca interfejs użytkownika (kokpit analizatora NLP).

- **Odpowiedź:** `text/html` (szablon kokpitu).

---

## 2. Jednoczęściowa analiza tekstu

### `POST /analyze`
Wykonuje analizę sentymentu lub wykrywanie tekstu wygenerowanego przez AI dla pojedynczego bloku tekstu.

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania (Request Body):**
```json
{
  "text": "Wspaniały hotel! Bardzo polecam.",
  "type": "sentiment" 
}
```
*Gdzie `type` może być `"sentiment"` (sentyment, emocje, tematy, statystyki) lub `"ai_detect"` (prawdopodobieństwo wygenerowania przez sztuczną inteligencję).*

- **Odpowiedź (dla `type: "sentiment"`):**
```json
{
  "language": {
    "code": "pl",
    "name": "Polski",
    "confidence": 95
  },
  "sentiment": {
    "category": "Pozytywny",
    "emoji": "😊",
    "combined_score": 0.815,
    "positive_pct": 90.8,
    "negative_pct": 9.2,
    "vader": {
      "positive": 15.2,
      "negative": 0.0,
      "neutral": 84.8,
      "compound": 0.5
    },
    "textblob": {
      "polarity": 0.1,
      "subjectivity": 40.0
    }
  },
  "words": [
    { "text": "Wspaniały", "score": 0.8, "type": "positive" },
    { "text": " ", "score": 0, "type": "space" },
    { "text": "hotel", "score": 0, "type": "neutral" }
  ],
  "emotions": {
    "scores": {
      "joy": 85.0,
      "anger": 0.0,
      "sadness": 0.0,
      "fear": 0.0,
      "surprise": 0.0,
      "disgust": 0.0
    },
    "dominant": "joy",
    "dominant_name": "Radość",
    "dominant_emoji": "😄"
  },
  "stats": {
    "word_count": 4,
    "sentence_count": 2,
    "char_count": 31,
    "char_no_spaces": 28,
    "avg_word_len": 6.2,
    "avg_sentence_len": 2.0,
    "unique_words": 4,
    "vocabulary_richness": 100.0,
    "readability": 85.0,
    "difficulty": "Łatwy",
    "difficulty_color": "#4ade80",
    "top_words": []
  },
  "topics": {
    "categories": {
      "jakość": {
        "keywords": ["dobry", "świetny"],
        "count": 1,
        "relevance": 25.0
      }
    },
    "top_keywords": [
      { "word": "wspaniały", "count": 1 }
    ],
    "total_topics_found": 1
  },
  "recommendations": [
    "✅ Tekst ma wyraźnie pozytywny wydźwięk. Dobrze nadaje się jako pozytywna recenzja."
  ]
}
```

---

## 3. Analiza sentymentu zdań

### `POST /analyze_sentences`
Rozbija tekst na osobne zdania i ocenia sentyment każdego z nich oddzielnie. Przydatne do wykresów liniowych zmiany nastroju.

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania:**
```json
{
  "text": "Wspaniałe śniadanie. Jednak obsługa była bardzo wolna."
}
```
- **Odpowiedź:**
```json
{
  "sentences": [
    {
      "sentence": "Wspaniałe śniadanie.",
      "sentiment_score": 0.81,
      "category": "Pozytywny"
    },
    {
      "sentence": "Jednak obsługa była bardzo wolna.",
      "sentiment_score": -0.65,
      "category": "Negatywny"
    }
  ]
}
```

---

## 4. Streszczanie tekstu (LSA)

### `POST /summarize`
Generuje trójzdaniowe streszczenie długich artykułów przy użyciu dekompozycji wartości osobliwych (Latent Semantic Analysis).

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania:**
```json
{
  "text": "[Długi wielozdaniowy artykuł...]"
}
```
- **Odpowiedź:**
```json
{
  "language": {
    "code": "pl",
    "name": "Polski",
    "confidence": 95
  },
  "summary": [
    "Zdanie kluczowe numer jeden.",
    "Drugie najważniejsze zdanie artykułu.",
    "Zdanie podsumowujące temat."
  ]
}
```

---

## 5. Sprawdzanie pisowni i gramatyki

### `POST /grammar_check`
Weryfikuje poprawność ortograficzną, interpunkcyjną i stylistyczną za pomocą LanguageTool.

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania:**
```json
{
  "text": "To jest zły tekst z blędem ortograficznym."
}
```
- **Odpowiedź:**
```json
{
  "errors": [
    {
      "offset": 20,
      "length": 7,
      "message": "Prawdopodobny błąd ortograficzny. Czy miałeś na myśli 'błędem'?",
      "ruleId": "HUNSPELL_RULE",
      "bad": "blędem",
      "suggestions": ["błędem", "pędem", "lędem"]
    }
  ]
}
```

---

## 6. Porównanie tekstów

### `POST /compare`
Porównuje dwa teksty obok siebie i wylicza różnicę we wskaźnikach stylometrycznych i nastroju.

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania:**
```json
{
  "text1": "To jest cudowny hotel.",
  "text2": "Okropna obsługa i drogo."
}
```
- **Odpowiedź:**
```json
{
  "text1": { "sentiment": { "combined_score": 0.8 }, "stats": { "word_count": 4 } },
  "text2": { "sentiment": { "combined_score": -0.7 }, "stats": { "word_count": 4 } },
  "comparison": {
    "sentiment_diff": 1.5,
    "ai_diff": 0.0,
    "word_count_diff": 0,
    "same_language": true
  }
}
```

---

## 7. Analiza wsadowa i grupowanie (Clustering)

### `POST /analyze_batch`
Przetwarza plik CSV z opiniami, wylicza statystyki zbiorcze oraz grupuje semantycznie zbliżone recenzje przy użyciu TF-IDF i Cosine Similarity.

- **Nagłówki:** `Content-Type: multipart/form-data`
- **Plik (FormData):** Klucz `file` z plikiem `.csv`. Pierwsza kolumna powinna zawierać treść recenzji.
- **Odpowiedź:**
```json
{
  "summary": {
    "total_reviews": 5,
    "average_sentiment": 0.25,
    "distribution": {
      "positive": 3,
      "negative": 1,
      "neutral": 1
    },
    "average_emotions": {
      "joy": 45.0,
      "anger": 12.0
    },
    "similarity_clusters": [
      {
        "cluster_id": 1,
        "review_indices": [0, 3] 
      }
    ]
  },
  "results": [
    {
      "row": 1,
      "text": "Bardzo szybka dostawa, polecam...",
      "dominant_emotion": "Radość",
      "sentiment": { "combined_score": 0.9 },
      "language": { "code": "pl" }
    }
  ]
}
```

---

## 8. Generowanie Raportu HTML do druku

### `POST /export_report`
Generuje i zwraca w pełni sformatowany, profesjonalny dokument HTML gotowy do wydrukowania lub zapisu jako PDF.

- **Nagłówki:** `Content-Type: application/json`
- **Ciało zapytania:** Obiekt JSON z polami `text`, `sentiment`, `emotions`, `stats`, `ai`, `topics`, `language` i `recommendations`.
- **Odpowiedź:** `text/html` o wysokiej jakości estetycznej z instrukcją druku do pliku PDF (`Content-Disposition: inline; filename=raport_analizy.html`).
