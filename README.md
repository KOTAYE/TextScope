# 📊 TextScope — Zaawansowany Analizator NLP i Stylometrii Tekstu

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-orange.svg)](https://www.nltk.org/)
[![LanguageTool](https://img.shields.io/badge/LanguageTool-2.7.1-red.svg)](https://languagetool.org/)

**TextScope** to nowoczesna aplikacja webowa oparta na Flasku, stworzona do wieloaspektowej analizy przetwarzania języka naturalnego (NLP) oraz oceny stylometrii tekstu. Projekt powstał z myślą o weryfikacji opinii, recenzji i artykułów pod kątem emocjonalnym, statystycznym oraz potencjalnego autorstwa maszynowego (AI).

Interfejs użytkownika jest w pełni dostosowany do **języka polskiego**, a zaawansowany hybrydowy analizator sentymentu łączy dedykowane słowniki leksykalne (dla języka polskiego i ukraińskiego) z bibliotekami VADER oraz TextBlob.

---

## ✨ Główne Funkcje

1. **Wielojęzyczna Analiza Sentymentu (PL, UK, EN):**
   * Hybrydowy mechanizm wyliczający wynik (od -1.0 do 1.0) z wagą leksykalną (80%), VADER (5%) i TextBlob (15%).
   * Kolorowanie wyrazów w czasie rzeczywistym w polu edycyjnym (zielony = pozytywne, czerwony = negatywne).
   * Wykres liniowy rozkładu nastroju zdanie po zdaniu (Chart.js).

2. **Weryfikacja Autorstwa AI (AI Detection):**
   * Autorski algorytm stylometryczny analizujący: jednorodność długości zdań, bogactwo słownictwa (TTR), formalność stylu, powtarzalność wzorców, gęstość konektorów logicznych oraz statystyki interpunkcyjne.

3. **Ocena Emocji i Tematów:**
   * Wykrywanie 6 podstawowych emocji (Radość, Złość, Smutek, Strach, Zaskoczenie, Obrzydzenie) wraz z wizualizacją.
   * Ekstrakcja tematów (np. Jedzenie, Obsługa, Cena, Atmosfera, Jakość, Dostawa) z przypisaniem trafności procentowej.

4. **Streszczanie i Korekta Gramatyczna:**
   * Algorytm **LSA (Latent Semantic Analysis)** redukujący długie teksty do esencji 3 kluczowych zdań.
   * Integracja z **LanguageTool** do korekty ortograficznej i gramatycznej bezpośrednio w przeglądarce.

5. **Porównanie Tekstów (Side-by-Side):**
   * Interaktywne zestawienie dwóch tekstów z wizualnymi wskaźnikami delta (strzałki różnic) ułatwiającymi porównanie dwóch wariantów opinii.

6. **Analiza Zbiorcza i Grupowanie (Clustering):**
   * Wgrywanie plików CSV z wieloma recenzjami.
   * Grupowanie podobnych opinii za pomocą **TF-IDF** oraz **Cosine Similarity** (próg podobieństwa 0.7) z automatyczną segmentacją.

7. **Eksport Wyników:**
   * Zapisywanie raportów do formatów **JSON**, surowego pliku HTML lub drukowanego/zapisanego jako PDF ze zoptymalizowanymi stylami CSS print.

---

## 🛠️ Architektura Projektu

Projekt został zrefaktoryzowany zgodnie z dobrymi praktykami PEP8 i podziałem na modularne komponenty:

```
TextScope/
├── app.py                  # Flask endpoints i routing HTTP
├── config.py               # Konfiguracja środowiska, NLTK i klucze
├── Dockerfile              # Konstrukcja obrazu Docker (z JRE dla LanguageTool)
├── docker-compose.yml      # Konfiguracja Compose do mapowania logów
├── requirements.txt        # Zablokowane wersje bibliotek
├── .env.example            # Szablon zmiennych środowiskowych
├── analyzers/              # Rdzeń analityczny NLP
│   ├── __init__.py
│   ├── ai_detection.py     # Detektor autorstwa AI i stylometria
│   ├── clustering.py       # Grupowanie TF-IDF i Cosine Similarity
│   ├── emotions.py         # Klasyfikator emocji na słowach kluczowych
│   ├── grammar.py          # Adapter LanguageTool (Java JRE)
│   ├── language.py         # Autodetekcja języka (PL, UK, EN, Unknown)
│   ├── sentiment.py        # Hybrydowa analiza sentymentu i kolorowanie słów
│   ├── stats.py            # Statystyki tekstu i podział zdań
│   ├── summarizer.py       # Streszczenia LSA (sumy)
│   └── topics.py           # Klasyfikator tematów i wyrazów kluczowych
├── lexicons/               # Słowniki lingwistyczne (brak słów RU!)
│   ├── __init__.py
│   ├── polish.py           # Rdzenie wyrazów pozytywnych i negatywnych PL
│   └── ukrainian.py        # Rdzenie wyrazów pozytywnych i negatywnych UK
├── utils/                  # Funkcje pomocnicze
│   ├── __init__.py
│   └── recommendations.py  # System rekomendacji w języku polskim
├── templates/              # Interfejs użytkownika
│   └── index.html          # Dynamiczny panel Glassmorphism
├── tests/                  # Testy automatyczne (pytest)
│   ├── __init__.py
│   ├── test_ai_detection.py
│   ├── test_language.py
│   ├── test_routes.py
│   ├── test_sentiment.py
│   └── test_stats.py
└── docs/
    └── API.md              # Pełna dokumentacja endpointów API
```

---

## 🚀 Uruchomienie Aplikacji

### Wymagania wstępne
* **Python 3.10+** lub zainstalowany **Docker**.
* **Java JRE** (wymagana lokalnie do uruchomienia silnika LanguageTool).

---

### Metoda 1: Uruchomienie Lokalne (Python)

1. **Sklonuj repozytorium i przejdź do folderu:**
   ```bash
   cd TextScope
   ```

2. **Utwórz wirtualne środowisko i je aktywuj:**
   ```bash
   python -m venv venv
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Zainstaluj wymagane zależności:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Przygotuj plik konfiguracyjny `.env`:**
   ```bash
   cp .env.example .env
   ```

5. **Uruchom aplikację:**
   ```bash
   python app.py
   ```
   Aplikacja będzie dostępna pod adresem: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Metoda 2: Uruchomienie w kontenerze Docker

Dzięki Dockerowi nie musisz instalować JRE ani Pythona lokalnie. Wszystkie zasoby zostaną pobrane i skonfigurowane wewnątrz kontenera.

1. **Zbuduj i uruchom kontenery za pomocą Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Dostęp do aplikacji:**
   Otwórz przeglądarkę i przejdź pod adres [http://localhost:5000](http://localhost:5000).

3. **Wyłączenie kontenerów:**
   ```bash
   docker-compose down
   ```

---

## 🧪 Testy Automatyczne

Do weryfikacji kodu przygotowany został kompletny zestaw testów `pytest`:

```bash
# Uruchomienie testów lokalnych
pytest -v
```

Zintegrowany plik workflow `.github/workflows/ci.yml` automatycznie uruchamia testy w GitHub Actions po każdym commicie w środowisku z zainstalowanym Java SDK.

---

## 📄 Licencja

Projekt dystrybuowany na licencji MIT. Szczegóły znajdziesz w pliku [LICENSE](LICENSE).
