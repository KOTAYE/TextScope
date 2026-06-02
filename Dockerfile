# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies, including JRE (required for language_tool_python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources and trigger language_tool download in build phase to speed up runtime container start
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('vader_lexicon', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
RUN python -c "import language_tool_python; language_tool_python.LanguageTool('en-US')"

# Copy the rest of the application
COPY . .

# Expose the Flask/Gunicorn port
EXPOSE 5000

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2", "--timeout", "120"]
