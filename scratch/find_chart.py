with open(r'c:\Users\syrot\OneDrive\Desktop\вова\TextScope\templates\index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'renderSentencesChart' in line or 'sentencesChart' in line or 'analyze_sentences' in line or 'analyze_batch' in line or 'csv' in line:
        print(f"Line {i+1}: {line.strip()[:100]}")
