with open(r'c:\Users\syrot\OneDrive\Desktop\вова\TextScope\templates\index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'showBatchResults' in line:
        try:
            print(f"Line {i+1}: {line.strip()[:100]}")
        except Exception:
            pass
