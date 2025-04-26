import os
import json

lecture_name = 'lecture01-eval'

folder_path = f'devset/nmtclass/{lecture_name}'
slides = []

for folder in sorted(os.listdir(folder_path)):
    for filename in os.listdir(os.path.join(folder_path, folder)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                slides.append({
                    "folder": folder,
                    "content": text
                })

with open(f"{lecture_name}_slides.json", "w", encoding="utf-8") as f:
    json.dump(slides, f, ensure_ascii=False, indent=2)

with open(f"{lecture_name}_full.txt", "w", encoding="utf-8") as f:
    for slide in slides:
        f.write(slide['content'] + "\n")
