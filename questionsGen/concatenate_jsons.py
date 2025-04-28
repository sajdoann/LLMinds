import json
import os

# List of root directories to process
root_dirs = ["../generated_questionsDevSet", "../generated_questionsTestSet"]  # adjust if needed

# Output dictionary
output = {}

for root_dir in root_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter only .json files containing "questions" in their name
        json_files = [f for f in filenames if f.endswith(".json") and "questions" in f]

        if not json_files:
            continue  # No JSON files in this folder

        # Prefer 'text.en_questions.json' if present
        if "text.en_questions.json" in json_files:
            selected_file = "text.en_questions.json"
        else:
            selected_file = sorted(json_files)[0]  # Pick the first available one alphabetically

        file_path = os.path.join(dirpath, selected_file)
        relative_folder = os.path.relpath(dirpath, root_dir).replace("\\", "/")
        virtual_txt_filename = relative_folder + "/text.en.txt"

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
                continue

        # Extract questions
        if isinstance(data, dict) and "all_questions" in data:
            qas = []
            for q in data["all_questions"]:
                question_text = q.get("question", "").strip()
                context_text = q.get("context", "").strip()

                if question_text:
                    qas.append({
                        "question": question_text,
                        "reference-answers": [context_text] if context_text else []
                    })

            if qas:
                output[virtual_txt_filename] = qas
        else:
            print(f"Skipping unexpected format in {file_path}")

# Save the big JSON
with open("concatenated_questions.json", "w", encoding="utf-8") as out_f:
    json.dump(output, out_f, indent=2, ensure_ascii=False)

print("✅ Finished! Output written to 'concatenated_questions.json'")

