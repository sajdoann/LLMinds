import json
import zipfile

import pandas as pd
import gzip
from tqdm import tqdm  # progress bar

data_split = 'dev'  # 'train', 'eval', 'dev'

# Load dataset
file_path = f"{data_split}_v2.1.json.gz"
if data_split == 'eval':
    file_path = f"{data_split}_v2.1_public.json.gz"
with gzip.open(file_path, "rt", encoding="utf-8") as f:
    data = json.load(f)

qa_data = []

# Assuming there is a "passages" key inside the dataset
passages_data = data.get("passages", {})
answers_data = data.get("answers", {})
query_type_data = data.get("query_type", {})  # <--- Get the query_type dictionary

for query_id, _ in tqdm(data["query_id"].items(), desc="Processing Queries", unit="query"):

    answers = answers_data.get(query_id, {})
    if "No Answer Present." in answers:  # Skip unanswered questions
        continue

    # Extract corresponding question
    question = data.get("query", {}).get(query_id, "")

    # Extract query type
    query_type = query_type_data.get(query_id, "UNKNOWN")  # fallback to "UNKNOWN" if missing

    # Extract relevant passages
    relevant_passages = []
    for passage in passages_data.get(query_id, []):
        if passage.get("is_selected", 0) == 1:
            relevant_passages.append(passage["passage_text"])

    # If relevant passages exist, save them
    if relevant_passages:
        qa_data.append({
            "query_id": query_id,
            "question": question,
            "answer": answers,
            "query_type": query_type,
            "passages": " ".join(relevant_passages)
        })
    elif data_split == 'eval':
        qa_data.append({
            "query_id": query_id,
            "question": question,
            "query_type": query_type
        })

# Convert to DataFrame
df = pd.DataFrame(qa_data)

# Display first few rows
print(df.head())
print(f"Number of questions: {len(df)}")

# Save to CSV for training
csv_filename = f"qa_data_{len(df)}.csv"
df.to_csv(csv_filename, index=False)

# Create ZIP file
zip_filename = f"qa_data_{len(df)}.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_filename)

print(f"{data_split} dataset saved as {csv_filename} and zip created as {zip_filename}")
