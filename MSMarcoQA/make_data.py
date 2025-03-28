import json
import pandas as pd
import gzip
from tqdm import tqdm # progress bar

data_split = 'dev' # 'train', 'eval', 'dev'

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

for query_id, _ in tqdm(data["query_id"].items(), desc="Processing Queries", unit="query"):

    answers = answers_data.get(query_id, {})
    if "No Answer Present." in answers:  # Skip unanswered questions
        continue

    # Extract corresponding question (assuming there's a 'questions' key)
    question = data.get("query", {}).get(query_id, "")


    # Extract relevant passages (if a structure exists)
    relevant_passages = []
    for passage in passages_data.get(query_id, []):  # Get passages linked to query_id
        if passage.get("is_selected", 0) == 1:  # Only use selected passages
            relevant_passages.append(passage["passage_text"])

    # If relevant passages exist, save them
    if relevant_passages:
        qa_data.append({
            "query_id": query_id,
            "question": question,
            "answer": answers,  # Assume single answer per query
            "passages": " ".join(relevant_passages)  # Concatenating all selected passages
        })
    # eval data
    elif data_split == 'eval':
        qa_data.append({
            "query_id": query_id,
            "question": question})


# Convert to DataFrame
df = pd.DataFrame(qa_data)

# Display first few rows
print(df.head())
print(f"Number of questions: {len(df)}")

# Save to CSV for training
df.to_csv(f"qa_{data_split}_data.csv", index=False)
print(f"{data_split} dataset saved as qa_{data_split}_data.csv")
