import csv
import json
import os
from datetime import datetime
import re


import json

def load_questions(filepath="questions.json"):
    print("Loading questions from:", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If data is a list of strings (new format)
    if all(isinstance(item, str) for item in data):
        questions = data
        reference_answers = [[] for _ in data]

    # If data is a list of dicts (old format)
    elif all(isinstance(item, dict) and "question" in item for item in data):
        questions = [item["question"] for item in data]
        reference_answers = [item.get("reference-answers", []) for item in data]

    else:
        raise ValueError("Unsupported question file format.")

    return questions, reference_answers



def extract_qa_structure(text):
    # careful order of INSTRUCTIONS, CONTEXT, QUESTION, ANSWER is important here
    # Extract parts
    instructions_match = re.search(r'INSTRUCTIONS:(.*?)CONTEXT:', text, re.IGNORECASE | re.DOTALL)
    context_match = re.search(r'CONTEXT:(.*?)QUESTION:', text, re.IGNORECASE | re.DOTALL)
    question_match = re.search(r'QUESTION:(.*?)ANSWER:', text, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r'ANSWER:(.*)$', text, re.IGNORECASE | re.DOTALL)

    return {
        #"instructions": instructions_match.group(1).strip() if instructions_match else '',
        "question": question_match.group(1).strip() if question_match else '',
        "answer": answer_match.group(1).strip() if answer_match else '',
        #"context": context_match.group(1).strip() if context_match else '',
    }


import os
import json
from datetime import datetime

def save_responses(responses, key, outdir, filename):
    # Create folder if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Convert responses to the desired structure
    responses = [str(r) for r in responses]
    responses = [extract_qa_structure(r) for r in responses]

    # Load existing file if it exists
    if os.path.exists(filename):
        with open(filename, mode='r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = {}

    # Overwrite if key exists, or add if not
    data[key] = responses

    # Save back to file
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"\nSUCCESS SAVED FOR '{key}' to: {filename}")

def is_answer_already_computed(json_path, questions_key):
    """
    Checks whether answers for a given key are already stored in the JSON file.

    Args:
        json_path (str): Path to the JSON file.
        questions_key (str): The key to check in the JSON (e.g. 'video-22/questions_by_TeamX.json').

    Returns:
        bool: True if the key exists, False otherwise.
    """
    if not os.path.exists(json_path):
        return False

    with open(json_path, mode='r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return False

    return questions_key in data


