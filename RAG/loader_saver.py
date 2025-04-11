import csv
import json
import os
from datetime import datetime
import re


def load_questions(filepath="questions.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    reference_answers = [item.get("reference-answers", []) for item in data]

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
        "context": context_match.group(1).strip() if context_match else '',
    }


def save_responses(responses, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"responses/response_{args.model}_k{args.top_k}_r{args.reserved_tokens}_{timestamp}.json"

    # Convert each response to a string from characters
    responses = [str(r) for r in responses]
    responses = [extract_qa_structure(r) for r in responses]

    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(responses, file, ensure_ascii=False, indent=2)

    print(f"\nResponses saved as strings to: {filename}")

