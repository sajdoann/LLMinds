#!/usr/bin/env python3
"""
Example script demonstrating how to use the question generation system.

This script shows basic usage patterns for generating, evaluating,
and selecting questions from a document.
"""

import os
import sys
import json

from llm_interface import get_llm_interface
from question_generator import QuestionGenerator
from question_evaluator import QuestionEvaluator
from config import Config


def load_document(file_path):
    """Load document from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    """Run the example script."""
    # Load configuration (from environment or default)
    config = Config()
    
    # Check for API key in environment
    # if not config.get("llm", "api_key"):
    #     print("Error: API key not found. Please set the OPENAI_API_KEY environment variable.")
    #     sys.exit(1)
    
    llm = get_llm_interface(
        provider=config.get("llm", "provider"),
        model_path=config.get("llm", "model_path")
    )
    
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "./text.en.txt"
    
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        sys.exit(1)

    document_text = load_document(doc_path)
    print(f"Loaded document ({len(document_text)} characters)")
    
    generator = QuestionGenerator(llm)
    
    # Generate questions
    num_questions = config.get("question_generation", "default_num_questions") or 5
    print(f"Generating {num_questions} questions...")
    
    questions = generator.generate_questions_from_text(document_text, num_questions)
    print(f"Generated {len(questions)} questions")
    
    # Print the first few questions
    print("\nSample questions:")
    for i, q in enumerate(questions[:3], 1):
        print(f"{i}. {q['question']} ({q['difficulty']}, {q['category']})")
    
    evaluator = QuestionEvaluator(llm if config.get("evaluation", "use_llm") else None)
    
    print("\nEvaluating questions...")
    evaluated_questions = evaluator.evaluate_questions(questions, document_text)
    
    # Select best
    print("Selecting best questions...")
    selected = evaluator.select_questions(
        evaluated_questions,
        num_to_select=3,
        min_difficulty=config.get("evaluation", "min_difficulty"),
        min_category=config.get("evaluation", "min_category")
    )
    # Print selected 
    print("\nSelected questions:")
    for i, q in enumerate(selected, 1):
        print(f"{i}. {q['question']} ({q['difficulty']}, {q['category']})")
        print(f"   Diversity score: {q['evaluation']['diversity_score']:.2f}")
        print(f"   Context: \"{q['context'][:50]}...\"")
        print()
    

    output_path = f"{os.path.splitext(doc_path)[0]}_selected_questions.json"
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(selected, file, indent=2)
    
    print(f"Saved selected questions to {output_path}")


if __name__ == "__main__":
    main()