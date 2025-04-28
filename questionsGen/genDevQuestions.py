import os
import json
import traceback
from typing import List, Dict, Any, Optional
from config import Config

from llm_interface import get_llm_interface
from question_generator import QuestionGenerator 
from question_evaluator import QuestionEvaluator

def find_txt_files(base_dir: str) -> List[str]:
    """Find all .txt files in the specified directory (recursively)."""
    txt_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


def generate_questions_for_file(file_path: str, config: Config) -> Optional[Dict[str, Any]]:
    """Generate and evaluate questions for a given text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        llm_interface = get_llm_interface(
            provider=config.get("llm", "provider"),
            model_path=config.get("llm", "model_path")
        )
        
        generator = QuestionGenerator(llm_interface)
        
        num_questions = 5
        print(f"Generating {num_questions} questions...")
        
        try:
            questions = generator.generate_questions_from_text(content, num_questions)
            print(f"Generated {len(questions)} questions")
            
            print("\nEvaluating questions...")
            evaluator = QuestionEvaluator(llm_interface if config.get("evaluation", "use_llm") else None)
            evaluated_questions = evaluator.evaluate_questions(questions, content)
            
            print("Selecting best questions...")
            selected = evaluator.select_questions(
                evaluated_questions,
                num_to_select=3,
                min_difficulty=config.get("evaluation", "min_difficulty"),
                min_category=config.get("evaluation", "min_category")
            )
            
            print("\nSelected questions:")
            for i, q in enumerate(selected, 1):
                print(f"{i}. {q['question']} ({q['difficulty']}, {q['category']})")
                print(f"   Diversity score: {q['evaluation']['diversity_score']:.2f}")
                print(f"   Context: \"{q['context'][:50]}...\"")
                print()
                
            return {
                "all_questions": evaluated_questions,
                "selected_questions": selected
            }
        except Exception as e:
            print(f"ERROR: Failed to generate or process questions: {str(e)}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"ERROR: Failed to process file: {str(e)}")
        traceback.print_exc()
        return None


def save_questions(file_path: str, question_data: Dict[str, Any], output_dir: str) -> Optional[str]:
    """Save the generated and evaluated questions to a JSON file and return the output path."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        rel_path = os.path.relpath(file_path, devset_dir)
        output_path = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}_questions.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    except Exception as e:
        print(f"ERROR: Failed to save questions: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    config = Config()
    
    # Base directory for the devset
    devset_dir = "/Users/hlava/Library/Mobile Documents/com~apple~CloudDocs/MFF/Mgr.2semestr/LLMinds/devset"
    
    # Directory to save the generated questions
    output_dir = "/Users/hlava/Library/Mobile Documents/com~apple~CloudDocs/MFF/Mgr.2semestr/LLMinds/generated_questions"
    
    failed_files = []
    txt_files = find_txt_files(devset_dir)
    print(f"Found {len(txt_files)} .txt files in the devset folder.")
    
    for i, file_path in enumerate(txt_files):
        print(f"[{i+1}/{len(txt_files)}] Processing file: {file_path}")
        try:
            question_data = generate_questions_for_file(file_path, config)
            
            if question_data is not None:
                saved_path = save_questions(file_path, question_data, output_dir)
                if saved_path:
                    print(f"  Saved to: {saved_path}")
                    print(f"  Selected {len(question_data['selected_questions'])} questions out of {len(question_data['all_questions'])}")
                else:
                    failed_files.append(file_path)
            else:
                failed_files.append(file_path)
                
            print("-" * 60)
        except Exception as e:
            print(f"ERROR: Unexpected error processing {file_path}: {str(e)}")
            traceback.print_exc()
            failed_files.append(file_path)
            print("-" * 60)
    
    # summary
    print("\nProcessing complete!")
    print(f"Successfully processed: {len(txt_files) - len(failed_files)}/{len(txt_files)} files")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for i, failed_file in enumerate(failed_files, 1):
            print(f"  {i}. {failed_file}")