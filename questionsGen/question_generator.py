from typing import List, Dict, Any, Optional

from llm_interface import LLMInterface


class QuestionGenerator:
    """Module for generating questions from a document using an LLM."""
    
    def __init__(self, llm_interface: LLMInterface):
        """Initialize the question generator.
        
        Args:
            llm_interface: The LLM interface to use for generation
        """
        self.llm = llm_interface
    
    def generate_questions_from_file(self, file_path: str, num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate questions from a text file.
        
        Args:
            file_path: Path to the document file
            num_questions: Optional number of questions to generate, if None let the LLM decide
            
        Returns:
            A list of question dictionaries with the following structure:
            [
                {
                    "question": "What is X?",
                    "context": "The part of the document this question refers to",
                    "difficulty": "easy/medium/hard",
                    "category": "factual/inferential/analytical"
                },
                ...
            ]
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read()
            
            return self.generate_questions_from_text(document_text, num_questions)
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    def generate_questions_from_text(self, document_text: str, num_questions: Optional[int] = None, json_mode = False) -> List[Dict[str, Any]]:
        """Generate questions from text content.
        
        Args:
            document_text: The document text content
            num_questions: Optional number of questions to generate, if None let the LLM decide
            
        Returns:
            A list of question dictionaries
        """
        prompt = self._create_question_generation_prompt(document_text, num_questions)
        
        response = self.llm.generate_completion(prompt, max_tokens=2000, temperature=0.7)

        if not json_mode:
            return response

        # Parse the response into a list of questions
        try:
            questions = self._extract_json_from_response(response)
            return questions
        except Exception as e:
            raise Exception(f"Failed to parse LLM response into questions: {str(e)}\nResponse: {response}")
    
    def _create_question_generation_prompt(self, document_text: str, num_questions: Optional[int] = None) -> str:
        """Create a prompt for generating questions.
        
        Args:
            document_text: The document text
            num_questions: Optional number of questions to request
            
        Returns:
            The formatted prompt
        """
        question_count_instruction = f"Generate exactly {num_questions} questions." if num_questions else "Generate an appropriate number of questions based on the document's length and complexity."
        
        prompt = f"""
        I'm going to provide you with a document. Your task is to create a list of questions based solely on the information in this document. 

        IMPORTANT GUIDELINES:
        1. {question_count_instruction}
        2. The questions must be diverse and cover different parts of the document.
        3. Each question must be answerable using ONLY the information in the document - no external knowledge should be required.
        4. For each question, include:
           - The question itself
           - The specific part of the document (context) that contains the answer
           - A difficulty rating (easy, medium, hard)
           - A category (factual, inferential, analytical)

        5. Tell me only the questions
        DOCUMENT:
        {document_text}
        """
        json_prompt = """
        Format your response as a JSON array of objects with this structure:
        ```json
        [
          {{
            "question": "The question text",
            "context": "The specific part of the document containing the answer",
            "difficulty": "easy/medium/hard",
            "category": "factual/inferential/analytical"
          }},
          ...
        ]
        """
        return prompt
    
    def _extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract and parse JSON from the LLM response. Tries to find JSON content between backticks or just in the response       
        Args:
            response: The LLM response text
            
        Returns:
            The parsed list of question dictionaries
        """
        
        import re
        import json
        
        # Look for JSON between triple backticks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code blocks found, try to find an array in the response
            json_match = re.search(r"\[\s*{[\s\S]*}\s*\]", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to parse the whole thing
                json_str = response
        
        try:
            questions = json.loads(json_str)
            return questions
        except json.JSONDecodeError:
            # Try cleaning the string more aggressively
            cleaned_str = re.sub(r'[^\[\]\{\}"\',:.\w\s\-_]', '', json_str)
            return json.loads(cleaned_str)