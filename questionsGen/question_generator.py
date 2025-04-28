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
    
    def generate_questions_from_text(self, document_text: str, num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate questions from text content.
        
        Args:
            document_text: The document text content
            num_questions: Optional number of questions to generate, if None let the LLM decide
            
        Returns:
            A list of question dictionaries
        """
        # For shorter texts, try the standard approach
        if len(document_text) < 4000:
            try:
                prompt = self._create_question_generation_prompt(document_text, num_questions)
                response = self.llm.generate_completion(prompt, max_tokens=2000, temperature=0.7)
                return self._extract_json_from_response(response)
            except Exception as e:
                print(f"Failed with standard approach: {str(e)}. Falling back to chunking.")
        
        # Fallback for longer texts or if parsing failed: chunk the document
        return self._generate_questions_with_chunking(document_text, num_questions)

    def _generate_questions_with_chunking(self, document_text: str, total_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate questions by splitting document into chunks and processing each separately.
        
        Args:
            document_text: The document text content
            total_questions: Total number of questions to generate
            
        Returns:
            A list of question dictionaries from all chunks combined
        """
        # Determine how many questions to generate
        if total_questions is None:
            total_questions = 5  # Default if not specified
        
        # Split the document into chunks
        chunks = self._split_into_chunks(document_text)
        print(f"Split document into {len(chunks)} chunks for processing")
        
        # Calculate questions per chunk, ensuring we get at least the requested number
        questions_per_chunk = max(1, total_questions // len(chunks))
        extra_questions = total_questions % len(chunks)
        
        all_questions = []
        
        for i, chunk in enumerate(chunks):
            # Determine how many questions for this chunk
            num_for_chunk = questions_per_chunk + (1 if i < extra_questions else 0)
            
            try:
                # Create a simpler prompt for a single chunk
                prompt = self._create_chunk_question_prompt(chunk, num_for_chunk)
                response = self.llm.generate_completion(prompt, max_tokens=1500, temperature=0.7)
                
                # Extract questions from this chunk's response
                try:
                    chunk_questions = self._extract_json_from_response(response)
                    all_questions.extend(chunk_questions)
                    print(f"Generated {len(chunk_questions)} questions from chunk {i+1}")
                except Exception as e:
                    # If JSON extraction fails, try a more aggressive approach
                    chunk_questions = self._extract_questions_manually(response)
                    all_questions.extend(chunk_questions)
                    print(f"Used fallback parsing for chunk {i+1}, got {len(chunk_questions)} questions")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
        
        return all_questions

    def _split_into_chunks(self, text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
        """Split text into overlapping chunks of approximately chunk_size characters.
        
        Args:
            text: The text to split
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Don't cut in the middle of a paragraph if possible
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1  # Include the period
            
            chunks.append(text[start:end])
            
            # Move start position for next chunk, considering overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks

    def _create_chunk_question_prompt(self, chunk_text: str, num_questions: int = 1) -> str:
        """Create a simpler prompt for generating questions from a document chunk.
        
        Args:
            chunk_text: The document chunk text
            num_questions: Number of questions to generate for this chunk
            
        Returns:
            The formatted prompt
        """
        # Simpler prompt focused on a single chunk
        prompt = f"""
        I'll provide you with a part of a document. Please create {num_questions} question(s) based solely on this text.

        For each question, include:
        - The question text
        - The specific part of the text containing the answer (context)
        - Difficulty (easy, medium, or hard)
        - Category (factual, inferential, or analytical)

        IMPORTANT: Format your response as valid JSON like this example:
        [
          {{
            "question": "What is X?",
            "context": "X is a technology used for Y.",
            "difficulty": "easy",
            "category": "factual"
          }}
        ]

        Only respond with the JSON array and nothing else.

        TEXT:
        {chunk_text}
        """
        return prompt

    def _extract_questions_manually(self, response: str) -> List[Dict[str, Any]]:
        """Extract questions when JSON parsing fails.
        
        Args:
            response: The LLM response text
            
        Returns:
            List of question dictionaries constructed from the response
        """
        import re
        
        questions = []
        
        # Look for question patterns
        q_patterns = re.finditer(r'(?:(?:\d+\.\s+)|(?:"question":\s*"))([^"]+)(?:")?', response)
        c_patterns = re.finditer(r'(?:"context":\s*")([^"]+)(?:")', response)
        d_patterns = re.finditer(r'(?:"difficulty":\s*")([^"]+)(?:")', response)
        cat_patterns = re.finditer(r'(?:"category":\s*")([^"]+)(?:")', response)
        
        # Extract the matched text
        qs = [match.group(1) for match in q_patterns]
        contexts = [match.group(1) for match in c_patterns]
        difficulties = [match.group(1) for match in d_patterns]
        categories = [match.group(1) for match in cat_patterns]
        
        # Construct as many complete questions as we can
        for i in range(min(len(qs), len(contexts), len(difficulties), len(categories))):
            questions.append({
                "question": qs[i],
                "context": contexts[i],
                "difficulty": difficulties[i].lower(),
                "category": categories[i].lower()
            })
        
        return questions

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
        ```

        DOCUMENT:
        {document_text}
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