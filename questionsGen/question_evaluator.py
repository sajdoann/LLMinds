from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from llm_interface import LLMInterface


class QuestionEvaluator:
    """Module for evaluating and selecting questions."""
    
    def __init__(self, llm_interface: LLMInterface = None):
        """Initialize the question evaluator.
        
        Args:
            llm_interface: Optional LLM interface for evaluation assistance
        """
        self.llm = llm_interface
    
    def evaluate_questions(self, questions: List[Dict[str, Any]], document_text: str) -> List[Dict[str, Any]]:
        """Evaluate questions for quality and add evaluation scores.
        
        Args:
            questions: List of question dictionaries
            document_text: The original document text
            
        Returns:
            The input questions with additional evaluation scores
        """
        evaluated_questions = []
        
        for q in questions:

            answerable = self._check_answerable(q, document_text)
            
            # Add evaluation metrics
            q["evaluation"] = {
                "answerable": answerable,
                "context_relevance": self._score_context_relevance(q, document_text),
                "diversity_score": 0  # Will be filled in later
            }
            
            evaluated_questions.append(q)
        
        # comparing each question against others
        evaluated_questions = self._compute_diversity_scores(evaluated_questions)
        
        return evaluated_questions
    
    def select_questions(self, questions: List[Dict[str, Any]], 
                        num_to_select: int = None, 
                        min_difficulty: Dict[str, int] = None,
                        min_category: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """Select a diverse subset of questions.
        
        Args:
            questions: List of question dictionaries with evaluation scores
            num_to_select: Total number of questions to select (default: half of input)
            min_difficulty: Minimum number of questions per difficulty level
                            e.g., {"easy": 2, "medium": 3, "hard": 1}
            min_category: Minimum number of questions per category
                          e.g., {"factual": 2, "inferential": 2, "analytical": 2}
            
        Returns:
            Selected subset of questions
        """
        if not questions:
            return []
        
        # Default to selecting half of the questions
        if num_to_select is None:
            num_to_select = max(1, len(questions) // 2)
        
        # Ensure num_to_select doesn't exceed the number of available questions
        num_to_select = min(num_to_select, len(questions))
        
        # Sort questions by evaluation metrics (answerable first, then by diversity score)
        sorted_questions = sorted(
            questions,
            key=lambda q: (
                q["evaluation"]["answerable"],
                q["evaluation"]["diversity_score"],
                q["evaluation"]["context_relevance"]
            ),
            reverse=True
        )
        
        selected = []
        remaining = sorted_questions.copy()
        
        if min_difficulty:
            selected, remaining = self._satisfy_minimum_requirements(
                selected, remaining, num_to_select, "difficulty", min_difficulty
            )
        
        if min_category:
            selected, remaining = self._satisfy_minimum_requirements(
                selected, remaining, num_to_select, "category", min_category
            )
        
        # Fill the rest of the selection based on diversity and quality
        while len(selected) < num_to_select and remaining:
            # Get the highest-scoring question from remaining
            selected.append(remaining.pop(0))
        
        return selected
    
    def _satisfy_minimum_requirements(self, 
                                     selected: List[Dict[str, Any]], 
                                     remaining: List[Dict[str, Any]], 
                                     num_to_select: int,
                                     attribute: str, 
                                     min_counts: Dict[str, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Helper to satisfy minimum requirements for a given attribute.
        
        Args:
            selected: Currently selected questions
            remaining: Remaining questions to choose from
            num_to_select: Total number of questions to select
            attribute: The attribute to check (e.g., "difficulty", "category")
            min_counts: Minimum count for each value of the attribute
            
        Returns:
            Tuple of (updated selected questions, updated remaining questions)
        """
        # How many of each type we have
        current_counts = defaultdict(int)
        for q in selected:
            current_counts[q[attribute]] += 1
        
        # Try to satisfy minimum requirements
        for attr_value, min_count in min_counts.items():

            if current_counts[attr_value] >= min_count:
                continue
            
            needed = min_count - current_counts[attr_value]
            matching_questions = [q for q in remaining if q[attribute] == attr_value]
            
            # Sort them by quality metrics
            matching_questions.sort(
                key=lambda q: (
                    q["evaluation"]["answerable"],
                    q["evaluation"]["diversity_score"],
                    q["evaluation"]["context_relevance"]
                ),
                reverse=True
            )
            
            to_select = matching_questions[:needed]
            
            # Add to selected and remove from remaining
            selected.extend(to_select)
            for q in to_select:
                remaining.remove(q)
            
            # Update counts
            current_counts[attr_value] += len(to_select)
            
            # Check if we've reached the total limit
            if len(selected) >= num_to_select:
                break
        
        return selected, remaining
    
    def _check_answerable(self, question: Dict[str, Any], document_text: str) -> bool:
        """Check if a question is answerable from the document.
        
        Args:
            question: The question dictionary
            document_text: The document text
            
        Returns:
            True if the question appears answerable, False otherwise
        """

        if question["context"] not in document_text:
            return False
        
        # If we have an LLM, we could ask it to verify answerability
        if self.llm:
            prompt = f"""
            Document excerpt: 
            "{question['context']}"
            
            Question: {question['question']}
            
            Can the question be answered completely and accurately using ONLY the information provided in the document excerpt?
            Answer with just "Yes" or "No".
            """
            
            response = self.llm.generate_completion(prompt, max_tokens=10, temperature=0.1)
            return response.strip().lower().startswith("yes")
        
        # If no LLM, assume answerable if context is in document
        return True
    
    def _score_context_relevance(self, question: Dict[str, Any], document_text: str) -> float:
        """Score how relevant the provided context is to the question.
        
        Args:
            question: The question dictionary
            document_text: The document text
            
        Returns:
            A relevance score from 0.0 to 1.0
        """
        
        # Basic implementation - could be enhanced with more sophisticated NLP
        # Here we just check if key terms from the question appear in the context
        context = question["context"].lower()
        q_text = question["question"].lower()
        
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "with", "by"}
        q_words = [w for w in q_text.split() if w not in stop_words]
        
        # How many question words appear in the context
        matches = sum(1 for word in q_words if word in context)
        
        if not q_words:
            return 0.0
            
        return matches / len(q_words)
    
    def _compute_diversity_scores(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute diversity scores for each question compared to others.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            The same questions with diversity_score added to each evaluation
        """

        # TODO also eval based on What, Why, How.. diversity
        if not questions:
            return questions
            
        texts = [q["question"].lower() for q in questions]
        contexts = [q["context"].lower() for q in questions]
        
        # For each question, calculate its average difference from others
        for i, q in enumerate(questions):

            text_similarities = []
            context_similarities = []
            
            for j in range(len(questions)): # similarity to other questions
                if i == j:
                    continue
                
                # word overlap for text similarity
                q1_words = set(texts[i].split())
                q2_words = set(texts[j].split())
                if q1_words:
                    text_sim = len(q1_words.intersection(q2_words)) / len(q1_words)
                else:
                    text_sim = 0
                
                # word overlap for context similarity
                c1_words = set(contexts[i].split())
                c2_words = set(contexts[j].split())
                if c1_words:
                    context_sim = len(c1_words.intersection(c2_words)) / len(c1_words)
                else:
                    context_sim = 0
                
                text_similarities.append(text_sim)
                context_similarities.append(context_sim)
            
            # Average similarities (lower is more diverse)
            avg_text_sim = sum(text_similarities) / max(1, len(text_similarities))
            avg_context_sim = sum(context_similarities) / max(1, len(context_similarities))
            
            # Overall diversity score (higher is better)
            diversity_score = 1.0 - ((avg_text_sim + avg_context_sim) / 2.0)
            
            q["evaluation"]["diversity_score"] = diversity_score
        
        return questions