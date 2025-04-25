from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import requests


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a completion based on the provided prompt.
        
        Args:
            prompt: The input text to generate a completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            The generated text completion
        """
        pass


class GPT4Interface(LLMInterface):
    """Implementation of the LLM interface using GPT-4o."""
    
    def __init__(self, api_key: str):
        """Initialize the GPT-4o interface.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        # Import here to avoid dependency if not using this provider
        import openai
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a completion using GPT-4o.
        
        Args:
            prompt: The input text to generate a completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            The generated text completion
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class LocalLLMInterface(LLMInterface):
    """Implementation for a locally running LLM."""
    
    def __init__(self, model_path: str, **kwargs):
        """Initialize the local LLM interface.
        
        Args:
            model_path: Path to the local model
            **kwargs: Additional arguments for the model
        """
        self.model_path = model_path
        self.kwargs = kwargs
        # This is a placeholder - TODO implement 

        import ollama
        self.client = ollama.Client()
        self.model = model_path
        

    def generate_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate a completion using a local LLM.
        
        Args:
            prompt: The input text to generate a completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            The generated text completion
        """
        
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response

class OllamaInterface(LLMInterface):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate_completion(self, prompt, max_tokens=1000, temperature=0.7):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }
        response = requests.post(f"{self.host}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]


def get_llm_interface(provider: str, **kwargs) -> LLMInterface:
    """Factory function to get the appropriate LLM interface.
    
    Args:
        provider: The name of the LLM provider ('gpt4o', 'local', etc.)
        **kwargs: Provider-specific arguments
        
    Returns:
        An instance of the appropriate LLM interface
    """
    if provider.lower() == 'gpt4o':
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("API key is required for GPT-4o interface")
        return GPT4Interface(api_key)
    elif provider.lower() == 'ollama':
        model_name = kwargs.get('model_path')
        if not model_name:
            raise ValueError("Model name (model_path) is required for Ollama interface")
        return OllamaInterface(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")