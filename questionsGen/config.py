import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv


class Config:
    """Configuration management for the question generator system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration, optionally from a file.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        load_dotenv()
        
        # Default
        self._config = {
            "llm": {
                "provider": "gpt4o", # PUT here "local" if wanted! + fill in model_path
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "model_path": "",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "question_generation": {
                "default_num_questions": None,  # Let LLM decide by default
                "prompt_template": "default"  # Use default prompt template
            },
            "evaluation": {
                "use_llm": False,
                "min_difficulty": {
                    "easy": 1,
                    "medium": 2,
                    "hard": 1
                },
                "min_category": {
                    "factual": 1,
                    "inferential": 2,
                    "analytical": 1
                }
            }
        }
        
        if config_path:
            self.load_from_file(config_path)
        
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to a JSON configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                file_config = json.load(file)
                
            # Merge with default config (only update specified values)
            self._merge_configs(self._config, file_config)
                
        except Exception as e:
            print(f"Warning: Failed to load configuration from {config_path}: {str(e)}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override config into base config.
        
        Args:
            base: Base configuration dictionary (modified in-place)
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get a configuration value.
        
        Args:
            section: Configuration section
            key: Optional key within the section
            
        Returns:
            Configuration value or section dictionary
        """
        if section not in self._config:
            raise ValueError(f"Configuration section '{section}' not found")
        
        if key is None:
            return self._config[section]
        
        if key not in self._config[section]:
            raise ValueError(f"Configuration key '{key}' not found in section '{section}'")
        
        return self._config[section][key]
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            section: Configuration section
            key: Key within the section
            value: Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
    
    def save_to_file(self, config_path: str) -> None:
        """Save the current configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration file
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(self._config, file, indent=2)
        except Exception as e:
            raise Exception(f"Failed to save configuration to {config_path}: {str(e)}")
    
    def as_dict(self) -> Dict[str, Any]:
        """Get a copy of the entire configuration as a dictionary.
        
        Returns:
            A copy of the configuration dictionary
        """
        return self._config.copy()
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        self.__init__()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary to merge
        """
        self._merge_configs(self._config, config_dict)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for easy access.
        
        Returns:
            LLM configuration dictionary
        """
        return self.get("llm")
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration for easy access.
        
        Returns:
            Evaluation configuration dictionary
        """
        return self.get("evaluation")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get question generation configuration for easy access.
        
        Returns:
            Question generation configuration dictionary
        """
        return self.get("question_generation")