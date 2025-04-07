import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name):
    try:
        print("Loading model... This might take a while.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16  # This ensures it's compatible with your GPU
        ).to("cuda")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Model is running on: {model.device}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
