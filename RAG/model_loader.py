import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, verbose=False):
    print("Loading model... This might take a while.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded from: {tokenizer.name_or_path}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return None, None

    try:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        print(f"Model loaded on: {model.device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

    if verbose:
        print("\nTokenizer info:")
        print(f"- Name or path: {tokenizer.name_or_path}")
        print(f"- Vocab size: {tokenizer.vocab_size}")
        print(f"- Max length: {tokenizer.model_max_length}")
        print(f"- EOS token: {tokenizer.eos_token} ({tokenizer.convert_tokens_to_ids(tokenizer.eos_token)})")
        print(f"- Tokenizer class: {tokenizer.__class__.__name__}")

        print("\nModel config:")
        print(f"- Name or path: {getattr(model, 'name_or_path', 'N/A')}")
        print(f"- Vocab size: {model.config.vocab_size}")
        print(f"- Model type: {model.config.model_type}")

    return model, tokenizer