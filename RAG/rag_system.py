from RAG.retriever import Retriever
from loading_spinner import generate_response
from model_loader import load_model
import argparse
import torch

available_models = {
    "tiny": "sshleifer/tiny-gpt2",  # Very small (~14M), debugging only

    "neo-small": "EleutherAI/gpt-neo-125M",  # GPT-3-like, best under 8GB RAM (~125M)

    # GPU models
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",  # LLaMA 7B model (requires 16GB+ VRAM)
}

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot (CPU-only)")
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models.keys(),
        default="neo-small",
        help="Model to load (default: neo-small)"
    )
    args = parser.parse_args()
    model_name = available_models[args.model]

    print(f"[INFO] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, verbose=True)

    retriever = Retriever()
    document_store = retriever.load_documents()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting the chat. Goodbye!")
            break

        retrieved_docs = retriever.retrieve(user_input, tokenizer=tokenizer, max_tokens=1000)
        augmented_input = (
            f"Answer the following QUESTION based only on the provided CONTEXT.\n"
            f"CONTEXT:\n{retrieved_docs}\n\n"
            f"QUESTION: {user_input}\n"
            f"ANSWER:"
        )
        # print (augmented_input)
        reserved_tokens = 150  # leave space for generation
        max_len = model.config.max_position_embeddings - reserved_tokens

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(
            augmented_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,  # Or less if you want to leave room for output
        ).to(device)

        # This is where we call our improved generation function
        response = generate_response(model, tokenizer, inputs)

        print(f"\nLamma: {response}\n")


if __name__ == "__main__":
    main()