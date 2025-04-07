from transformers import AutoTokenizer
import torch
import threading
from retriever import Retriever
from loading_spinner import loading_spinner, generate_response
from model_loader import load_model
import torch
import threading
from loading_spinner import loading_spinner

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # Load model and tokenizer
    model, tokenizer = load_model(model_name)
    
    retriever = Retriever()
    document_store = retriever.load_documents()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting the chat. Goodbye!")
            break

        retrieved_docs = retriever.retrieve(user_input, tokenizer=tokenizer, max_tokens=7000)
        augmented_input = (
            f"INSTRUCTION: Use ONLY the following CONTEXT to answer the question.\n"
            f"CONTEXT:\n{retrieved_docs}\n\n"
            f"QUESTION: {user_input}\n"
            f"ANSWER:"
        )
        # print (augmented_input)
        inputs = tokenizer(
            augmented_input,
            return_tensors="pt",
            truncation=True,
            max_length=8192,     # Or less if you want to leave room for output
        ).to("cuda")
        
        # This is where we call our improved generation function
        response = generate_response(model, tokenizer, inputs)
        
        print(f"\nLamam: {response}\n")

if __name__ == "__main__":
    main()