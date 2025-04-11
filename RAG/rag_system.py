from RAG.loader_saver import load_questions, save_responses
from RAG.retriever import Retriever
from loading_spinner import generate_response
from model_loader import load_model
import argparse
import torch

available_models = {
    "tiny": "sshleifer/tiny-gpt2",  # Very small (~14M), debugging only
    "neo-small": "EleutherAI/gpt-neo-125M",  # GPT-3-like, best under 8GB RAM (~125M) CPU only okay

    # GPU models
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",  # LLaMA 7B model (requires 16GB+ VRAM)
}

def ask_and_answer(question, model, tokenizer, retriever, top_k, max_retireve_tokens, instruction, answer_prompt):

    retrieved_docs = retriever.retrieve(
        question, tokenizer=tokenizer, max_tokens=max_retireve_tokens, top_k=top_k
    )

    context_input = f"CONTEXT: {retrieved_docs}\n"
    augmented_input = f"{instruction}\n{context_input}\nQUESTION:{question}\n{answer_prompt}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(
        augmented_input,
        return_tensors="pt",
        truncation=False,
        #max_length=max_len,
    ).to(device)

    response = generate_response(model, tokenizer, inputs)
    print(f"\n 🦙 Response: {response}\n")
    return response


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models.keys(),
        default="neo-small",
        help="Model to load (default: neo-small)"
    )
    default_reserved_tokens = 256 # needs to be experimented with
    parser.add_argument(
        "--reserved_tokens",
        type=int,
        default=default_reserved_tokens,
        help=f"Number of tokens to reserve for generation for sure (default: {default_reserved_tokens})"
    )
    default_top_k = 20
    parser.add_argument(
        "--top_k",
        type=int,
        default=default_top_k,
        help=f"Number of documents to retrieve (default: {default_top_k})"
    )
    # flag for interactive mode, don't need to pass when loading questions from file
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode (default: False)"
    )

    args = parser.parse_args()
    model_name = available_models[args.model]
    reserved_tokens = args.reserved_tokens  # leave space for generation
    top_k = args.top_k  # number of documents to retrieve
    interactive = args.interactive


    print(f"[INFO] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, verbose=True)

    retriever = Retriever()
    document_store = retriever.load_documents()

    instruction_prompt = (f"INSTRUCTIONS: Answer the following QUESTION based only on the provided CONTEXT.\n")
    answer_prompt = "ANSWER:"

    instruction_tokens = tokenizer(instruction_prompt, return_tensors="pt")["input_ids"].shape[1]
    answer_tokens = tokenizer(answer_prompt, return_tensors="pt")["input_ids"].shape[1]

    max_retrieve_tokens_all = model.config.max_position_embeddings - reserved_tokens - instruction_tokens - answer_tokens

    responses = []

    if not interactive:
        questions, golden_answers = load_questions('questions.json')
        for question in questions:
            max_retrieve_tokens = max_retrieve_tokens_all - tokenizer(question, return_tensors="pt")["input_ids"].shape[1] #subtract question tokens
            response = ask_and_answer(question, model, tokenizer, retriever, top_k, max_retrieve_tokens, instruction_prompt, answer_prompt)
            responses.append(response)
    else:
        print("Entering interactive mode. Type 'exit', 'quit', or 'q' to stop.")
        while True:
            question = input("You: ")
            if question.lower() in ['exit', 'quit', 'q']:
                print("Exiting the chat. Goodbye!")
                break
            max_retrieve_tokens = max_retrieve_tokens_all - tokenizer(question, return_tensors="pt")["input_ids"].shape[1] #subtract question tokens
            response = ask_and_answer(question, model, tokenizer, retriever, top_k, max_retrieve_tokens, instruction_prompt, answer_prompt)
            responses.append(response)

    # Save responses to a file
    save_responses(responses, args) # careful about the response structure, extract regex

if __name__ == "__main__":
    main()