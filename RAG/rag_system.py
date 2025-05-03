from loader_saver import load_questions, save_responses, is_answer_already_computed
from retriever import Retriever
from loading_spinner import generate_response
from model_loader import load_model
import argparse
import torch
import gc
from tqdm import tqdm

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnNewlineCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.newline_token_id


available_models = {
    "tiny": "sshleifer/tiny-gpt2",
    "neo-small-125M": "EleutherAI/gpt-neo-125M",
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
    "qwen-1.5b": "Qwen/Qwen1.5-1.8B",
    "mistral-7b":"mistralai/Mistral-7B-v0.1",
    "euler": "EleutherAI/gpt-neox-20b",
    "distqwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

def main():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.ipc_collect()

    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--model", type=str, choices=available_models.keys(), default="neo-small")
    parser.add_argument("--reserved_tokens", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--document", type=str, default="documents.txt")
    parser.add_argument("--questions", type=str, default="questions.json")
    parser.add_argument("--outdir", type=str, default="responses")
    parser.add_argument("--data", type=str, default="all")

    args = parser.parse_args()
    model_name = available_models[args.model]
    questions_filepath = args.questions
    outdir = args.outdir
    data_name = args.data if args.data != "all" else f"all_{args.model}"
    key = questions_filepath.split('/', 1)[1]
    save_filename = f"{outdir}/{data_name}.json"

    if is_answer_already_computed(save_filename, key):
        print(f"ALREADY DONE FOR KEY {key}\nRETURN")
        return

    print(f"[INFO] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, verbose=True)
    tokenizer.pad_token = tokenizer.eos_token

    retriever = Retriever()
    retriever.load_documents(args.document)

    instruction_prompt = "INSTRUCTIONS: Answer the following QUESTION based only on the provided CONTEXT. The ANSWER is 1 short sentence."
    answer_prompt = "ANSWER:"

    instruction_tokens = tokenizer(instruction_prompt, return_tensors="pt")["input_ids"].shape[1]
    answer_tokens = tokenizer(answer_prompt, return_tensors="pt")["input_ids"].shape[1]
    max_retrieve_tokens_all = model.config.max_position_embeddings - args.reserved_tokens - instruction_tokens - answer_tokens

    responses = []

    if not args.interactive:
        questions, _ = load_questions(questions_filepath)
        batch_size = 1 # llama needs small. qwen cna have 8
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
            batch = questions[i:i + batch_size]
            prompts = []

            for question in batch:
                max_retrieve_tokens = max_retrieve_tokens_all - tokenizer(question, return_tensors="pt")["input_ids"].shape[1]
                retrieved_docs = retriever.retrieve(question, tokenizer=tokenizer, max_tokens=max_retrieve_tokens, top_k=args.top_k)
                context_input = f"CONTEXT: {retrieved_docs}\n"
                prompts.append(f"{instruction_prompt}\n{context_input}\nQUESTION:{question}\n{answer_prompt}")

            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            try:
                with torch.no_grad():
                    stopping_criteria = StoppingCriteriaList([StopOnNewlineCriteria(tokenizer)])
                    outputs = model.generate(
                        **tokenized,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=stopping_criteria
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                responses.extend(decoded)
            except Exception as e:
                print(f"[ERROR] Generation failed for batch starting at question {i}: {e}")
    else:
        print("Interactive mode not supported in batched version.")

    save_responses(responses, key, outdir, save_filename)

if __name__ == "__main__":
    main()

