#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
from datetime import datetime
from collections import Counter
from typing import List, Dict

import torch
import gc

from model_loader import load_model
from loader_saver import extract_qa_structure

from rag_system import available_models

def majority_vote(answers: List[str]) -> str:
    """Return the most frequent normalised answer or empty string."""
    norm = [re.sub(r"\s+", " ", a.strip().lower()) for a in answers]
    if not norm:
        return ""
    counter = Counter(norm)
    best, _ = counter.most_common(1)[0]
    # Return the original answer string that matches best
    for a in answers:
        if re.sub(r"\s+", " ", a.strip().lower()) == best:
            return a.strip()
    return best


def build_selector_prompt(question: str, candidate_answers: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {a.strip()}" for i, a in enumerate(candidate_answers))
    return (
        "SYSTEM: You are an expert judge for question‑answering. "
        "Select **one** best answer. If several candidates are equally good, "
        "return the one that occurs most frequently (majority vote). Do NOT "
        "explain your choice. The response must be a single short sentence."\
        "\n\nQUESTION: " + question.strip() + "\n\nCANDIDATE_ANSWERS:\n" + numbered + "\n\nBEST_ANSWER:"
    )


def select_answer(selector, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = selector.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract everything after BEST_ANSWER: (robust against chat templates)
    m = re.search(r"BEST_ANSWER:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    answer = m.group(1).strip() if m else text.split("BEST_ANSWER:")[-1].strip()
    # post‑clean - only first sentence
    answer = answer.split(". ")[0].rstrip(".") + "."
    return answer


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble answer selector for RAG outputs")
    parser.add_argument("--data", required=True, help="Dataset key prefix (e.g. nmt, bio1, …)", default="bio1")
    parser.add_argument("--selector_model", default="llama-7b", choices=available_models.keys(),
                        help="Model alias to use for answer selection (default: llama-7b)")
    parser.add_argument("--responses_dir", default="responses", help="Directory with *_*.json files")
    parser.add_argument("--outdir", default="responses", help="Where to save the final ensemble file")
    parser.add_argument("--include_candidates", action="store_true", help="Keep candidate answers in output json", default=True)
    args = parser.parse_args()

    pattern = os.path.join(args.responses_dir, f"{args.data}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No response files found for pattern {pattern}")
    print(f"[INFO] Found {len(files)} files: \n  - " + "\n  - ".join(os.path.basename(f) for f in files))

    # load
    all_responses: List[List[Dict[str, str]]] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            all_responses.append(json.load(fh))

    # assume same order / same questions across files – sanity check
    questions = [item["question"] for item in all_responses[0]]
    n_q = len(questions)
    for idx, resp in enumerate(all_responses[1:], 1):
        if len(resp) != n_q:
            raise ValueError(f"File {files[idx]} has {len(resp)} answers, expected {n_q}")
        # Further check by question text
        for i, q in enumerate(questions):
            if resp[i]["question"] != q:
                raise ValueError(f"Question mismatch in {files[idx]} at index {i}")

    print(f"[INFO] {n_q} questions detected.")

    # --- load selector model ------------------------------------------------
    sel_model_name = available_models[args.selector_model]
    print(f"[INFO] Loading selector model {sel_model_name} …")
    selector, tokenizer = load_model(sel_model_name, verbose=False)

    final_items = []

    for i, q in enumerate(questions):
        candidates = [resp[i]["answer"] for resp in all_responses]
        # quick majority vote shortcut (cheap) – if strict majority exists, skip LLM
        maj = majority_vote(candidates)
        if candidates.count(maj) > len(candidates) // 2:
            chosen = maj
        else:
            prompt = build_selector_prompt(q, candidates)
            chosen = select_answer(selector, tokenizer, prompt)
        item = {
            "question": q,
            "final_answer": chosen,
        }
        if args.include_candidates:
            item["candidate_answers"] = candidates
        final_items.append(item)
        print(f"\nQ{i+1}: {q}\n→ {chosen}\n")

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{args.data}_ENSEMBLE_{args.selector_model}_{ts}.json"
    out_path = os.path.join(args.outdir, out_name)
    os.makedirs(args.outdir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(final_items, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] Ensemble answers saved to {out_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()