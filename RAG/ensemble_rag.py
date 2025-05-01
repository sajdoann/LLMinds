#!/usr/bin/env python3

import argparse, glob, json, os, re, gc, torch
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple

from model_loader import load_model
from rag_system   import available_models


def _normalise(text: str) -> str:
    """Cheap normalisation for majority voting / deduplication."""
    return re.sub(r"\s+", " ", text.strip().lower())

def majority_vote(cands: List[str]) -> Tuple[str, int]:
    """Return (answer, count) for the most frequent normalised answer."""
    if not cands:
        return "", 0
    normed = [_normalise(c) for c in cands]
    best, cnt = Counter(normed).most_common(1)[0]
    for c in cands:
        if _normalise(c) == best:
            return c.strip(), cnt
    return cands[0].strip(), cnt # fallback

def build_prompt(q: str, cands: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
    return (
        "SYSTEM: You are a strict grader. Given a QUESTION and several "
        "CANDIDATE ANSWERS, reply with ONLY the number of the best answer "
        "(1-N). No other text.\n\n"
        f"QUESTION: {q.strip()}\n\nCANDIDATE ANSWERS:\n{numbered}\n\n"
        "BEST ANSWER NUMBER:"
    )

def select_index(model, tok, prompt: str, n_cands: int) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 3,
            do_sample = False, # deterministic
            pad_token_id = tok.eos_token_id,
        )
    txt = tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r"(\d+)", txt)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= n_cands:
            return idx - 1
    return -1 # invalid / fallback

def unique_candidates(cands: List[str],
                      max_keep: int = 8,
                      max_len:  int  = 300
                      ) -> Tuple[List[str], Dict[str, str]]:
    """
    Deduplicate, shorten and cap candidates.

    Returns
    -------
    short_list : Strings fed to the selector.
    mapping : dict[str -> str]
        `_normalise(short)` -> full original answer.
    """
    # keep the shortest representative of each normalised bucket
    bucket = {}
    for c in cands:
        k = _normalise(c)
        if k not in bucket or len(c) < len(bucket[k]):
            bucket[k] = c
    full_keep = list(bucket.values())[:max_keep]

    short_list, mapping = [], {}
    for full in full_keep:
        short = full if len(full) <= max_len else (full[:max_len] + "…")
        short_list.append(short)
        mapping[_normalise(short)] = full
    return short_list, mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--selector_model", default="llama-7b",
                    choices=available_models.keys())
    ap.add_argument("--responses_dir", default="responses")
    ap.add_argument("--outdir",        default="responses")
    ap.add_argument("--include_candidates", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.responses_dir,
                                          f"{args.data}_*.json")))
    if not files:
        raise FileNotFoundError(f"No response files for prefix {args.data}")
    print(f"[INFO] Ensemble of {len(files)} systems.")

    runs  = [json.load(open(f, encoding="utf-8")) for f in files]
    qlist = [item["question"] for item in runs[0]]
    for r in runs[1:]:
        assert len(r) == len(qlist)
        assert all(r[i]["question"] == qlist[i] for i in range(len(qlist)))

    model_name = available_models[args.selector_model]
    selector, tok = load_model(model_name, verbose=False)
    print(f"[INFO] Selector loaded: {model_name}")

    out = []
    for i, q in enumerate(qlist, 1):
        cands_full = [r[i-1]["answer"] for r in runs]  # 0-based access
        best, cnt  = majority_vote(cands_full)

        # strict majority -> accept
        if cnt > len(cands_full)//2 and best:
            chosen = best
        else:
            short_cands, short2full = unique_candidates(cands_full)
            prompt = build_prompt(q, short_cands)
            idx    = select_index(selector, tok, prompt, len(short_cands))
            if idx != -1:
                key    = _normalise(short_cands[idx])
                chosen = short2full.get(key, short_cands[idx])
            else:                       # fallback to majority / first
                chosen, _ = majority_vote(cands_full)
                if not chosen:
                    chosen = cands_full[0] if cands_full else ""

        out_item = {"question": q, "final_answer": chosen.strip()}
        if args.include_candidates:
            out_item["candidate_answers"] = cands_full
        out.append(out_item)

        print(f"Q{i}/{len(qlist)} → {chosen[:120]}{'…' if len(chosen)>120 else ''}")

    # save
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{args.data}_ENSEMBLE_{args.selector_model}_{ts}.json"
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, name), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(out)} answers → {name}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    main()
