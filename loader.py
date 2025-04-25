import gc
import json
import datetime
from collections import defaultdict

import torch
import tqdm

from instances import Instance, llama32_3b_instruct, llama32_1b, deepseek_r1_gguf_14b_q4_k_l
from data_loader import load_data, list_subfolders

FOLDERS_DEVSET: list[tuple[str, str]] = [
    *((str(x), "text.txt") for x in list_subfolders("devset/nmtclass")),
    *((str(x), "text.en.txt") for x in list_subfolders("devset/popular")),
    *((str(x), "text.en.txt") for x in list_subfolders("devset/ukrbiology")),
]

FOLDERS_TESTSET: list[tuple[str, str]] = [
    *((str(x), "text.en.txt") for x in list_subfolders("testset/demagog-statements-public")),
    *((str(x), "text.txt") for x in list_subfolders("testset/flat-earth-book")),
    *((str(x), "text.txt") for x in list_subfolders("testset/nmt-book")),
    *((str(x), "text.txt") for x in list_subfolders("testset/nmt-class")),
    *((str(x), "text.en.txt") for x in list_subfolders("testset/popular")),
    *((str(x), "text.en.txt") for x in list_subfolders("testset/ukr-biology")),
    *((str(x), "text.txt") for x in list_subfolders("testset/world-history")),
]

FOLDERS: list[tuple[str, str]] = FOLDERS_TESTSET

INSTANCES: list[Instance] = [
    llama32_3b_instruct,
    llama32_1b,
    deepseek_r1_gguf_14b_q4_k_l,
]

QUESTIONS_PER_DOCUMENT = 3

MAX_CHARS = 1024 * 16

def main():
    responses: dict[str, list[str]] = {}

    for instance in INSTANCES:
        llm = instance.model.llm_factory()
        print(f"Using model: {instance.model.MODEL_ID}")

        for i, (folder, txt_file) in enumerate(tqdm.tqdm(FOLDERS)):
                txt_data = load_data(folder, txt_file)
                if not txt_data:
                    print(f"Warning: No data found in {folder} with name {txt_file}")
                    continue

                for txt_path, txt in txt_data:
                    print(f"Processing {txt_path}")

                    if len(txt) > MAX_CHARS:
                        print(f"Warning: input text of {txt_path} is too long, truncating from {len(txt)} to last {MAX_CHARS} characters.")
                        txt = txt[-MAX_CHARS:]

                    response = llm(instance.prompt_generator(txt, instance.prompt), QUESTIONS_PER_DOCUMENT)

                    responses[str(txt_path)] = response

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"responses_{datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")}.json", "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
