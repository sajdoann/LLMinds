import gc
import json
import datetime
from collections import defaultdict

import torch
import tqdm

from instances import Instance, llama32_3b_instruct, llama32_1b, deepseek_r1_gguf_14b_q4_k_l
from data_loader import load_data, list_subfolders

FOLDERS: list[tuple[str, str]] = [
    ("devset/ukrbiology/book01/topic01-Різноманітність тварин", "text.en.txt"),
    ("devset/ukrbiology/book01/topic02-Процеси життєдіяльностітварин", "text.en.txt"),
    ("devset/ukrbiology/book01/topic03-Поведінка тварин", "text.en.txt"),
    ("devset/popular/video-22", "text.en.txt"),
    *((str(x), "text.txt") for x in list_subfolders("devset/nmtclass/lecture01-eval")),
]

INSTANCES: list[Instance] = [
    llama32_3b_instruct,
    llama32_1b,
    deepseek_r1_gguf_14b_q4_k_l,
]

QUESTIONS_PER_DOCUMENT = 3

MAX_CHARS = 1024 * 16

def main():
    responses: dict[str, dict[int, list[str]]] = defaultdict(dict)

    for instance in INSTANCES:
        llm = instance.model.llm_factory()
        print(f"Using model: {instance.model.MODEL_ID}")

        for i, (folder, txt_file) in enumerate(tqdm.tqdm(FOLDERS)):
                txt_data = load_data(folder, txt_file)
                if not txt_data:
                    print(f"Warning: No data found in {folder} with name {txt_file}")
                    continue

                txt = "\n".join(txt_data)
                if len(txt) > MAX_CHARS:
                    print(f"Warning: input text of {folder} / {txt_file} is too long, truncating from {len(txt)} to last {MAX_CHARS} characters.")
                    txt = txt[-MAX_CHARS:]

                response = llm(instance.prompt_generator(txt, instance.prompt), QUESTIONS_PER_DOCUMENT)

                responses[instance.model.MODEL_ID][i] = response

                gc.collect()
                torch.cuda.empty_cache()

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    with open(f"responses_{datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")}.json", "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
