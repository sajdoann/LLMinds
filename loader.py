import gc
import json
import datetime

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

QUESTIONS_PER_DOCUMENT = 5

MAX_CHARS = 1024 * 16


def main():
    for instance in INSTANCES:
        llm = instance.model.llm_factory()
        print(f"Using model: {instance.model.MODEL_ID}")

        responses: dict[str, list[dict[str, str]]] = {}
        for i, (folder, txt_file) in enumerate(tqdm.tqdm(FOLDERS)):
            txt_data = load_data(folder, txt_file)
            if not txt_data:
                print(f"Warning: No data found in {folder} with name {txt_file}")
                continue

            for txt_path, txt in txt_data:
                print(f"Processing {txt_path}")

                if len(txt) > MAX_CHARS:
                    print(
                        f"Warning: input text of {txt_path} is too long, truncating from {len(txt)} to last {MAX_CHARS} characters.")
                    txt = txt[-MAX_CHARS:]

                response_questions = llm(
                    instance.prompt_question_generator(txt, instance.prompt_question),
                    QUESTIONS_PER_DOCUMENT,
                )
                response_answers = [
                    llm(
                        instance.prompt_answer_generator(
                            txt,
                            question.strip(),
                            instance.prompt_answer,
                        ),
                        1
                    )[0]
                    for question in response_questions
                ]

                responses[str(txt_path)[len("testset/"):]] = [
                    {
                        "question": question,
                        "reference-answer": answer,
                    }
                    for question, answer in zip(
                        response_questions,
                        response_answers,
                        strict=True
                    )
                ]

            with open(
                    (
                            f"responses"
                            f"_{instance.model.MODEL_ID.replace('/', '-')}"
                            f"_{datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")}"
                            f".json"
                    ),
                    "w",
            ) as f:
                json.dump(responses, f, ensure_ascii=False, indent=4)

        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
