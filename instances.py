from dataclasses import dataclass
from typing import Callable

from models import AbstractModel, LLAMAInstruct, LLAMADefault, DeepSeekR1GGUF


@dataclass
class Instance:
    model: AbstractModel
    prompt_question_generator: Callable[[str, str], str]
    prompt_question: str
    prompt_answer_generator: Callable[[str, str, str], str]
    prompt_answer: str


llama32_3b_instruct = Instance(
    model=LLAMAInstruct(
        MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=128,
    ),
    prompt_question_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}",
    prompt_answer_generator=lambda docs, question, prompt: f"TEXT:\n{docs}\n\nQUESTION:\n{question}\n\n{prompt}",
    prompt_question="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else.
""".strip(),
    prompt_answer="""
Based on previous TEXT generate an ANSWER for an the question. Answer only using the information in the TEXT.
Respond only with the ANSWER and nothing else.
""".strip(),
)

llama32_1b = Instance(
    model=LLAMADefault(
        MODEL_ID="meta-llama/Llama-3.2-1B",
        max_new_tokens=128,
    ),
    prompt_question_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}\n\nQUESTION: ",
    prompt_answer_generator=lambda docs, question, prompt: f"TEXT:\n{docs}\n\nQUESTION:\n{question}\n\n{prompt}\n\nANSWER: ",
    prompt_question="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else. 
""".strip(),
    prompt_answer="""
Based on previous TEXT generate an ANSWER for an the question. Answer only using the information in the TEXT.
Respond only with the ANSWER and nothing else.
QUESTION:
""".strip(),
)

deepseek_r1_gguf_14b_q4_k_l = Instance(
    model=DeepSeekR1GGUF(
        MODEL_ID="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        GGUF_FILE="DeepSeek-R1-Distill-Qwen-14B-Q4_K_L.gguf",
        max_new_tokens=2048,
    ),
    prompt_question_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}",
    prompt_answer_generator=lambda docs, question, prompt: f"TEXT:\n{docs}\n\nQUESTION:\n{question}\n\n{prompt}",
    prompt_question="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else.
""".strip(),
    prompt_answer="""
Based on previous TEXT generate an ANSWER for an the question. Answer only using the information in the TEXT.
Respond only with the ANSWER and nothing else.
""".strip(),
)
