from dataclasses import dataclass
from typing import Callable

from models import AbstractModel, LLAMAInstruct, LLAMADefault, DeepSeekR1GGUF


@dataclass
class Instance:
    model: AbstractModel
    prompt_generator: Callable[[str, str], str]
    prompt: str


llama32_3b_instruct = Instance(
    model=LLAMAInstruct(
        MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
        system_prompt="You are a helpful assistant.",
        max_new_tokens=128,
    ),
    prompt_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}",
    prompt="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else.
""".strip()
)

llama32_1b = Instance(
    model=LLAMADefault(
        MODEL_ID="meta-llama/Llama-3.2-1B",
        max_new_tokens=128,
    ),
    prompt_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}",
    prompt="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else. 
""".strip()
)

deepseek_r1_gguf_14b_q4_k_l = Instance(
    model=DeepSeekR1GGUF(
        MODEL_ID="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        GGUF_FILE="DeepSeek-R1-Distill-Qwen-14B-Q4_K_L.gguf",
        max_new_tokens=2048,
        system_prompt="You are a helpful assistant."
    ),
    prompt_generator=lambda docs, prompt: f"TEXT:\n{docs}\n\n{prompt}",
    prompt="""
Based on previous TEXT generate a single QUESTION for an exam which is answerable using only the information in the TEXT.
Do not answer the question.
Respond only with the QUESTION and nothing else.
""".strip()
)
