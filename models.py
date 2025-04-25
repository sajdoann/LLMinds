import abc
from dataclasses import dataclass
from typing import Callable
from llama_cpp import Llama
from transformers import (
    pipeline
)


class AbstractModel(abc.ABC):
    MODEL_ID: str

    @abc.abstractmethod
    def llm_factory(self) -> Callable[[str, int], list[str]]:
        pass


@dataclass
class DeepSeekR1GGUF(AbstractModel):
    MODEL_ID: str = "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"
    GGUF_FILE: str = "DeepSeek-R1-Distill-Qwen-14B-Q4_K_L.gguf"
    max_new_tokens: int = 2048
    system_prompt: str = "You are a helpful assistant."

    def llm_factory(self) -> Callable[[str, int], list[str]]:
        context_window = 1024 * 16
        llm = Llama.from_pretrained(
            n_gpu_layers=-1,
            repo_id=self.MODEL_ID,
            filename=self.GGUF_FILE,
            n_ctx=context_window,
            verbose=False
        )

        def _llm(input_text: str, count: int) -> list[str]:
            responses = []
            for _ in range(count):
                input_tokens = llm.tokenize(bytes(input_text, "utf-8"))
                if len(input_tokens) > context_window - self.max_new_tokens:
                    print(f"Warning: input text is too long, truncating from {len(input_tokens)} to {context_window - self.max_new_tokens} tokens.")
                    input_text = llm.detokenize(input_tokens[-(context_window - self.max_new_tokens):]).decode("utf-8", errors="ignore")

                response = llm(
                    f'<｜begin▁of▁sentence｜>{self.system_prompt}<｜User｜>{input_text}<｜Assistant｜><think>',
                    max_tokens=self.max_new_tokens + len(input_tokens),
                    echo=False,
                    stop=[
                        '<｜User｜>',
                        '<｜Assistant｜>',
                        '<｜end▁of▁sentence｜>'
                    ],
                )
                text = response['choices'][0]['text']
                last_think = text.rfind('</think>')
                responses.append((text[last_think + len('</think>'):] if last_think != -1 else text).strip())
            return responses
        return _llm


@dataclass
class LLAMAInstruct(AbstractModel):
    MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
    system_prompt: str = "You are a helpful assistant."
    max_new_tokens: int = 256

    def llm_factory(self) -> Callable[[str, int], list[str]]:
        gen_pipeline = pipeline(
            "text-generation",
            model=self.MODEL_ID,
        )
        eos_tokens = [
            t for t in
            (
                gen_pipeline.tokenizer.eos_token_id,
                gen_pipeline.tokenizer.convert_tokens_to_ids(""),
            )
            if t is not None
        ]

        def _llm(input_text: str, count: int) -> list[str]:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ]
            outputs = gen_pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=count,
                eos_token_id=eos_tokens,
                pad_token_id=gen_pipeline.tokenizer.eos_token_id,
            )
            return [x["generated_text"][-1]['content'] for x in outputs]
        return _llm


@dataclass
class LLAMADefault(AbstractModel):
    MODEL_ID: str = "meta-llama/Llama-3.2-1B",
    max_new_tokens: int = 256

    def llm_factory(self) -> Callable[[str, int], list[str]]:
        gen_pipeline = pipeline(
            "text-generation",
            model=self.MODEL_ID
        )
        eos_tokens = [
            t for t in
            (
                gen_pipeline.tokenizer.eos_token_id,
                gen_pipeline.tokenizer.convert_tokens_to_ids(""),
            )
            if t is not None
        ]

        def _llm(input_text: str, count: int) -> list[str]:
            outputs = gen_pipeline(
                input_text,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=count,
                eos_token_id=eos_tokens,
                pad_token_id=gen_pipeline.tokenizer.eos_token_id,
            )
            return [x["generated_text"][len(input_text):] for x in outputs]
        return _llm
