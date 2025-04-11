import torch
from dataclasses import dataclass
from typing import Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)


@dataclass
class QwenConfig:
    MODEL_ID: str = "Qwen/QwQ-32B"
    max_new_tokens: int = 512
    device_map: str = "auto"
    torch_dtype: str = "auto"
    add_generation_prompt: bool = True


@dataclass
class Llama31Config:
    MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 256
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    system_prompt: str = "You are a helpful assistant."


@dataclass
@dataclass
class Llama32_1BConfig:
    MODEL_ID: str = "meta-llama/Llama-3.2-1B"
    max_new_tokens: int = 256
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


@dataclass
class Llama32_3BConfig:
    MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_new_tokens: int = 256
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    system_prompt: str = "You are a helpful assistant."


@dataclass
class QwenGGUFConfig:
    MODEL_ID: str = "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"
    GGUF_FILE: str = "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"
    max_new_tokens: int = 512
    torch_dtype: torch.dtype = torch.float16
    system_prompt: str = "You are a helpful assistant."


def _generate_text_qwen(input_text: str, config: QwenConfig) -> str:
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map
    )

    messages = [
        {"role": "user", "content": input_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=config.add_generation_prompt
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_tokens
    )

    # Remove the prompt portion so we only get the generated text
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def _generate_text_llama31(input_text: str, config: Llama31Config) -> str:
    gen_pipeline = pipeline(
        "text-generation",
        model=config.MODEL_ID,
        model_kwargs={"torch_dtype": config.torch_dtype},
        device_map=config.device_map
    )
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": input_text}
    ]
    outputs = gen_pipeline(
        messages,
        max_new_tokens=config.max_new_tokens
    )
    return outputs[0]["generated_text"]


def _generate_text_llama32_1B(input_text: str, config: Llama32_1BConfig) -> str:
    gen_pipeline = pipeline(
        "text-generation",
        model=config.MODEL_ID,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map
    )
    outputs = gen_pipeline(
        input_text,
        max_new_tokens=config.max_new_tokens
    )
    return outputs[0]["generated_text"]


def _generate_text_llama32_3B(input_text: str, config: Llama32_3BConfig) -> str:
    gen_pipeline = pipeline(
        "text-generation",
        model=config.MODEL_ID,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map
    )
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": input_text}
    ]
    outputs = gen_pipeline(messages, max_new_tokens=config.max_new_tokens)
    return outputs[0]["generated_text"]


def _generate_text_qwen_gguf(input_text: str, config: QwenGGUFConfig) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        gguf_file=config.GGUF_FILE
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        gguf_file=config.GGUF_FILE,
        torch_dtype=config.torch_dtype
    )
    prompt_text = f"<|begin▁of▁sentence|>{config.system_prompt}<｜User｜>{input_text}<|Assistant|>"
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def generate_text(input_text: str, config: Union[
    QwenConfig,
    Llama31Config,
    Llama32_1BConfig,
    Llama32_3BConfig,
    QwenGGUFConfig
]) -> str:
    """
    Generate text using the specified model configuration.

    Example usage:
    ```python
    qwen_cfg = QwenConfig()
    llama31_cfg = Llama31Config()
    llama32_cfg = Llama32Config()
    qwen_gguf_cfg = QwenGGUFConfig()

    response = generate_text("Hello!", qwen_cfg)
    print("Qwen response:\n", response)

    response = generate_text("Who are you?", llama31_cfg)
    print("Llama 3.1 response:\n", response)
    ```
    :param input_text: The input text to generate a response for.
    :param config: The configuration object for the model to use.
    :return: The generated text response.
    """
    if isinstance(config, QwenConfig):
        return _generate_text_qwen(input_text, config)
    elif isinstance(config, Llama31Config):
        return _generate_text_llama31(input_text, config)
    elif isinstance(config, Llama32_1BConfig):
        return _generate_text_llama32_1B(input_text, config)
    elif isinstance(config, Llama32_3BConfig):
        return _generate_text_llama32_3B(input_text, config)
    elif isinstance(config, QwenGGUFConfig):
        return _generate_text_qwen_gguf(input_text, config)
    else:
        raise ValueError("Unsupported config type passed to generate_text.")
