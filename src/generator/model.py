"""Load and manage the Qwen3-0.6B model."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from typing import Tuple

MODEL_NAME = "Qwen/Qwen3-0.6B"

_model = None
_tokenizer = None


def _load() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the model and tokenizer (once)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    _model.eval()

    return _model, _tokenizer


def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Generate a completion from the prompt.

    The prompt is wrapped in Qwen3's chat template so the model emits the
    proper end-of-turn token. Repetition penalties prevent loops.
    """
    model, tokenizer = _load()

    # Wrap the prompt in Qwen3's chat template. This is critical:
    # without it, the model doesn't know when to stop and loops.
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # disable Qwen3's "thinking" mode for speed
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            # used for stopping the loop
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]  # noqa: E203
    text: str = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()
