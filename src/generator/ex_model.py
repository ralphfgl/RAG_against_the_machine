"""Load and manage the Qwen3-0.6B model."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"

_model = None
_tokenizer = None


def _load():
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
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    _model.eval()
    return _model, _tokenizer


def generate(
    prompt: str, max_new_tokens: int = 256, temperature: float = 0.0
) -> str:
    """Generate a completion for the prompt.
    Args:
        prompt: The full prompt string.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: 0.0 = greedy  decoding (deterministic).
    Returns:
        The generated text (without the prompt).
    """

    model, tokenizer = _load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
