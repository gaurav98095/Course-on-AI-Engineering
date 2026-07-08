"""KV cache: prove it exists by turning it off, then measure its real size.

Two experiments:
  1. Generate N tokens WITH the KV cache vs WITHOUT it, and time both — the
     gap between them IS the KV cache's entire contribution.
  2. Read the model's real layer/head config and compute the theoretical
     cache size in bytes, then check it against actual GPU memory growth.

Run:  python kv_cache.py
"""

import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = ("Explain, in detail, how an aircraft's pitot-static system works, "
          "and why it can fail in icing conditions.")
N_TOKENS = 60   # kept short on purpose — the no-cache run is O(n^2), it gets slow fast


def cache_size_bytes(cfg, tokens: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x tokens x batch x bytes/element."""
    text_cfg = getattr(cfg, "text_config", cfg)
    L, H, D = text_cfg.num_hidden_layers, text_cfg.num_key_value_heads, text_cfg.head_dim
    return 2 * L * H * D * tokens * batch * dtype_bytes


@torch.inference_mode()
def timed_generate(model, inputs, max_new_tokens: int, use_cache: bool) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=use_cache, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    print(f"loading {MODEL} ...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)

    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"layers={text_cfg.num_hidden_layers}  "
          f"kv_heads={text_cfg.num_key_value_heads} (query heads: {text_cfg.num_attention_heads})  "
          f"head_dim={text_cfg.head_dim}\n")

    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"prompt: {prompt_tokens} tokens\n")

    print(f"generating {N_TOKENS} tokens WITH the KV cache (the normal way)...")
    t_cache = timed_generate(model, inputs, N_TOKENS, use_cache=True)
    print(f"  {t_cache:.2f}s  ({N_TOKENS / t_cache:.1f} tok/s)\n")

    print(f"generating {N_TOKENS} tokens WITHOUT it (recomputes the whole sequence, every step)...")
    t_nocache = timed_generate(model, inputs, N_TOKENS, use_cache=False)
    print(f"  {t_nocache:.2f}s  ({N_TOKENS / t_nocache:.1f} tok/s)\n")

    print(f"KV cache speedup at {N_TOKENS} generated tokens: {t_nocache / t_cache:.1f}x\n")

    print("--- cache size: theory vs measured ---")
    theory = cache_size_bytes(cfg, tokens=prompt_tokens + N_TOKENS)
    print(f"theoretical cache size at {prompt_tokens + N_TOKENS} tokens: {theory / 2**20:.1f} MiB")

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model.generate(**inputs, max_new_tokens=N_TOKENS, use_cache=True, do_sample=False)
    grew_by = torch.cuda.max_memory_allocated() - baseline
    print(f"measured extra GPU memory for that generation: {grew_by / 2**20:.1f} MiB "
          f"(cache + activations + generation bookkeeping — not pure cache, but close)")


if __name__ == "__main__":
    main()
