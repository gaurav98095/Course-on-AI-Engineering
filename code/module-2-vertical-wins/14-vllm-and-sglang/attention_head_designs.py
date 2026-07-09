"""Four ways to share (or not share) K/V heads across query heads, and
what each one actually costs in KV cache -- no GPU needed for either half
of this script.

Part 1 reproduces transformers' real `repeat_kv` function verbatim and
uses it to show exactly what GQA stores versus what it computes with --
the whole savings mechanism in eleven lines.

Part 2 extends Lecture 05's cache formula to all four head designs (MHA,
GQA, MQA, MLA) and computes real numbers for our own course model's
dimensions, plus a labeled hypothetical MLA configuration using
DeepSeek-V2's own published ratios (our model doesn't use MLA -- Qwen3-VL
uses GQA -- so this is "what if", not a measured fact about our model).

Run:  python attention_head_designs.py
"""

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Verbatim from transformers' modeling_llama.py -- the actual GQA mechanism.
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep). Goes from
    (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_demo() -> None:
    torch.manual_seed(0)
    B, H_KV, N_REP, SEQ, D = 1, 8, 4, 5, 128   # our course model's real shape: 8 KV heads, x4 -> 32 query heads

    k_cache = torch.randn(B, H_KV, SEQ, D)   # what's ACTUALLY stored in the KV cache
    k_expanded = repeat_kv(k_cache, N_REP)   # what the attention dot product actually uses

    print("--- 1. What repeat_kv actually does ---")
    print(f"stored (cache) K shape:     {tuple(k_cache.shape)}  ({k_cache.numel():,} elements)")
    print(f"expanded (compute) K shape: {tuple(k_expanded.shape)}  ({k_expanded.numel():,} elements)")
    print(f"{k_expanded.numel() / k_cache.numel():.0f}x more elements at COMPUTE time, "
          f"zero extra bytes ever written to the CACHE\n")

    matches = all(torch.equal(k_expanded[0, i], k_cache[0, 0]) for i in range(N_REP))
    print(f"query heads 0-3 all see source KV head 0's identical vector: {matches}")
    print("This -- and nothing more exotic -- is what 'grouped' means in grouped-query attention.\n")


def mha_bytes(n_heads: int, head_dim: int, layers: int, precision_bytes: int) -> int:
    return 2 * n_heads * head_dim * layers * precision_bytes


def gqa_bytes(n_groups: int, head_dim: int, layers: int, precision_bytes: int) -> int:
    return 2 * n_groups * head_dim * layers * precision_bytes   # MQA is just n_groups=1


def mla_bytes(latent_dim: int, rope_dim: int, layers: int, precision_bytes: int) -> int:
    return (latent_dim + rope_dim) * layers * precision_bytes   # one shared latent, no factor of 2 or n_heads


def cache_comparison() -> None:
    L, N_H, D_H, S = 36, 32, 128, 2   # course model's real config.json: layers, query heads, head_dim, bf16 bytes
    N_G_ACTUAL = 8                     # course model's REAL kv_heads (Lecture 05) -- this is GQA, measured fact

    mha = mha_bytes(N_H, D_H, L, S)
    gqa = gqa_bytes(N_G_ACTUAL, D_H, L, S)
    mqa = gqa_bytes(1, D_H, L, S)

    # hypothetical: our model doesn't use MLA. Scaled using DeepSeek-V2's own published ratios
    # (d_c ~ 4x head_dim, decoupled RoPE dim ~ 0.5x head_dim) applied to OUR head_dim, purely illustrative.
    d_c, d_h_r = 4 * D_H, D_H // 2
    mla_hypothetical = mla_bytes(d_c, d_h_r, L, S)

    print("--- 2. KV cache per token, all four designs, our course model's dimensions ---")
    print(f"(layers={L}, query_heads={N_H}, head_dim={D_H}, bf16)\n")
    rows = [
        ("MHA (32 separate KV heads)", mha, "hypothetical -- our model does not use this"),
        ("GQA (8 groups)", gqa, "ACTUAL -- Qwen3-VL-8B-Instruct's real config"),
        ("MQA (1 shared group)", mqa, "hypothetical"),
        ("MLA (DeepSeek-V2 ratios, hypothetical)", mla_hypothetical, "hypothetical -- our model does not use this"),
    ]
    for name, b, note in rows:
        print(f"{name:42s} {b:>8,} B/token = {b / 1024:6.1f} KiB/token   [{note}]")

    print(f"\nGQA vs MHA: {mha / gqa:.1f}x smaller -- matches Lecture 05's '4x smaller cache' claim exactly")
    print(f"MQA vs GQA: {gqa / mqa:.1f}x smaller again")
    print(f"MLA vs GQA (hypothetical): {gqa / mla_hypothetical:.2f}x -- MLA's real pitch isn't 'smallest cache', "
          f"it's 'MHA-level quality at close to MQA-level memory'")


def main() -> None:
    repeat_kv_demo()
    cache_comparison()


if __name__ == "__main__":
    main()
