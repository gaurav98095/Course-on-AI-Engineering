"""Three ways to tell a transformer where a token is -- no GPU needed for
any of it, this is pure math on small vectors.

Part 1 implements RoPE (the course model's real mechanism) and verifies,
numerically, the one property that makes it work: the attention score
between a rotated query and a rotated key depends on their RELATIVE
distance only, never on their absolute positions.

Part 2 implements ALiBi's explicit distance penalty for comparison --
a completely different philosophy (bias the score directly, don't touch
the vectors at all).

Part 3 shows why naively running RoPE past its trained context length is
risky: some rotation frequencies wrap around thousands of times over a
long context (always "locally periodic," safe to extrapolate); others
barely turn at all (globally informative, genuinely out-of-distribution
past the trained range) -- the exact asymmetry YaRN's frequency-dependent
rescaling is built to fix.

Run:  python positional_encodings.py
"""

import numpy as np

HEAD_DIM = 128          # course model's real head_dim (config.json)
ROPE_THETA = 5_000_000  # course model's real rope_theta (config.json)
MAX_POS = 262_144       # course model's real max_position_embeddings (config.json)


def rope_freqs(head_dim: int, theta: float) -> np.ndarray:
    """One rotation frequency per 2D pair of dimensions -- lower pairs rotate
    fast (local detail), higher pairs rotate slow (long-range position)."""
    i = np.arange(0, head_dim, 2)
    return 1.0 / (theta ** (i / head_dim))


def rope_rotate(x: np.ndarray, pos: int, freqs: np.ndarray) -> np.ndarray:
    angles = pos * freqs
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * cos - x2 * sin
    out[1::2] = x1 * sin + x2 * cos
    return out


def relative_position_demo() -> None:
    rng = np.random.default_rng(0)
    freqs = rope_freqs(HEAD_DIM, ROPE_THETA)
    q, k = rng.standard_normal(HEAD_DIM), rng.standard_normal(HEAD_DIM)

    def score(m: int, n: int) -> float:
        return float(rope_rotate(q, m, freqs) @ rope_rotate(k, n, freqs))

    print(f"--- 1. RoPE: the score depends only on (m - n) ---")
    print(f"(head_dim={HEAD_DIM}, theta={ROPE_THETA:,} -- our course model's real config)\n")

    same_gap_0 = [(0, 0), (5, 5), (100, 100), (50_000, 50_000)]
    print("pairs with m - n = 0:")
    for m, n in same_gap_0:
        print(f"  m={m:>7,} n={n:>7,}  score={score(m, n):.6f}")

    same_gap_7 = [(10, 3), (110, 103), (5_010, 5_003), (100_007, 100_000)]
    print("\npairs with m - n = 7 (note: identical score, every time):")
    for m, n in same_gap_7:
        print(f"  m={m:>7,} n={n:>7,}  score={score(m, n):.6f}")


def alibi_slopes(n_heads: int) -> np.ndarray:
    """Press et al. 2021: m_i = 1 / 2^((8/h) * i), i = 1..h -- a geometric
    sequence of penalty strengths, one per head."""
    i = np.arange(1, n_heads + 1)
    return 1.0 / (2.0 ** ((8.0 / n_heads) * i))


def alibi_demo() -> None:
    n_heads = 8
    slopes = alibi_slopes(n_heads)
    print("\n--- 2. ALiBi: an explicit distance penalty, added to the score directly ---")
    print(f"({n_heads} heads' worth of slopes, Press et al.'s formula)\n")
    for h, m in enumerate(slopes):
        penalty_at_100 = m * 100
        print(f"  head {h}: slope={m:.5f}  penalty at distance 100 tokens = -{penalty_at_100:.3f}")
    print("\nNo rotation, no relative-position algebra -- just score -= slope * |i - j|. "
          "Steeper-slope heads go local; shallow-slope heads stay global.")


def extrapolation_risk_demo() -> None:
    freqs = rope_freqs(HEAD_DIM, ROPE_THETA)
    fastest, slowest = freqs[0], freqs[-1]

    angle_fast = MAX_POS * fastest
    angle_slow = MAX_POS * slowest

    print("\n--- 3. Why extrapolation risk isn't the same for every dimension ---")
    print(f"(our model's real config: theta={ROPE_THETA:,}, max_position_embeddings={MAX_POS:,})\n")
    print(f"fastest-rotating pair:  {angle_fast / (2*np.pi):,.1f} full rotations over the trained context "
          f"-- always locally periodic, safe to extrapolate")
    print(f"slowest-rotating pair:  {angle_slow / (2*np.pi):.4f} full rotations over the trained context "
          f"-- has barely turned at all; every position past training is a genuinely new angle")

    # a smaller, more typical config makes the same asymmetry easier to feel
    theta_small, max_pos_small = 10_000, 4096
    freqs_small = rope_freqs(HEAD_DIM, theta_small)
    angle_slow_small = max_pos_small * freqs_small[-1]
    angle_slow_2x = (max_pos_small * 2) * freqs_small[-1]
    print(f"\nfor scale: theta=10,000, max_pos=4,096 (a smaller, more typical config) --")
    print(f"  slowest pair's angle at the trained max: {angle_slow_small / (2*np.pi):.4f} rotations")
    print(f"  the SAME dimension at 2x that length:    {angle_slow_2x / (2*np.pi):.4f} rotations")
    print("  -> a genuinely new angle the network never trained on. This is exactly the gap "
          "YaRN closes: rescale slow dimensions back into the trained range, leave fast ones alone.")


def main() -> None:
    relative_position_demo()
    alibi_demo()
    extrapolation_risk_demo()


if __name__ == "__main__":
    main()
