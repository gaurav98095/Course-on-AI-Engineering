"""Static batching vs continuous batching: same GPU, same requests, same
total work -- radically different wall time. No GPU needed, this is a
scheduling simulation, not a benchmark.

Static batching: group requests into fixed batches of size B, and don't
start the next batch until every request in the current one has finished
-- so a batch's wall time is dictated by its single longest request, and
every slot that finished early just sits idle until the stragglers catch up.

Continuous batching: at every decode step, evict any request that just
finished and immediately backfill the next one waiting in the queue --
the batch never drains, it just keeps rotating.

Run:  python continuous_batching_simulator.py
"""

import random

N_REQUESTS = 200
BATCH_SIZE = 8
SEED = 42


def sample_length(rng: random.Random) -> int:
    """A realistic traffic mix: mostly short answers, some long ones -- the
    same shape as every other lecture's traffic-mix simulation in this course."""
    r = rng.random()
    if r < 0.70:
        return rng.randint(20, 100)
    elif r < 0.95:
        return rng.randint(100, 400)
    return rng.randint(400, 1000)


def static_batching(lengths: list[int], batch_size: int) -> tuple[int, int]:
    """Fixed groups of batch_size; a group's wall time is its slowest member's length."""
    wall_steps, slot_steps = 0, 0
    for i in range(0, len(lengths), batch_size):
        group = lengths[i:i + batch_size]
        group_time = max(group)
        wall_steps += group_time
        slot_steps += group_time * len(group)   # every slot "occupied" for the whole group
    return wall_steps, slot_steps


def continuous_batching(lengths: list[int], batch_size: int) -> tuple[int, int]:
    """A rotating window of batch_size active requests, backfilled the instant one finishes."""
    queue = list(lengths)
    active = [queue.pop(0) for _ in range(min(batch_size, len(queue)))]

    wall_steps, slot_steps = 0, 0
    while active:
        wall_steps += 1
        slot_steps += batch_size
        still_active = []
        for remaining in active:
            remaining -= 1
            if remaining > 0:
                still_active.append(remaining)
            elif queue:
                still_active.append(queue.pop(0))
        active = still_active
    return wall_steps, slot_steps


def main() -> None:
    rng = random.Random(SEED)
    lengths = [sample_length(rng) for _ in range(N_REQUESTS)]
    total_useful_steps = sum(lengths)

    static_wall, static_slots = static_batching(lengths, BATCH_SIZE)
    cont_wall, cont_slots = continuous_batching(lengths, BATCH_SIZE)

    static_util = total_useful_steps / static_slots
    cont_util = total_useful_steps / cont_slots

    print(f"requests: {N_REQUESTS}, batch_size: {BATCH_SIZE}, mean length: "
          f"{total_useful_steps / N_REQUESTS:.1f} tokens")
    print(f"total useful decode steps (identical either way): {total_useful_steps:,}\n")

    print(f"{'':20s} {'wall steps':>12s} {'slot-steps':>14s} {'utilization':>12s}")
    print(f"{'static batching':20s} {static_wall:>12,} {static_slots:>14,} {static_util * 100:>11.1f}%")
    print(f"{'continuous batching':20s} {cont_wall:>12,} {cont_slots:>14,} {cont_util * 100:>11.1f}%")

    print(f"\ncontinuous batching finishes {static_wall / cont_wall:.2f}x faster (wall time)")
    print(f"continuous batching wastes {(1 - cont_util) * 100:.1f}% of slot-steps, "
          f"vs static's {(1 - static_util) * 100:.1f}%")


if __name__ == "__main__":
    main()
