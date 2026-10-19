"""Small timing helper shared by the benchmark scripts."""

from __future__ import annotations

import statistics
import time


def time_call(fn, sync=None, target_s=0.3, max_reps=100, warmup=2, slow_s=1.0):
    """Median wall time of fn() in milliseconds.

    Runs `warmup` calls first (just one if a call takes longer than `slow_s`),
    then repeats until `target_s` seconds have passed or `max_reps` calls ran.
    """
    def once():
        t0 = time.perf_counter()
        out = fn()
        if sync is not None:
            sync()
        return time.perf_counter() - t0, out

    first, _ = once()
    if first < slow_s:
        for _ in range(warmup - 1):
            once()
    times = []
    total = 0.0
    while len(times) < max_reps and (total < target_s or len(times) < 3):
        dt, _ = once()
        times.append(dt)
        total += dt
        if dt > slow_s and len(times) >= 1:
            break
    return 1000.0 * statistics.median(times), len(times)


def recall(got_ids, ref_ids):
    """Fraction of reference ids found, per query averaged. Inputs: lists of lists."""
    hit = 0
    tot = 0
    for a, b in zip(got_ids, ref_ids):
        hit += len(set(a) & set(b))
        tot += len(b)
    return hit / max(tot, 1)
