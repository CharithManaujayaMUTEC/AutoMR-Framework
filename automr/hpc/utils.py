"""
HPC Utilities

Shared utility functions for HighPerformanceAutoMR.

Responsibilities
----------------
- Timing
- CPU information
- Batch sizing
- Memory estimation
- Progress helpers
"""

import os
import time
import multiprocessing


# ---------------------------------------------------------
# CPU Information
# ---------------------------------------------------------

def available_cpus():
    """
    Return available logical CPU cores.
    """

    return multiprocessing.cpu_count()


# ---------------------------------------------------------
# Batch Size Recommendation
# ---------------------------------------------------------

def recommend_batch_size(
    image_shape=None,
    device="cpu",
):
    """
    Recommend a reasonable batch size.

    This is intentionally conservative.
    """

    if device == "cuda":
        return 128

    return max(
        16,
        min(
            available_cpus() * 8,
            128,
        ),
    )


# ---------------------------------------------------------
# Chunk Size Recommendation
# ---------------------------------------------------------

def recommend_chunk_size(
    dataset_size,
    workers,
):
    """
    Compute a balanced chunk size.
    """

    if workers <= 0:
        workers = 1

    return max(
        32,
        dataset_size // (workers * 4),
    )


# ---------------------------------------------------------
# Timer
# ---------------------------------------------------------

class Timer:
    """
    Lightweight performance timer.

    Example
    -------
    timer = Timer()

    ...

    print(timer.elapsed())
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.start = time.perf_counter()

    def elapsed(self):

        return time.perf_counter() - self.start


# ---------------------------------------------------------
# Progress formatting
# ---------------------------------------------------------

def format_seconds(seconds):
    """
    Convert seconds into HH:MM:SS.
    """

    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02}:{m:02}:{s:02}"


# ---------------------------------------------------------
# Throughput
# ---------------------------------------------------------

def throughput(
    processed,
    elapsed,
):
    """
    Samples per second.
    """

    if elapsed <= 0:
        return 0.0

    return processed / elapsed


# ---------------------------------------------------------
# ETA
# ---------------------------------------------------------

def eta(
    processed,
    total,
    elapsed,
):
    """
    Estimated remaining time.
    """

    if processed == 0:
        return float("inf")

    rate = processed / elapsed

    remaining = total - processed

    return remaining / rate


# ---------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------

def estimate_batch_memory(
    image_shape,
    batch_size,
    dtype_bytes=4,
):
    """
    Rough memory estimate in MB.

    image_shape = (H, W, C)
    """

    h, w, c = image_shape

    bytes_required = (
        h * w * c *
        batch_size *
        dtype_bytes
    )

    return bytes_required / (1024 ** 2)


# ---------------------------------------------------------
# System summary
# ---------------------------------------------------------

def system_summary():
    """
    Return useful HPC information.
    """

    return {
        "cpu_count": available_cpus(),
        "pid": os.getpid(),
        "recommended_batch": recommend_batch_size(),
    }