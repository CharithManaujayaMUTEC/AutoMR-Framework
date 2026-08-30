"""
High Performance AutoMR API

This class extends the standard AutoMR API with
high-performance execution features.

Features
--------
- Multi-process dataset execution
- Batched inference
- Shared prediction cache
- Checkpoint support (future)
- Multi-GPU support (future)

The goal is to keep the existing AutoMR API untouched.
"""

from automr.api import AutoMR
from automr.hpc.executor import HPCExecutor
from automr.hpc.cache import PredictionCache


class HighPerformanceAutoMR(AutoMR):
    """
    High-performance implementation of AutoMR.

    Example
    -------
    automr = HighPerformanceAutoMR(
        model=model,
        task="regression",
        input_type="image",
        num_workers=8,
        batch_size=64,
    )
    """

    def __init__(
        self,
        model,
        task="regression",
        input_type="image",
        epsilon=0.05,
        range_threshold=5.0,
        # Optional user-defined transformation ranges.
        # If None, AutoMR's built-in defaults are used.
        transform_ranges=None,
        num_workers=32,
        batch_size=512,
    ):
        super().__init__(
            model=model,
            task=task,
            input_type=input_type,
            epsilon=epsilon,
            range_threshold=range_threshold,

            # Pass optional custom transformation ranges
            # to the parent AutoMR API.
            transform_ranges=transform_ranges,
        )

        # HPC configuration
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Shared prediction cache
        self.cache = PredictionCache()

        # HPC execution engine
        self.executor = HPCExecutor(
            automr=self,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            cache=self.cache,
        )

    def run_dataset(
        self,
        dataset,
        **kwargs,
    ):
        """
        High-performance dataset execution.
        """

        kwargs.setdefault(
            "cache",
            self.cache,
        )

        kwargs.setdefault(
            "show_progress",
            True,
        )

        return self.executor.run_dataset(
            dataset=dataset,
            **kwargs,
        )

    def clear_cache(self):
        """
        Clear cached predictions.
        """

        self.cache.clear()

    def cache_size(self):
        """
        Return number of cached predictions.
        """

        return len(self.cache)

    def cache_stats(self):
        """
        Return prediction cache statistics.

        Returns
        -------
        dict
            Cache hits, misses, requests, hit ratio,
            and current cache size.
        """

        return self.cache.get_stats()

    def reset_cache_stats(self):
        """
        Reset cache instrumentation statistics
        without clearing cached predictions.
        """

        self.cache.reset_stats()
