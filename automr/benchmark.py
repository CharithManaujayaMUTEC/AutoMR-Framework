"""
AutoMR Performance Benchmark Runner.

Provides controlled benchmarking between the standard
AutoMR implementation and HighPerformanceAutoMR.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import psutil


class AutoMRBenchmark:
    """
    Controlled benchmark runner for comparing standard
    AutoMR and HighPerformanceAutoMR execution.

    Measures:
    - Runtime
    - Throughput
    - CPU utilization
    - RAM usage
    - GPU utilization when available
    - GPU memory when available
    - Cache statistics
    - Speedup
    """

    def __init__(
        self,
        standard_automr,
        hpc_automr,
    ):
        self.standard_automr = standard_automr
        self.hpc_automr = hpc_automr

    # ==================================================
    # GPU metrics
    # ==================================================

    @staticmethod
    def _get_gpu_metrics() -> Dict[str, Optional[float]]:
        """
        Retrieve GPU memory information when PyTorch/CUDA
        is available.

        GPU utilization is not universally available through
        PyTorch, so it is reported as None unless another
        monitoring backend is added.
        """

        metrics = {
            "gpu_available": False,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }

        try:
            import torch

            if torch.cuda.is_available():

                metrics["gpu_available"] = True

                metrics[
                    "gpu_memory_allocated_mb"
                ] = (
                    torch.cuda.memory_allocated()
                    / (1024 ** 2)
                )

                metrics[
                    "gpu_memory_reserved_mb"
                ] = (
                    torch.cuda.memory_reserved()
                    / (1024 ** 2)
                )

        except Exception:
            pass

        return metrics

    # ==================================================
    # Single benchmark execution
    # ==================================================

    def _run_single(
        self,
        automr,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        show_progress=False,
    ) -> Dict[str, Any]:
        """
        Execute one benchmark run.
        """

        process = psutil.Process()

        cpu_before = process.cpu_times()

        memory_before = process.memory_info().rss

        gpu_before = self._get_gpu_metrics()

        start_time = time.perf_counter()

        results = automr.run_dataset(
            dataset=dataset,
            max_samples=max_samples,
            samples_per_mr=samples_per_mr,
            show_progress=show_progress,
        )

        end_time = time.perf_counter()

        cpu_after = process.cpu_times()

        memory_after = process.memory_info().rss

        gpu_after = self._get_gpu_metrics()

        runtime_seconds = (
            end_time - start_time
        )

        if max_samples is not None:

            sample_count = min(
                max_samples,
                len(dataset),
            )

        else:

            sample_count = len(dataset)

        throughput = (
            sample_count / runtime_seconds
            if runtime_seconds > 0
            else 0.0
        )

        cpu_time_seconds = (
            (
                cpu_after.user
                + cpu_after.system
            )
            -
            (
                cpu_before.user
                + cpu_before.system
            )
        )

        memory_delta_mb = (
            memory_after
            - memory_before
        ) / (1024 ** 2)

        result = {
            "runtime_seconds":
                runtime_seconds,

            "sample_count":
                sample_count,

            "throughput_samples_per_second":
                throughput,

            "cpu_time_seconds":
                cpu_time_seconds,

            "memory_before_mb":
                memory_before
                / (1024 ** 2),

            "memory_after_mb":
                memory_after
                / (1024 ** 2),

            "memory_delta_mb":
                memory_delta_mb,

            "gpu":
                gpu_after,

            "results":
                results,
        }

        return result

    # ==================================================
    # Standard vs HPC
    # ==================================================

    def run(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        show_progress=False,
    ) -> Dict[str, Any]:
        """
        Run a controlled comparison between standard
        AutoMR and HPC AutoMR.

        Both implementations receive identical benchmark
        parameters.
        """

        # ----------------------------------------------
        # Standard execution
        # ----------------------------------------------

        standard_result = self._run_single(
            automr=self.standard_automr,
            dataset=dataset,
            max_samples=max_samples,
            samples_per_mr=samples_per_mr,
            show_progress=show_progress,
        )

        # ----------------------------------------------
        # Reset HPC cache statistics
        #
        # We do not clear the cache itself because cache
        # behavior is part of the HPC measurement.
        # ----------------------------------------------

        if hasattr(
            self.hpc_automr,
            "reset_cache_stats",
        ):

            self.hpc_automr.reset_cache_stats()

        # ----------------------------------------------
        # HPC execution
        # ----------------------------------------------

        hpc_result = self._run_single(
            automr=self.hpc_automr,
            dataset=dataset,
            max_samples=max_samples,
            samples_per_mr=samples_per_mr,
            show_progress=show_progress,
        )

        # ----------------------------------------------
        # Cache statistics
        # ----------------------------------------------

        cache_stats = None

        if hasattr(
            self.hpc_automr,
            "cache_stats",
        ):

            cache_stats = (
                self.hpc_automr.cache_stats()
            )

        # ----------------------------------------------
        # Speedup
        # ----------------------------------------------

        standard_runtime = (
            standard_result[
                "runtime_seconds"
            ]
        )

        hpc_runtime = (
            hpc_result[
                "runtime_seconds"
            ]
        )

        speedup = (
            standard_runtime
            / hpc_runtime
            if hpc_runtime > 0
            else None
        )

        return {
            "standard": standard_result,
            "hpc": hpc_result,
            "cache": cache_stats,
            "speedup": speedup,
        }