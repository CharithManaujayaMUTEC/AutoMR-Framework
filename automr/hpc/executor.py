"""
High Performance Executor

Responsible for executing AutoMR over large datasets
using multiprocessing and batched execution.

Responsibilities
----------------
- Split dataset into chunks
- Execute chunks in parallel
- Merge results
- Share prediction cache
"""

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


class HPCExecutor:
    """
    Executes AutoMR over large datasets using
    multiple CPU processes.

    Parameters
    ----------
    automr : HighPerformanceAutoMR
        Parent AutoMR instance.

    num_workers : int
        Number of worker processes.

    batch_size : int
        Dataset chunk size assigned to each worker.

    cache : PredictionCache
        Shared prediction cache.
    """

    def __init__(
        self,
        automr,
        num_workers=8,
        batch_size=64,
        cache=None,
    ):

        self.automr = automr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.cache = cache

    # -------------------------------------------------
    # Split dataset
    # -------------------------------------------------

    def split_dataset(self, dataset):

        chunks = []

        total = len(dataset)

        for start in range(0, total, self.batch_size):

            end = min(start + self.batch_size, total)

            chunks.append((start, end))

        return chunks

    # -------------------------------------------------
    # Worker
    # -------------------------------------------------

    @staticmethod
    def _worker(
        automr,
        dataset,
        start,
        end,
        kwargs,
    ):
        """
        Execute AutoMR on one dataset chunk.
        """

        dfs = []

        for idx in range(start, end):

            sample = dataset[idx]

            df = automr._process_single_sample(
                (
                    idx,
                    sample,
                    kwargs["samples_per_mr"],
                    kwargs["df_temp"],
                    kwargs["prediction_cache"],
                )
            )

            dfs.append(df)

        if len(dfs) == 0:
            return pd.DataFrame()

        return pd.concat(
            dfs,
            ignore_index=True,
        )

    # -------------------------------------------------
    # Run dataset
    # -------------------------------------------------

    def run_dataset(
        self,
        dataset,
        max_samples=None,
        samples_per_mr=5,
        include_temporal=True,
        show_progress=True,
        prediction_cache=None,
        **kwargs,
    ):
        """
        Parallel dataset execution.
        """

        if max_samples is not None:
            dataset = dataset[:max_samples]

        # Temporal MR executed once
        df_temp = None

        if include_temporal:

            try:

                temporal_data = [
                    dataset[i]
                    for i in range(len(dataset))
                ]

                df_temp, _ = self.automr.run_mr(
                    input_data=temporal_data,
                    mr_name="temporal",
                    samples=samples_per_mr,
                )

            except Exception:

                df_temp = None

        chunks = self.split_dataset(dataset)

        results = []

        worker_args = {
            "samples_per_mr": samples_per_mr,
            "df_temp": df_temp,
            "prediction_cache": prediction_cache,
        }

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:

            futures = []

            for start, end in chunks:

                futures.append(

                    executor.submit(
                        HPCExecutor._worker,
                        self.automr,
                        dataset,
                        start,
                        end,
                        worker_args,
                    )

                )

            if show_progress:

                from tqdm import tqdm

                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing dataset",
                )

            else:

                iterator = as_completed(futures)

            for future in iterator:

                results.append(
                    future.result()
                )

        if len(results) == 0:
            return pd.DataFrame()

        return pd.concat(
            results,
            ignore_index=True,
        )