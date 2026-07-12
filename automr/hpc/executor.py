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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

import pandas as pd
from tqdm import tqdm


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
        num_workers=32,
        batch_size=512,
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

    def _run_chunk(
        self,
        dataset,
        start,
        end,
        worker_args,
    ):
        """
        Execute one chunk using multiple threads.
        """

        dfs = []

        with ThreadPoolExecutor(
            max_workers=self.num_workers,
        ) as executor:

            futures = []

            for idx in range(start, end):

                sample = dataset[idx]

                futures.append(
                    executor.submit(
                        self.automr._process_single_sample,
                        (
                            idx,
                            sample,
                            worker_args["samples_per_mr"],
                            worker_args["df_temp"],
                            worker_args["prediction_cache"],
                        ),
                    )
                )

            for future in as_completed(futures):

                df = future.result()

                if df is not None and not df.empty:
                    dfs.append(df)

        if not dfs:
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

            class DatasetView:
                def __init__(self, dataset, limit):
                    self.dataset = dataset
                    self.limit = limit

                def __len__(self):
                    return self.limit

                def __getitem__(self, idx):
                    return self.dataset[idx]

            dataset = DatasetView(
                dataset,
                min(max_samples, len(dataset)),
            )

        # -----------------------------
        # Temporal MR
        # -----------------------------

        df_temp = None

        if include_temporal:

            try:

                temporal_limit = min(300, len(dataset))

                temporal_data = [
                    dataset[i]
                    for i in range(temporal_limit)
                ]

                df_temp, _ = self.automr.run_mr(
                    input_data=temporal_data,
                    mr_name="temporal",
                    samples=samples_per_mr,
                )

            except Exception as e:

                print("Temporal MR skipped:", e)

                df_temp = None

        worker_args = {
            "samples_per_mr": samples_per_mr,
            "df_temp": df_temp,
            "prediction_cache": prediction_cache,
        }

        chunks = self.split_dataset(dataset)

        results = []

        with ThreadPoolExecutor(
            max_workers=min(
                len(chunks),
                self.num_workers,
            )
        ) as executor:

            futures = {

                executor.submit(
                    self._run_chunk,
                    dataset,
                    start,
                    end,
                    worker_args,
                ): (start, end)

                for start, end in chunks
            }

            iterator = as_completed(futures)

            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc="Processing dataset",
                )

            for future in iterator:

                df = future.result()

                if df is not None and not df.empty:
                    results.append(df)

        if len(results) == 0:
            return pd.DataFrame()

        return pd.concat(
            results,
            ignore_index=True,
        )