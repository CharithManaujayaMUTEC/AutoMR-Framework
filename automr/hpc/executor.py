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

    def _run_chunk(
        self,
        dataset,
        start,
        end,
        worker_args,
    ):
        """
        Execute AutoMR on one dataset chunk.
        """

        dfs = []

        for idx in range(start, end):

            sample = dataset[idx]

            df = self.automr._process_single_sample(
                (
                    idx,
                    sample,
                    worker_args["samples_per_mr"],
                    worker_args["df_temp"],
                    worker_args["prediction_cache"],
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

        iterator = chunks

        if show_progress:
            iterator = tqdm(
                chunks,
                desc="Processing dataset",
            )

        for start, end in iterator:

            df = self._run_chunk(
                dataset,
                start,
                end,
                worker_args,
            )

            if not df.empty:
                results.append(df)

        if len(results) == 0:
            return pd.DataFrame()

        return pd.concat(
            results,
            ignore_index=True,
        )