"""
Task Scheduler

Responsible for scheduling work across CPU workers.

Responsibilities
----------------
- Split datasets into chunks
- Balance workload
- Generate worker assignments
- Support future distributed execution

This module does not execute work; it only schedules it.
"""


class TaskScheduler:
    """
    Creates balanced workloads for multiprocessing.

    Parameters
    ----------
    num_workers : int
        Number of worker processes.

    chunk_size : int
        Number of samples assigned per task.
    """

    def __init__(
        self,
        num_workers=8,
        chunk_size=64,
    ):

        self.num_workers = num_workers
        self.chunk_size = chunk_size

    # -------------------------------------------------
    # Dataset chunks
    # -------------------------------------------------

    def create_chunks(
        self,
        dataset_size,
    ):
        """
        Split dataset into fixed-size chunks.

        Returns
        -------
        List[(start, end)]
        """

        chunks = []

        for start in range(
            0,
            dataset_size,
            self.chunk_size,
        ):

            end = min(
                start + self.chunk_size,
                dataset_size,
            )

            chunks.append(
                (start, end)
            )

        return chunks

    # -------------------------------------------------
    # Worker assignment
    # -------------------------------------------------

    def assign_workers(
        self,
        chunks,
    ):
        """
        Assign chunks to workers.

        Returns
        -------
        List[(worker_id, start, end)]
        """

        assignments = []

        for i, (start, end) in enumerate(chunks):

            worker = (
                i % self.num_workers
            )

            assignments.append(
                (
                    worker,
                    start,
                    end,
                )
            )

        return assignments

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    def workload_summary(
        self,
        assignments,
    ):
        """
        Return workload statistics.
        """

        summary = {
            worker: 0
            for worker in range(
                self.num_workers
            )
        }

        for worker, start, end in assignments:

            summary[worker] += (
                end - start
            )

        return summary

    # -------------------------------------------------
    # Update configuration
    # -------------------------------------------------

    def set_chunk_size(
        self,
        chunk_size,
    ):

        self.chunk_size = int(
            chunk_size
        )

    def set_workers(
        self,
        workers,
    ):

        self.num_workers = int(
            workers
        )

    # -------------------------------------------------
    # Information
    # -------------------------------------------------

    def __repr__(self):

        return (
            f"TaskScheduler("
            f"workers={self.num_workers}, "
            f"chunk_size={self.chunk_size})"
        )