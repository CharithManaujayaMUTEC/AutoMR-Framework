"""
Prediction Cache

Provides a thread-safe prediction cache for
HighPerformanceAutoMR.

Purpose
-------
- Avoid repeated model inference.
- Reuse cached predictions.
- Support performance measurement.
- Provide cache hit/miss statistics.
"""

from threading import Lock


class PredictionCache:
    """
    Thread-safe prediction cache.

    Cache statistics
    ----------------
    hits:
        Number of successful cache retrievals.

    misses:
        Number of attempted retrievals where the
        requested key was not present.

    hit_ratio:
        hits / (hits + misses)

    Notes
    -----
    Statistics are updated only by ``get()``.
    Methods such as ``exists()`` do not affect the
    counters because they are metadata checks rather
    than prediction retrievals.
    """

    def __init__(self):

        self._cache = {}

        self._lock = Lock()

        # ---------------------------------------------
        # Cache instrumentation
        # ---------------------------------------------

        self._hits = 0

        self._misses = 0

    # -------------------------------------------------
    # Basic operations
    # -------------------------------------------------

    def get(self, key, default=None):
        """
        Retrieve a cached prediction.

        A successful lookup increments ``hits``.

        A missing key increments ``misses``.
        """

        with self._lock:

            if key in self._cache:

                self._hits += 1

                return self._cache[key]

            self._misses += 1

            return default

    def put(self, key, value):
        """
        Store a prediction.
        """

        with self._lock:

            self._cache[key] = value

    def exists(self, key):
        """
        Check whether a key exists.

        This does not modify hit/miss statistics.
        """

        with self._lock:

            return key in self._cache

    def remove(self, key):
        """
        Remove one cached prediction.
        """

        with self._lock:

            if key in self._cache:

                del self._cache[key]

    def clear(self):
        """
        Clear the cache and reset statistics.
        """

        with self._lock:

            self._cache.clear()

            self._hits = 0

            self._misses = 0

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    @property
    def hits(self):
        """
        Number of successful cache retrievals.
        """

        with self._lock:

            return self._hits

    @property
    def misses(self):
        """
        Number of unsuccessful cache retrievals.
        """

        with self._lock:

            return self._misses

    @property
    def requests(self):
        """
        Total cache retrieval attempts.
        """

        with self._lock:

            return (
                self._hits
                + self._misses
            )

    @property
    def hit_ratio(self):
        """
        Cache hit ratio.

        Returns
        -------
        float
            hits / (hits + misses)

        Returns 0.0 when no cache lookups have
        occurred.
        """

        with self._lock:

            total = (
                self._hits
                + self._misses
            )

            if total == 0:

                return 0.0

            return (
                self._hits
                / total
            )

    def get_stats(self):
        """
        Return all cache instrumentation statistics.

        Returns
        -------
        dict
        """

        with self._lock:

            total = (
                self._hits
                + self._misses
            )

            hit_ratio = (
                self._hits / total
                if total > 0
                else 0.0
            )

            return {
                "hits": self._hits,
                "misses": self._misses,
                "requests": total,
                "hit_ratio": hit_ratio,
                "cache_size": len(
                    self._cache
                ),
            }

    def reset_stats(self):
        """
        Reset hit/miss instrumentation without
        clearing cached predictions.
        """

        with self._lock:

            self._hits = 0

            self._misses = 0

    # -------------------------------------------------
    # Existing cache information
    # -------------------------------------------------

    def __len__(self):
        """
        Number of cached predictions.
        """

        with self._lock:

            return len(
                self._cache
            )

    def size(self):
        """
        Alias for len(cache).
        """

        return len(self)

    def keys(self):

        with self._lock:

            return list(
                self._cache.keys()
            )

    def values(self):

        with self._lock:

            return list(
                self._cache.values()
            )

    def items(self):

        with self._lock:

            return list(
                self._cache.items()
            )

    # -------------------------------------------------
    # Dictionary compatibility
    # -------------------------------------------------

    def __contains__(self, key):

        return self.exists(key)

    def __getitem__(self, key):

        return self.get(key)

    def __setitem__(self, key, value):

        self.put(
            key,
            value,
        )

    def __repr__(self):

        return (
            f"PredictionCache("
            f"size={len(self._cache)}, "
            f"hits={self._hits}, "
            f"misses={self._misses}, "
            f"hit_ratio={self.hit_ratio:.4f}"
            f")"
        )