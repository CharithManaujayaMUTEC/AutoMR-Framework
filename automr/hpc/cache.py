"""
Prediction Cache

Provides a thread-safe prediction cache for
HighPerformanceAutoMR.

Purpose
-------
- Avoid repeated model inference.
- Reuse transformed predictions.
- Share predictions across epsilon sweeps.
- Support future disk persistence.
"""

from threading import Lock


class PredictionCache:
    """
    Simple thread-safe prediction cache.

    Cache key
    ---------
    Any hashable object.

    Examples
    --------
    ("BrightnessRelation", 0.25)

    ("NoiseRelation", 15.0)

    ("sample42", "rotation", 30)
    """

    def __init__(self):

        self._cache = {}

        self._lock = Lock()

    # -------------------------------------------------
    # Basic operations
    # -------------------------------------------------

    def get(self, key, default=None):
        """
        Retrieve a cached prediction.
        """

        with self._lock:

            return self._cache.get(
                key,
                default,
            )

    def put(self, key, value):
        """
        Store a prediction.
        """

        with self._lock:

            self._cache[key] = value

    def exists(self, key):
        """
        Check whether a key exists.
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
        Clear the cache.
        """

        with self._lock:

            self._cache.clear()

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    def __len__(self):
        """
        Number of cached predictions.
        """

        return len(self._cache)

    def size(self):
        """
        Alias for len(cache).
        """

        return len(self)

    def keys(self):
        """
        Return cache keys.
        """

        with self._lock:

            return list(self._cache.keys())

    def values(self):
        """
        Return cache values.
        """

        with self._lock:

            return list(self._cache.values())

    def items(self):
        """
        Return cache entries.
        """

        with self._lock:

            return list(self._cache.items())

    # -------------------------------------------------
    # Dictionary compatibility
    # -------------------------------------------------

    def __contains__(self, key):

        return self.exists(key)

    def __getitem__(self, key):

        return self.get(key)

    def __setitem__(self, key, value):

        self.put(key, value)

    def __repr__(self):

        return (
            f"PredictionCache("
            f"size={len(self._cache)})"
        )