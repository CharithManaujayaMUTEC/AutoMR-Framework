"""
Transformation registry.

This module implements a lightweight registry used to store and
retrieve transformation functions by name.
"""


class TransformationRegistry:
    """
    Registry for input transformations.
    """

    def __init__(self):
        """Initialize an empty transformation registry."""
        self.transforms = {}

    def register(self, name, transform):
        """
        Register a transformation.

        Parameters
        ----------
        name : str
            Transformation identifier.
        transform : callable
            Transformation function.
        """
        self.transforms[name] = transform

    def get(self, name):
        """
        Retrieve a registered transformation.

        Parameters
        ----------
        name : str
            Transformation identifier.

        Returns
        -------
        callable
            Registered transformation function.
        """
        return self.transforms[name]

    def list(self):
        """
        Return all registered transformation names.

        Returns
        -------
        list[str]
            Registered transformation identifiers.
        """
        return list(self.transforms.keys())