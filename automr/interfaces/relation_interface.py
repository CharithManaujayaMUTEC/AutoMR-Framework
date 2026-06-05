from abc import ABC, abstractmethod


class BaseRelation(ABC):
    """
    Base interface for metamorphic relations.
    """

    @abstractmethod
    def check(self, original_output, transformed_output):
        """
        Validate MR condition.

        Returns
        -------
        tuple
            (difference, passed)
        """
        pass

    def expected(self):
        """
        Human-readable MR description.
        """
        return "Expected behavior not specified."