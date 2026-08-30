"""
Relation registry.

This module implements a lightweight registry used to store and
retrieve metamorphic relation instances by name.
"""


class RelationRegistry:
    """
    Registry for metamorphic relations.
    """

    def __init__(self):
        """Initialize an empty relation registry."""
        self.relations = {}

    def register(self, name, relation):
        """
        Register a metamorphic relation.

        Parameters
        ----------
        name : str
            Relation identifier.
        relation : BaseRelation
            Relation instance.
        """
        self.relations[name] = relation

    def get(self, name):
        """
        Retrieve a registered relation.

        Parameters
        ----------
        name : str
            Relation identifier.

        Returns
        -------
        BaseRelation
            Registered relation instance.
        """
        return self.relations[name]

    def list(self):
        """
        Return all registered relation names.

        Returns
        -------
        list[str]
            Registered relation identifiers.
        """
        return list(self.relations.keys())