class RelationRegistry:

    def __init__(self):
        self.relations = {}

    def register(self, name, relation):
        self.relations[name] = relation

    def get(self, name):
        return self.relations[name]

    def list(self):
        return list(self.relations.keys())