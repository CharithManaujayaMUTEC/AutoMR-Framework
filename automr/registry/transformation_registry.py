class TransformationRegistry:

    def __init__(self):
        self.transforms = {}

    def register(self, name, transform):
        self.transforms[name] = transform

    def get(self, name):
        return self.transforms[name]

    def list(self):
        return list(self.transforms.keys())