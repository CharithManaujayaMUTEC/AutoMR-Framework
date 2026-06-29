from automr.interfaces import BaseComparator


class ClassificationComparator(BaseComparator):

    def compare(self, y1, y2):

        passed = y1 == y2
        diff = 0 if passed else 1

        return diff, passed