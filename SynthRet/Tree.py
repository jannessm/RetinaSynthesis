from .Branch import Branch
import numpy as np

class Tree:
    def __init__(self, startingPoint):
        self.branches = []

        for i in range(4):
            g = self.getRandomGoal()
            b = Branch(self, startingPoint, g)
            self.branches.append(b)

        # constants
        self.covThreshold = 0.9      # coverage threshold

    def getRandomGoal(self):
        return np.array((1,1))

    def growTree(self):
        cov = self.coverage()
        while np.mean(cov) < self.covThreshold:
            for b in self.branches:
                b.addSegment()
            cov = self.coverage()

    def createTreeMap(self):
        treeMap = np.zeros((300, 300, 4))
        #TODO draw all branches onto treeMap
        return treeMap

    def coverage(self):
        treeMap = self.createTreeMap()
        #TODO erode treeMap
        return treeMap
