from .Branch import Branch
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

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
        #treeMap = np.zeros((300, 300, 4))
        # draw all branches onto treeMap
        x,y = np.array(zip(*self.branches))   # seperate x and y coordinates from Branches
        # interpolate 
        s = 0   # smoothing condition (0 means passing all points)
        tck, t = interpolate.splprep([x, y], s=s) 
        xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 200), tck) 
        treeMap = np.array(zip(np.append(xi,x),np.append(yi,y)))
        return treeMap

    def coverage(self):
        treeMap = self.createTreeMap()
        #TODO erode treeMap
        return treeMap
