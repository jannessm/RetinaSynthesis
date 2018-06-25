import numpy as np
from Branch import Branch
from scipy import interpolate
from skimage import draw
from TreeMap import TreeMap
from utils import showImage, addMask, makeBinary, coverage, meanCoverage

class Tree:
    def __init__(self, startingPoint, fovea):
        self.branches = []
        self.growingBranches = []
        self.fovea = fovea          # fovea location [x, y]
        self.opticaldisc = startingPoint
        self.nbranches = 0
        self.treeMap = TreeMap()

        for i in range(4):          # number of arteries
            g = self.getRandomGoal(i)
            b = Branch(self, startingPoint, g, artery=True)
            self.branches.append(b)
            self.growingBranches.append(b)
        for i in range(4):          # number of veins
            g = self.getRandomGoal(i)
            b = Branch(self, startingPoint, g, artery=False)
            self.branches.append(b)
            self.growingBranches.append(b)

        # constants
        self.covThreshold = 0.53           # coverage threshold
        # self.covThreshold = 0.93107      # coverage threshold

    def getRandomGoal(self, i):
        switch = {
            0: np.array([[-300, -150], [50, 200]]) + self.fovea,
            1: np.array([[-300, -150], [-200, -50]]) + self.fovea,
            2: np.array([[50, 200], [50, 200]]) + self.opticaldisc,
            3: np.array([[50, 200], [-200, -50]]) + self.opticaldisc
        }
        boundaries = switch.get(i%4)
        if boundaries is not None:
            goal_x = np.random.randint(boundaries[0][0], boundaries[0][1])  # maybe random.choice?
            goal_y = np.random.randint(boundaries[1][0], boundaries[1][1])
        return np.array((goal_x, goal_y))

    def growTree(self):
        tMap = self.createTreeMap()
        while (meanCoverage(tMap, self.fovea) < self.covThreshold and 
            len(self.growingBranches) > 0):    # while coverage is not reached
            
            branches = self.growingBranches[:]
            for b in branches:          # grow all branches in list until they have reached goal point
                while not b.finished:
                    b.addSegment()
                self.growingBranches.remove(b)
            for b in branches:
                for p in b.points[::-1]:
                    if np.array_equal(p, b.start):
                        continue
                    tMap = self.createTreeMap()
                    if meanCoverage(tMap, self.fovea) > self.covThreshold:
                        break
                    b.addBranch(p)

    def createTreeImage(self):
        return self.treeMap.getImg()

    def createTreeMap(self):
        treeMap = self.createTreeImage()
        treeMap = makeBinary(treeMap, 10)
        notransp = np.ones(treeMap.shape) * 255
        treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))
        return treeMap.astype(int)

    def coverage(self):
        treeMap = self.createTreeMap()
        return coverage(treeMap, self.fovea)

    def b2arr(self):
        arr = np.array([[0,0]])
        for b in self.branches:
            arr = np.vstack((arr, np.asarray(b.points)))
        return arr[1:]

if __name__ == '__main__':
    import faulthandler
    faulthandler.enable()
    for i in range(10):
        print i
        t = Tree([250,150], [150, 150])
        t.growTree()
        points = np.array([0,0])
        for b in t.branches:
            points = np.vstack((points, b.points))
            points = np.vstack((points, b.goal))
        points = np.vstack((points, t.fovea))
        showImage(t.createTreeMap())
        showImage(t.createTreeImage())