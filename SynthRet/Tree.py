import numpy as np

from Branch import Branch
from scipy import interpolate
from scipy.ndimage.morphology import binary_dilation
from skimage import draw
from utils import showImage, addMask, makeBinary

class Tree:
    def __init__(self, startingPoint, fovea):
        self.branches = []
        self.growingBranches = []
        self.fovea = fovea          # fovea location [x, y]
        self.opticaldisc = startingPoint
        self.nbranches = 0

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
        self.covThreshold = 0.9      # coverage threshold

    def addBranch(self, startingPoint, goalPoint, level, artery):
        b = Branch(self, startingPoint, goalPoint, level, artery=artery)
        self.branches.append(b)
        self.growingBranches.append(b)

    def getRandomGoal(self, i):
        switch = {
            0: np.array([[-300, -100], [0, 100]]) + self.fovea,
            1: np.array([[-300, -100], [-100, 0]]) + self.fovea,
            2: np.array([[50, 200], [50, 200]]) + self.opticaldisc,
            3: np.array([[50, 200], [-200, -50]]) + self.opticaldisc
        }
        boundaries = switch.get(i%4)
        if boundaries is not None:
            goal_x = np.random.randint(boundaries[0][0], boundaries[0][1])  # maybe random.choice?
            goal_y = np.random.randint(boundaries[1][0], boundaries[1][1])
        return np.array((goal_x, goal_y))

    def growTree(self):
        while np.mean(self.coverage()) / 255 < .9 and len(self.growingBranches) > 0:    # while coverage is not reached
            branches = self.growingBranches[:]
            for b in branches:          # grow all branches in list until they have reached goal point
                while not b.finished:
                    b.addSegment()
                    #showImage(self.createTreeImage(), [self.fovea], sec=.01)
                self.growingBranches.remove(b)
            for b in branches:
                for p in b.points:
                    if np.array_equal(p, b.start):
                        continue
                    b.addBranch(p)

    # TODO add different diameters
    def createTreeMap(self):
        treeMap = np.zeros((300, 300, 4))
        treeMap[:,:,3] = 255

        # draw all branches onto treeMap
        for branch in self.branches:
            diameter = 3 - (branch.level) if branch.level < 3 else 1
            color = 255 if branch.artery else 150
            x,y = np.array(zip(*branch.points))     # seperate x and y coordinates from Branches
            
            # interpolate 
            s = 0   # smoothing condition (0 means passing all points)
            k = 3 if x.shape[0] > 3 else x.shape[0]-1
            if k == 0:
                continue
            tck, t = interpolate.splprep([x, y], s=s, k=k) 
            xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 400), tck)
            xi, yi = xi.astype(int), yi.astype(int)                         # convert to int

            p = np.vstack((xi, yi)).T                                       # remove all points > 300 and < 0
            toHigh = np.unique(np.where(p > 299)[0])
            p = np.delete(p, toHigh, axis=0)
            toLow = np.unique(np.where(p < 0)[0])
            p = np.delete(p, toLow, axis=0)

            xi, yi = np.array(zip(*p))

            branchImage = np.zeros((300,300,4))
            branchImage[:,:,3] = 255
            branchImage[xi, yi] = [255, 255, 255, 255]                      # make points of branches white
            bina = makeBinary(branchImage, 200)
            bina = binary_dilation(bina, iterations=diameter)
            treeMap[bina] = color

        return treeMap

    def createTreeImage(self):
        treeMap = self.createTreeMap()
        #iterate over treemap and set alpha to 0 for black points
        eq = np.where(np.sum(treeMap, axis=2) == 255)
        arteries = np.where(np.sum(treeMap, axis=2) == 1020)
        veins = np.where(np.sum(treeMap, axis=2) == 600)
        treeMap[eq] = [0,0,0,0]
        treeMap[arteries] = [242, 12, 0, 255]
        treeMap[veins] = [242, 12, 0, 220] 
        return treeMap#.astype(int)

    def coverage(self, k=10):
        treeMap = self.createTreeMap()

        # dilation
        binary = makeBinary(treeMap, 200)
        binary = binary_dilation(binary, iterations=k)

        rgba = np.zeros(treeMap.shape) # make rgba image from binary
        rgba[np.where(binary)] = [0,0,0, 255] # draw binary on it

        # add fovea
        rr, cc = draw.circle(self.fovea[0], self.fovea[1],15)
        draw.set_color(rgba, [rr,cc], [0,0,0,255])

        # add mask
        binary = addMask(rgba)    # add Mask
        binary = np.abs(binary - [255, 255, 255, 0])
        return makeBinary(binary, 200)

if __name__ == '__main__':
    for i in range(10):
        t = Tree([250,150], [150, 150])
        t.growTree()
        points = np.array([0,0])
        for b in t.branches:
            points = np.vstack((points, b.points))
            points = np.vstack((points, b.goal))
        points = np.vstack((points, t.fovea))
        #showImage(t.createTreeMap(), points=points)
        showImage(t.createTreeMap())
        showImage(t.createTreeImage())
        #showImage(t.createTreeMap())