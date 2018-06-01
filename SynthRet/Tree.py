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

        for i in range(1):          # number of branches
            g = self.getRandomGoal(i)
            b = Branch(self, startingPoint, g)
            self.branches.append(b)
            self.growingBranches.append(b)

        # constants
        self.covThreshold = 0.9      # coverage threshold

    def addBranch(self, startingPoint, goalPoint):
        b = Branch(self, startingPoint, goalPoint)
        self.branches.append(b)
        self.growingBranches.append(b)

    def getRandomGoal(self, i):
        switch = {
            0: np.array([[-200, -50], [0, 100]]) + self.fovea,
            1: np.array([[-200, -50], [-100, 0]]) + self.fovea,
            2: np.array([[50, 200], [50, 200]]) + self.opticaldisc,
            3: np.array([[50, 200], [-200, -50]]) + self.opticaldisc
        }
        boundaries = switch.get(i%4)
        if boundaries is not None:
            goal_x = np.random.randint(boundaries[0][0], boundaries[0][1])
            goal_y = np.random.randint(boundaries[1][0], boundaries[1][1])
        return np.array((goal_x, goal_y))

    def growTree(self):
        while len(self.growingBranches) > 0:
            for b in self.growingBranches:
                if b.finished:
                    self.growingBranches.remove(b)
                else:
                    b.addSegment()

    # TODO add different diameters
    def createTreeMap(self):
        treeMap = np.zeros((300, 300, 4))
        treeMap[:,:,3] = 255

        # draw all branches onto treeMap
        branches = [b.points for b in self.branches]
        for branch in branches:
            x,y = np.array(zip(*branch))     # seperate x and y coordinates from Branches
            
            # interpolate 
            s = 0   # smoothing condition (0 means passing all points)
            k = 3 if x.shape[0] > 3 else x.shape[0]-1
            if k == 0:
                continue
            tck, t = interpolate.splprep([x, y], s=s, k=k) 
            xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 400), tck)
            xi, yi = xi.astype(int), yi.astype(int)                         # convert to int
            xout = np.where( xi >= treeMap.shape[0])
            yout = np.where( yi >= treeMap.shape[1])
            xi[xout] = 299                                                  # remove out of bounds indexes
            yi[yout] = 299                                                  # remove out of bounds indexes
            treeMap[xi, yi] = [255, 255, 255, 255]                          # make points of branches white

        return treeMap

    def createTreeImage(self):
        treeMap = self.createTreeMap()
        #iterate over treemap and set alpha to 0 for black points
        eq = np.where(np.sum(treeMap, axis=2) == 255)
        neq = np.where(np.sum(treeMap, axis=2) > 255)
        treeMap[eq] = [0,0,0,0]
        treeMap[neq] = [200,0,0,255]
        return treeMap

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
    for i in range(1):
        t = Tree([250,150], [150, 150])
        t.growTree()
        #points = np.array([0,0])
        #for b in t.branches:
        #    points = np.vstack((points, b.points))
        #    points = np.vstack((points, b.goal))
        #points = np.vstack((points, t.fovea))
        showImage(t.createTreeMap())
        showImage(t.coverage())
        #showImage(t.coverage())