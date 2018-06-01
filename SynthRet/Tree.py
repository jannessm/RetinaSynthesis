import numpy as np

from Branch import Branch
from scipy import interpolate
from scipy.ndimage.morphology import binary_dilation
from utils import showImage

class Tree:
    def __init__(self, startingPoint, fovea):
        self.branches = []
        self.fovea = fovea          # fovea location [x, y]

        # for i in range(4):
        #     g = self.getRandomGoal(i)
        #     b_artery = Branch(self, startingPoint, g)
        #     b_vein = Branch(self, startingPoint, g)
        #     self.branches.append(b_artery)
        #     self.branches.append(b_vein)

        for i in range(8):
            g = self.getRandomGoal(i)
            b = Branch(self, startingPoint, g)
            self.branches.append(b)

        # constants
        self.covThreshold = 0.9      # coverage threshold

    def addBranch(self, startingPoint, goalPoint):
        b = Branch(self, startingPoint, goalPoint)
        self.branches.append(b)

    def getRandomGoal(self, i):
        switch = {
            0: [[-100, 100], [0, 150]],
            1: [[-100, 100], [150, 300]],
            2: [[300, 400], [0, 100]],
            3: [[300, 400], [200, 300]]
        }
        boundaries = switch.get(i%4)
        if boundaries is not None:
            goal_x = np.random.randint(boundaries[0][0], boundaries[0][1])
            goal_y = np.random.randint(boundaries[1][0], boundaries[1][1])
        return np.array((goal_x, goal_y))

    def growTree(self):
        #cov = self.coverage()
        branches = self.branches[:]
        while len(branches) > 0:
            for b in branches:
                if b.finished:
                    branches.remove(b)
                else:
                    b.addSegment()
            cov = self.coverage()

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
        binary = self.makeBinary(treeMap, 200)
        binary = binary_dilation(binary, iterations=k)
        return binary

    def makeBinary(self, img, threshold):
        r = np.multiply(img[:, :, 0], img[:, :, 3])
        g = np.multiply(img[:, :, 1], img[:, :, 3])
        b = np.multiply(img[:, :, 2], img[:, :, 3])
        rgb = np.dstack((r, g, b))
        grey = np.multiply(rgb, [0.21, 0.72, 0.07])
        grey = np.sum(grey, axis=2)
        grey[np.where(grey < threshold)] = 0
        grey[np.where(grey > threshold)] = 255
        return grey

if __name__ == '__main__':
    for i in range(10):
        t = Tree([250,150], [150, 150])
        t.growTree()
        print t.iteration
        #points = np.array([0,0])
        #for b in t.branches:
        #    points = np.vstack((points, b.points))
        #    points = np.vstack((points, b.goal))
        #points = np.vstack((points, t.fovea))
        #showImage(t.createTreeMap(), points[1:])
        #showImage(t.createTreeMap())
        showImage(t.coverage())