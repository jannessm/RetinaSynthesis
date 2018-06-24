import numpy as np
from Branch import Branch
from scipy import interpolate
from scipy.ndimage.morphology import binary_dilation,binary_erosion
from skimage import draw
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import PIL.ImageOps
from utils import showImage, addMask, makeBinary, coverage, meanCoverage, fig2ndarray
import pylab
from cStringIO import StringIO

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
        while (meanCoverage(self.createTreeMap(), self.fovea, 0) < 0.388 and 
            len(self.growingBranches) > 0):    # while coverage is not reached
            
            branches = self.growingBranches[:]
            for b in branches:          # grow all branches in list until they have reached goal point
                while not b.finished:
                    b.addSegment()
                self.growingBranches.remove(b)
            for b in branches:
                for p in b.points:
                    if np.array_equal(p, b.start):
                        continue
                    b.addBranch(p)
            print "meanCov:         ", meanCoverage(self.createTreeMap(), self.fovea, 0)
            print "growingBranches: ", len(self.growingBranches)

    def createTreeImage(self):
        fig, axes = plt.subplots(figsize=(3,3),dpi=100)
        plt.axis('off')
        plt.xlim(0,300)
        plt.ylim(0,300)

        # draw all branches onto treeMap
        for branch in self.branches:
            color = (201. / 255, 31. / 255, 55. / 255, 1) if branch.artery else (243. / 255, 83. / 255, 54. / 255, 1)
            x,y = np.array(zip(*branch.points))     # seperate x and y coordinates from Branches
            
            # interpolate 
            s = 0   # smoothing condition (0 means passing all points)
            k = 3 if x.shape[0] > 3 else x.shape[0]-1
            if k == 0:
                continue
            
            x_len = max(x) - min(x)
            y_len = max(y) - min(y)
            total_len = np.sqrt(x_len**2 + y_len**2)

            tck, t = interpolate.splprep([x, y], s=s, k=k) 
            xi, yi = interpolate.splev(np.linspace(t[0], t[-1], total_len * 2), tck)
            
            r = np.linspace(0, total_len * 2, total_len * 2)
            if branch.level == 1:
                widths = 0.003 * r + 0.4
            else:
                widths = 0.003 * r + 0.3
            points = np.array([xi, yi]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]
            lc = LineCollection(segments, linewidths=widths, color=color)
            axes.add_collection(lc)

        plt.show(block=False)
        treeImg = fig2ndarray(fig)
        plt.close()

        showImage(treeImg)
        return treeImg

    def createTreeMap(self):
        treeMap = self.createTreeImage()
        showImage(treeMap)
        treeMap = makeBinary(treeMap, 10)
        showImage(treeMap)
        notransp = np.ones(treeMap.shape) * 255
        treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))
        return treeMap.astype(int)

    def coverage(self, k=10):
        treeMap = self.createTreeMap()
        return coverage(treeMap, self.fovea, k)

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