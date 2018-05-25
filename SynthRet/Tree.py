from Branch import Branch
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage.morphology import binary_dilation

class Tree:
    def __init__(self, startingPoint, fovea):
        self.branches = []
        self.fovea = fovea          # fovea location [x, y]

        for i in range(1):
            g = self.getRandomGoal()
            b = Branch(self, startingPoint, g)
            self.branches.append(b)

        # constants
        self.covThreshold = 0.9      # coverage threshold

    def getRandomGoal(self):
        return np.array((100,100))

    def growTree(self):
        #cov = self.coverage()
        #while np.mean(cov) < self.covThreshold:
        for i in range(1): #debugging loop
            for b in self.branches:
                b.addSegment()
            cov = self.coverage()

    def createTreeMap(self):
        treeMap = np.zeros((300, 300, 4))
        treeMap[:,:,3] = 1

        # draw all branches onto treeMap
        branches = [b.points for b in self.branches]
        for branch in branches:
            x,y = np.array(zip(*branch))     # seperate x and y coordinates from Branches
            
            # interpolate 
            s = 0   # smoothing condition (0 means passing all points)
            k = 3 if x.shape[0] >= 3 else x.shape[0]-1
            tck, t = interpolate.splprep([x, y], s=s, k=k) 
            xi, yi = interpolate.splev(np.linspace(t[0], t[-1], 200), tck)
            xi, yi = xi.astype(int), yi.astype(int)                         # convert to int
            xout = np.where( xi >= treeMap.shape[0])
            yout = np.where( yi >= treeMap.shape[1])
            xi[xout] = 299                                                  # remove out of bounds indexes
            yi[yout] = 299                                                  # remove out of bounds indexes
            treeMap[xi, yi] = [255, 255, 255, 1]                            # make points of branches white

        return treeMap

    def coverage(self, k=10):
        for branch in self.branches:
            if len(branch.points) < 2:
                return None
        treeMap = self.createTreeMap()
        binary = self.makeBinary(treeMap, 200)
        for i in range(k):
            binary = binary_dilation(binary)
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
    t = Tree([200,200], [150, 150])
    t.growTree()
    plt.imshow(t.coverage())
    plt.show()