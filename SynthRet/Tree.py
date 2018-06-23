import numpy as np
from Branch import Branch
from scipy import interpolate
from scipy.ndimage.morphology import binary_dilation,binary_erosion
from skimage import draw
from matplotlib import pyplot as plt
from PIL import Image
from utils import showImage, addMask, makeBinary, coverage, meanCoverage
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
        # print "meanCov:         ", meanCoverage(self.createTreeMap(), self.fovea, 0)
        # print "growingBranches: ", len(self.growingBranches)

    # TODO add different diameters
    def createTreeMap(self, unicolor=False):
        treeMap = np.zeros((300, 300, 4))
        treeMap[:,:,3] = 255

        # draw all branches onto treeMap
        for branch in self.branches:
            #diameter = 3 - (branch.level) if branch.level < 3 else 1
            #erosion = 1
            color = 255 if branch.artery or unicolor else 150
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
            p = np.delete(p, toLow, axis=0)         # p:[[x1,y1],[x2,y2]....]
          
            xi, yi = np.array(zip(*p)) 
            #print xi.shape,yi.shape 

            #qie pian
            r = len(xi)         # alomst 400
            fig = plt.figure(figsize=(3,3),dpi=100)
            buffer_ = StringIO()
            #for i in range(r/100):
                #if (i == (r/10-1)):
                    #plt.plot(xi[10*i:],yi[10*i:],linewidth=(r-i)/r*2)
                #else:
                #plt.plot(xi[100*i:100*i+100],yi[100*i:100*i+100],linewidth=(r-i*100)/r*3)
            plt.plot(xi,yi,linewidth=2)
            plt.savefig(buffer_,format='png')
            #plt.show()
            buffer_.seek(0)
            plt.close(fig)
            im = Image.open(buffer_)
            im_arr = np.asarray(im)     #shape is (300,300,4)
            #print im_arr[1][1]
            #xi,yi = np.array(zip(*im_arr))
            
            #xi = xi[np.where(xi <= 300)]
            #yi = yi[np.where(yi <= 300)]
            buffer_.close()
            #print im_arr[0,0,:]#.shape,xi.shape,yi.shape           #17910


            #branchImage = np.zeros((300,300,4))
            #branchImage[:,:,3] = 255
            #branchImage[xi, yi] = [255, 255, 255, 255]                      # make points of branches white
            #bina = makeBinary(branchImage, 200)
            bina = makeBinary(im_arr, 200)
            bina = bina.astype(int)
            #print bina.dtype
            #bina = binary_dilation(bina, iterations=diameter)
            #bina = binary_erosion(bina, iterations=erosion)
            treeMap[bina] = color

        return treeMap

    def createTreeImage(self):
        treeMap = self.createTreeMap()
        #iterate over treemap and set alpha to 0 for black points
        eq = np.where(np.sum(treeMap, axis=2) == 255)
        arteries = np.where(np.sum(treeMap, axis=2) == 1020)
        veins = np.where(np.sum(treeMap, axis=2) == 600)
        treeMap[eq] = [0,0,0,0]
        treeMap[arteries] = [201,31,55,255]
        treeMap[veins] = [243, 83, 54, 255] 
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
        #showImage(t.createTreeMap(), points=points)
        showImage(t.createTreeMap())
        showImage(t.createTreeImage())