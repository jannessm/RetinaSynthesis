import numpy as np
from Branch import Branch
from TreeMap import TreeMap
from utils import showImage, makeBinary, meanCoverage

'''
    Tree
    startingPoint   - point where to start the tree at
    fovea           - position of the fovea
    This class manages the whole tree.
'''
class Tree:
    def __init__(self, sizeX, sizeY, startingPoint, fovea):
        self.branches = []                  # all branches
        self.growingBranches = []           # all branches which are not finished yet
        self.fovea = fovea                  # fovea location [x, y]
        self.opticaldisc = startingPoint    # optical disc location [x, y]
        self.treeMap = TreeMap(sizeX, sizeY)            # treeMap object to handle the images
        self.centers = []                   # list of used centers for branches with a level > 1
        self.sizeX = sizeX
        self.sizeY = sizeY

        for i in range(4):                  # init 4 arteries
            g = self.getRandomGoal(i)       # get it's goalpoint
            b = Branch(self, startingPoint, g, artery=True)
            self.branches.append(b)         # add it to list of branches
            self.growingBranches.append(b)  # add it to list of growing branches

        for i in range(4):                  # init 4 veins
            g = self.getRandomGoal(i)       # get it's goalpoint
            b = Branch(self, startingPoint, g, artery=False)
            self.branches.append(b)         # add it to list of branches
            self.growingBranches.append(b)  # add it to list of growing branches

        # constants
        self.covThreshold = 0.0001         # coverage threshold of groundtruth

    '''
        getRandomGoal
        i - id of branch
        get a random goal point
    '''
    def getRandomGoal(self, i):
        switch = {
            0: np.array([[-1, -0.5], [0.167, 0.667]]) * self.sizeX + self.fovea,
            1: np.array([[-1, -0.5], [-.667, -.167]]) * self.sizeY + self.fovea,
            2: np.array([[0.167, 0.667], [0.067, 0.667]]) * self.sizeX + self.opticaldisc,
            3: np.array([[0.067, 0.667], [-.667, -.067]]) * self.sizeY + self.opticaldisc
        }
        boundaries = switch.get(i%4)    # get random boundaries
        if boundaries is not None:      # if there are boundaries run np.randint()
            goal_x = np.random.randint(boundaries[0][0], boundaries[0][1])
            goal_y = np.random.randint(boundaries[1][0], boundaries[1][1])
        return np.array((goal_x, goal_y))

    '''
        growTree
        trigger generation of the tree
    '''
    def growTree(self):
        tMap = self.createTreeMap()             # get current binary treeMap

        # while the mean coverage is below the wanted threshold and there are non finished branches
        # keep growing
        meanCoverageValue = .0
        meanCoverageStart = .0
        while (meanCoverageValue < meanCoverageStart + self.covThreshold and 
            len(self.growingBranches) > 0):
            
            branches = self.growingBranches[:]      # deepcopy of all non finished branches
            for b in branches:                      # grow all branches in list until they have reached goal point
                while not b.finished:
                    b.addSegment()
                self.growingBranches.remove(b)      # when branch is finished remove it from list


            for b in branches:                      # add branches for each point on the created branches
                for p in b.points[::-1]:
                    if np.array_equal(p, b.start):  # exclude the starting point
                        continue
                    tMap = self.createTreeMap()     # get the current binary treeMap

                    # if the mean coverage is reached quit the loop
                    meanCoverageValue = meanCoverage(tMap, self.sizeX, self.sizeY)
                    if meanCoverageValue > 0.0001 and meanCoverageStart == .0:
                        meanCoverageStart = meanCoverageValue
                    if meanCoverageValue > self.covThreshold + meanCoverageStart:
                        break
                    b.addBranch(p)
            meanCoverageValue = meanCoverage(tMap, self.sizeX, self.sizeY)
            print(meanCoverageValue, meanCoverageStart)

    '''
        createTreeImage
        get the current tree image
    '''
    def createTreeImage(self):
        return self.treeMap.getImg()

    '''
        createTreeMap
        get the current binary treemap
    '''
    def createTreeMap(self):
        return self.treeMap.getMap()

    '''
        createAliasedTreeImage
        get the current tree image
    '''
    def createSupersampledImages(self, supersampling):
        return self.treeMap.createSupersampledImages(supersampling)

    '''
        b2arr
        collect all points from all branches of this tree
    '''
    def b2arr(self):
        arr = np.array([[0,0]])                 # init array
        for b in self.branches:                 # add points
            arr = np.vstack((arr, np.asarray(b.points)))
        return arr[1:]                          # remove init point
