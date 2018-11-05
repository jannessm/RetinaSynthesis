import numpy as np
from utils import showImage, meanCoverage
from Goals import nextGoalPoint

'''
    class Branch
    tree            - tree the branch will belong to
    startingPoint   - point the branch is starting at
    goalPoint       - point where the branch will grow to
    level           - level of the branch
    artery          - is the branch an artery?
    this class is used to organize each branch which is generated.
'''
class Branch:
    def __init__(self, tree, startingPoint, goalPoint, level=1, artery=True):
        self.points = [np.array(startingPoint)]
        self.start = startingPoint
        self.goal = np.array(goalPoint)                 # goal point
        self.tree = tree                                # tree object

        #attributes
        self.finished = False
        self.level = level
        self.artery = artery
        self.subBranchesFovea = 0                       # amount of subbranches that go to the fovea
        self.subBranchesNotFovea = 0                    # amount of subbranches that go into the opposite direction

        #constants
        self.goalThreshold = 0.03 * self.tree.sizeX     # pixels away from goal point are sufficent
        self.maxAngle = 60                              # upper boundary for random angle of the next segment
    
    '''
        setLevel
        l - new level
        sets the level of a branch
    '''
    def setLevel(self, l):
        self.level = l

    '''
        addSegment
        adds a segment to the current branch if goal point is not reached yet (see goalThreshold) 
        or the current point is out of the images boundaries
        else set finished to true
    '''
    def addSegment(self):
        x = self.points[len(self.points) - 1]                   # current point

        # if goal point distance < goalThreshold Branch is finished or if x is out of the images range
        # add this branch to the treeMap and set the finished property to true
        if (np.mean(np.abs(self.goal - x)) < self.goalThreshold / self.level
                or                                              
            x[0] < 0 or x[0] >= self.tree.sizeX or x[1] < 0 or x[1] >= self.tree.sizeY):

            self.finished = True
            self.tree.treeMap.addBranch(self)                   # add Branch to Map
            
            return
        
        length = np.random.randint(0.0167 * self.tree.sizeX, 0.0833 * self.tree.sizeX) / self.level          # set random length
        i = self.getCurrentGoalPoint(x, length)                 # get currentGoalPoint
        # get random angle to make vessel curly
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2

        rot = self.Rotate(angle)
        newX = np.dot(rot, i - x) + x                           # calculate new Point
        self.points.append(newX)                                # add new Point to branchs points

    '''
        addBranch
        x - starting point from new branch
        this method adds a subbranch to this branch starting from point x
    '''
    def addBranch(self, x):
        newBranch = np.random.rand()                            # roll the dice for new branch

        # if dice was successful and the starting point is not next to another branch
        # add a branch
        if (
            newBranch <  0.5 and 
            not np.array_equal(x, self.points[len(self.points) - 1]) and 
            not self.closeToAnotherBranch(x)
        ):

            g = nextGoalPoint(self, x)                           # get goal point for branch
            if type(g) == np.ndarray:                            # if a goalPoint was found
                # create a branch
                b = Branch(self.tree, x, g, self.level + 1, self.artery)
                self.tree.growingBranches.append(b)             # add branch to tree
                self.tree.branches.append(b)
                
                # if level is greater than 0, first grow the new branch before continuing
                if self.level > 1:
                    while not b.finished:
                        b.addSegment()
                    #print('current mean coverage: ' + str(meanCoverage(self.tree.createTreeMap(), self.tree.sizeX, self.tree.sizeY)) + ' from: '+str(self.tree.covThreshold))

    '''
        getCurrenGoalPoint
        x - starting point (x_x,x_y)
        l - length
        if angle between x -> fovea and fovea -> goal is greater than 90 deg
        assume circle with radius of x -> fovea around fovea
        else just use the direct line between x and g and scale it to the length l
    '''
    def getCurrentGoalPoint(self, x, l):
        r = x - self.tree.fovea                         # radius
        rg = self.goal - self.tree.fovea                # radius to goal

        # if angle is greater than 90 deg use radius from x to fovea
        # else use the direct path to goal
        if np.dot(r, rg) < 0:

            # angle from fovea -> x to fovea -> goal
            theta = l * 180 / (np.pi * (np.linalg.norm(r)) + np.random.randint(-20,10))
            
            # if the goal point is lower than the fovea rotate the radius clockwise,
            # else in the opposite direction
            if self.goal[1] - self.tree.fovea[1] < 0: 
                Rot = self.Rotate(-theta)               # rotation matrix
            else:
                Rot = self.Rotate(theta)
            i = np.dot(Rot, r) + self.tree.fovea        # new segment goal
        else:
            i = self.goal - x                           # direction x -> goalpoint 
            i = (i / np.linalg.norm(i) * l) + x         # set length to l and add vector to x

        return i

    '''
        Rotate
        alpha - angle of rotation
        return 2D rotation matrix
    '''
    def Rotate(self, alpha):
        theta = np.radians(alpha)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    '''
        closeToAnotherBranch
        x - point to check wheter another branch is close to it
    '''
    def closeToAnotherBranch(self, x):
        branches = self.tree.b2arr()                    # get all points of branches of the tree as an np.ndarray
        if branches.shape[0] > 0:                       # if branches has any point compute the shortest distance to x
            shortestDistance = np.min(np.linalg.norm(branches - x))
            return shortestDistance < 1000 / (self.level * self.tree.sizeX) # if the shortest distance is below 1000 it is near to another branch
        else:
            return False