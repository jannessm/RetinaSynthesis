import numpy as np
from utils import showImage
from Goals import nextGoalPoint

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
        self.subBranchesFovea = 0
        self.subBranchesNotFovea = 0

        #constants
        self.goalThreshold = 10                         # pixels away from goal point are sufficent
        self.maxAngle = 60                              # upper boundary for random angle
        self.covThreshold = 0.9                         # threshold for total coverage
    

    def setLevel(self, l):
        self.level = l

    '''
        addSegment
        adds a segment to the current branch if goal point is not reached yet (see goalThreshold)
        else set finished to true
    '''
    def addSegment(self):
        x = self.points[len(self.points) - 1]                   # current point
        if (np.mean(np.abs(self.goal - x)) < self.goalThreshold / self.level # if goal point distance < goalThreshold Branch is finished
                or                                              # or
            x[0] < 0 or x[0] > 299 or x[1] < 0 or x[1] > 299):  # if x is out of the image
            self.finished = True
            self.tree.treeMap.addBranch(self)                   # add Branch to Map
            return
        
        length = np.random.randint(5, 25) / self.level          # set random length
        i = self.getCurrentGoalPoint(x, length)                 # get currentGoalPoint
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2 # set random angle around currentGoalPoint

        rot = self.Rotate(angle)
        newX = np.dot(rot, i - x) + x                           # calculate new Point to branch
        self.points.append(newX)                                # add new Point to queue

    def addBranch(self, x):
        newBranch = np.random.rand()                            # roll the dice for new branch
        if (newBranch <  0.5 and 
            not np.array_equal(x, self.points[len(self.points) - 1]) and 
            not self.closeToAnotherBranch(x)):

            g = nextGoalPoint(self, x)                           # get goal point for branch
            if type(g) == np.ndarray:
                b = Branch(self.tree, x, g, self.level + 1, self.artery)
                
                self.tree.growingBranches.append(b)
                self.tree.branches.append(b)
                
                if self.level > 0:
                    while not b.finished:
                        b.addSegment()
                    #showImage(self.tree.createTreeImage())

    '''
        getCurrenGoalPoint
        x -> starting point (x_x,x_y)
        l -> length
        if angle between x -> fovea and fovea -> goal is greater than 90 deg
        assume circle with radius of x -> fovea around fovea
        else just use the direct line between x and g and scale it to the length l
    '''
    def getCurrentGoalPoint(self, x, l):
        r = x - self.tree.fovea                         # radius
        rg = self.goal - self.tree.fovea                # radius to goal
        if np.dot(r, rg) < 0:
            theta = l * 180 / (np.pi * (np.linalg.norm(r)) + np.random.randint(-20,10)) # angle from fovea -> x to fovea -> goal
            blub = self.goal[1] - self.tree.fovea[1]
            if blub < 0: 
                Rot = self.Rotate(-theta)               # current goal point if r=1 (identity circle)
            else:
                Rot = self.Rotate(theta)
            i = np.dot(Rot, r) + self.tree.fovea        # goal point shifted by fovea location
        else:
            i = self.goal - x
            i = (i / np.linalg.norm(i) * l) + x         # set length to l and add vector to x

        return i

    '''
        Rotate
        alpha angle of rotation
        return rotation matrix
    '''
    def Rotate(self, alpha):
        theta = np.radians(alpha)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def closeToAnotherBranch(self, x):
        branches = self.tree.b2arr()
        if branches.shape[0] > 0:
            shortestDistance = np.min(np.linalg.norm(branches - x))
            #showImage(self.tree.createTreeMap(), branches, pointsYellow=[x], sec=0.1)
            return shortestDistance < 1000
        else:
            return False