import numpy as np
from sklearn.cluster import KMeans
from utils import showImage

class Branch:
    def __init__(self, tree, startingPoint, goalPoint):
        self.points = [np.array(startingPoint)]
        self.goal = np.array(goalPoint)                 # goal point
        self.tree = tree                                # tree object

        #attributes
        self.finished = False
        self.branches = 0

        #constants
        self.goalThreshold = 10                         # pixels away from goal point are sufficent
        self.maxAngle = 60                              # upper boundary for random angle
        self.covThreshold = 0.9                         # threshold for total coverage
    

    '''
        addSegment
        adds a segment to the current branch if goal point is not reached yet (see goalThreshold)
        else set finished to true
    '''
    def addSegment(self):
        x = self.points[len(self.points) - 1]               # current point
        if np.mean(np.abs(self.goal - x)) < self.goalThreshold: # if goal point distance < goalThreshold Branch is finished
            self.finished = True
            return
        
        length = np.random.randint(20, 30)                   # set random length
        i = self.getCurrentGoalPoint(x, length)              # get currentGoalPoint
        cov = self.tree.coverage()                           # update coverage
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2 # set random angle around currentGoalPoint
        
        #TODO implement new branches
        newBranch = np.random.rand()                        # roll the dice for new branch
        if newBranch < 0.5 and self.branches < 6:
            self.branches += 1
            g = self.nearestUncoveredArea(x)                # get goal point for branch
            self.tree.addBranch(x, g)                       # add a branch to queue

        rot = self.Rotate(angle)
        newX = np.dot(rot, i - x) + x                        # calculate new Point to branch
        self.points.append(newX)                             # add new Point to queue


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
            theta = l * 180 / (np.pi * np.linalg.norm(r)) # angle from fovea -> x to fovea -> goal
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
        nearestUncoveredArea
        get the coordinates of the nearest uncovered area according to x
    '''
    def nearestUncoveredArea(self, point):
        coverageMap = self.tree.coverage()
        
        ids = np.where(coverageMap < 200)
        X = np.vstack(ids).T

        km = KMeans(n_clusters=10)
        km.fit(X)
        x = km.cluster_centers_[km.predict([point])[0]]

        return x

    '''
        Rotate
        alpha angle of rotation
        return rotation matrix
    '''
    def Rotate(self, alpha):
        theta = np.radians(alpha)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))