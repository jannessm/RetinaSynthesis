import numpy as np

class Branch:
    def __init__(self, tree, startingPoint, goalPoint):
        self.points = [startingPoint]
        self.goal = goalPoint
        self.tree = tree

        #attributes
        self.finished = False

        #constants
        self.goalThreshold = 5                          # pixels away from goal point are sufficent
        self.maxAngle = 60                              # upper boundary for random angle
        self.covThreshold = 0.9                         # threshold for total coverage
    
    '''
        addSegment
        adds a segment to the current branch if goal point is not reached yet (see goalThreshold)
        else set finished to true
    '''
    def addSegment(self):
        x = self.points[len(self.points) - 1]               # current point
        if np.abs(self.goal - x < self.goalThreshold):      # if goal point distance < goalThreshold Branch is finished
            self.finished = True
            return
        
        length = np.random.randint(1, 10)                   # set random length
        i = self.getCurrentGoalPoint(x, length)             # get currentGoalPoint
        cov = self.tree.coverage()                           # update coverage
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2 # set random angle around currentGoalPoint
        
        newBranch = np.random.rand()                        # roll the dice for new branch
        if (np.mean(cov) < self.covThreshold and            # if new Branch and some uncovered area left
            newBranch < np.mean(cov)):
            g = self.nearestUncoveredArea(cov)              # get goal point for branch
            self.tree.addBranch(x, g)                       # add a branch to queue

        newX = x + self.Rotate(angle) * i                   # calculate new Point to branch
        self.points.append(newX)                            # add new Point to queue

    '''
        getCurrenGoalPoint
        x -> starting point (x_x,x_y)
        l -> length
        if angle between x -> fovea and fovea -> goal is greater than 90° 
        assume circle with radius of x -> fovea around fovea
        else just use the direct line between x and g and scale it to the length l
    '''
    def getCurrentGoalPoint(self, x, l):
        r = x - self.tree.fovea                         # radius

        if np.dot(r, self.tree.fovea - self.goal) < 0:
            theta = l * 180 / (np.pi * r)               # angle from fovea -> x to fovea -> goal
            s, c = np.sin(theta), np.cos(theta)         # current goal point if r=1 (identity circle)
            i = np.array((r * c, r * s)) + self.tree.fovea # goal point shifted by fovea location
        else:
            i = x - self.goal
            i = (i / np.linalg.norm(i) * l) + x         # set length to l and add vector to x

        return i
    
    '''
        nearestUncoveredArea
        get the coordinates of the nearest uncovered area according to x
    '''
    def nearestUncoveredArea(self, coverageMap):
        x = [1, 1]
        # TODO
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