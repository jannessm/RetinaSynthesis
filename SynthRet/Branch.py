import numpy as np
from sklearn.cluster import KMeans
from utils import showImage

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
        x = self.points[len(self.points) - 1]               # current point
        if (np.mean(np.abs(self.goal - x)) < self.goalThreshold # if goal point distance < goalThreshold Branch is finished
                or                                              # or
            x[0] < 0 or x[0] > 299 or x[1] < 0 or x[1] > 299):  # if x is out of the image
            self.finished = True
            return
        
        length = np.random.randint(5, 25) / self.level                   # set random length
        i = self.getCurrentGoalPoint(x, length)              # get currentGoalPoint
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2 # set random angle around currentGoalPoint

        rot = self.Rotate(angle)
        newX = np.dot(rot, i - x) + x                        # calculate new Point to branch
        self.points.append(newX)                             # add new Point to queue

    def addBranch(self, x):
        #TODO implement new branches
        newBranch = np.random.rand()                        # roll the dice for new branch
        if newBranch <  0.5 and not np.array_equal(x, self.points[len(self.points) - 1]):
            self.tree.nbranches += 1
            g = self.nearestUncoveredArea(x)                # get goal point for branch
            if type(g) == np.ndarray:
                self.tree.addBranch(x, g, self.level + 1, self.artery)          # add a branch to queue


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
        nearestUncoveredArea
        get the coordinates of the nearest uncovered area according to x
    '''
    def nearestUncoveredArea(self, point):
        if self.level == 1:
            pf = point - self.tree.fovea

            # if point is on the opposite site of the fovea according to the od,
            # not go to fovea
            if point[0] - self.tree.opticaldisc[0] < 0:
                x = np.random.randint(-70, -20) + point[0]
            # else go to fovea or in opposite direction
            else:
                x = np.random.randint(10,40) + point[0]
            
            toFovea = np.random.rand()
            overFovea = 1 if pf[1] < 0 else -1
            if toFovea < 0.5:
                y = self.tree.fovea[1] - overFovea * np.random.randint(10,20)
            else:
                y = point[1] - overFovea * np.random.randint(10, 200 / (2 * self.level))
            return np.array((x, y))
        if self.level > 1:
            # parentDirection = self.goal - point
            # angle = np.random.randint(20, 60)
            # k_l = self.Rotate(-angle).dot(np.array((- parentDirection[1], parentDirection[0])))
            # k_l = k_l / np.linalg.norm(k_l)
            # k_r = self.Rotate(angle).dot(np.array((parentDirection[1], - parentDirection[0])))
            # k_r = k_r / np.linalg.norm(k_r)
            # tmap = self.tree.createTreeMap()
            # for l in range(3, 100):
            #     p = (k_l*l + point).astype(int)
            #     showImage(tmap, [p, point], 0.5)
            #     if p[0] > 299 or p[0] < 0 or p[1] > 299 or p[1] < 0 or tmap[p[0], p[1], 0] == 255:
            #         break
            # for r in range(3, 100):
            #     p = (k_r*r + point).astype(int)
            #     showImage(tmap, [p, point], 0.5)
            #     if p[0] > 299 or p[0] < 0 or p[1] > 299 or p[1] < 0 or tmap[p[0], p[1], 0] == 255:
            #         break
            # if np.max(np.array((l, r))) < 10:
            #     return None
            # if l < r:
            #     return point + k_r*r/2
            # else:
            #     return point + k_l*l/2
            return None

    '''
        Rotate
        alpha angle of rotation
        return rotation matrix
    '''
    def Rotate(self, alpha):
        theta = np.radians(alpha)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))