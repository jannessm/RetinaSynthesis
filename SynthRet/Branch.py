import numpy as np

class Branch:
    def __init__(self, tree, startingPoint, goalPoint):
        self.points = [startingPoint]
        self.goal = goalPoint
        self.tree = tree

        #attributes
        self.finished = False

        #constants
        self.goalThreshold = 5  #pixels away from goal point are sufficent
        self.maxAngle = 60      #upper boundary for random angle
        self.covThreshold = 0.9 #threshold for total coverage
    
    def addSegment(self):
        x = self.points[len(self.points) - 1]            #current point
        if np.abs(self.goal - x < self.goalThreshold):
            self.finished = True
            return
        
        length = np.random.randint(1, 10)
        intersection = self.getArc(length)
        cov = self.coverage()
        angle = np.random.rand() * self.maxAngle - self.maxAngle / 2
        
        newBranch = np.random.rand()
        if np.mean(cov) < self.covThreshold and newBranch < np.mean(cov):
            g = self.nearestUncoveredArea(cov)
            self.tree.addBranch(x, g)

        newX = x + self.Rotate(angle) * intersection
        self.points.append(newX)

    def getArc(self, len):
        # TODO
        return [1, 1]

    def coverage(self):
        # TODO
        return np.ones((300,300,4))
    
    def nearestUncoveredArea(self, coverageMap):
        x = [1, 1]
        # TODO
        return x

    def Rotate(self, alpha):
        theta = np.radians(alpha)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))