import numpy as np
from utils import showImage, saveImage
from skimage import measure

'''
    nextGoalPoint
    point - starting point for the next branch

    get the coordinates of the next goal point according to some heuristics.
    if the vessel level is 1 the next goal point should either point towards the fovea or 
        in the opposite direction.
    else cluster the area around point and get a cluster center as a new goal point.
'''
def nextGoalPoint(self, point):
    if self.level == 1:
        #showImage(self.tree.createTreeMap())
        pf = point - self.tree.fovea                # direction from fovea to the point
        overFovea = 1 if pf[1] < 0 else -1          # starting point is over the fovea

        # if point is on the left according to the OD,
        # not go to fovea
        # else go to fovea or in opposite direction
        if point[0] - self.tree.opticaldisc[0] > 0:
            
            # normalized vector from starting point to the parents goal point
            goalVector = (self.goal - point) / np.linalg.norm(self.goal - point)

            length = np.random.randint(40, 100)     # random length
            alpha = np.random.randint(30, 70)       # random angle
            left = -1 if np.random.rand() else 1    # roll the dice if the it should go left
            
            # final coordinates
            x, y = np.dot(self.Rotate(left * alpha), goalVector) * length + point

        else:
            x = point[0] -  np.random.randint(20, 60) # random x
            
            toFovea = np.random.rand()              # roll the dice
            
            # if same amount goes to fovea and same goes not in fovea direction 
            # roll the dice else use the fraction of toFovea / notToFovea
            if not self.subBranchesFovea == self.subBranchesNotFovea:
                toFovea = 1 - toFovea / 3 if self.subBranchesFovea > self.subBranchesNotFovea else 0 + toFovea / 3

            # if new branch goes to fovea
            # set y to a point with a random distance to the fovea
            # else select a random point in opposite direction
            if toFovea < 0.5:
                self.subBranchesFovea += 1
                y = self.tree.fovea[1] - overFovea * np.random.randint(10, 30)
            else:
                self.subBranchesNotFovea += 1
                y = point[1] - overFovea * np.random.randint(40, 100)
        return np.array((x, y))
    
    if self.level > 1:
        tmap = self.tree.createTreeMap()                        # current treeMap
        #imgs = [self.tree.createTreeImage()]
        size = 20                                               # window size
        centers = []                                            # all candidates
        new_centers = []
        old_ncenters = 0                                        # amount of old centers

        # increase the window until #(centers) decreased or size is larger than 200
        while len(centers) >= old_ncenters and size < 200:
            # label the window
            img, new_centers, _ = createLabeledImage(size, tmap, point)

            breakWhile = False                                  # finish the search for a fitting point
            for p in centers:                                   # test if a center point has moved more
                if (len(new_centers) > 0 and                    # than it should
                    np.min(np.abs(new_centers - p)) > 3):
                    size -= 5                                   # reduce size again to get last good window
                    breakWhile = True
            
            # break loop if breakWhile or no center was found until a size of 50x50
            if breakWhile or (len(new_centers) == 0 and size > 50):
                break
            
            # set all variables for next iteration
            old_ncenters = len(centers)
            centers = new_centers
            new_centers = []
            size += 5
        
        result = None                                           # init return value
        # label last working image
        img, centers, areas = createLabeledImage(size, tmap, point)

        # if a candidate was found
        if len(areas) > 0:
            max_area = np.where(areas == np.max(areas))[0][0]   # get id for the largest area
            
            result = centers[max_area]  # get coordinates of the center from the largest area

            to_goal = self.goal - point                         # vector point -> parents goal
            to_result = result - point                          # vector point -> center
            normed_g = to_goal / np.linalg.norm(to_goal)
            len_result = np.linalg.norm(to_result)
            normed_result = to_result / len_result

            # if angle between to_goal and to_result > 20 deg, use result as a new goal point
            if normed_g.dot(normed_result) > 0.94:              

                # test if center was already used by comparing all centers in tree.centers with result
                for c in self.tree.centers:
                    if (c[0] - 30 < result[0] and c[0] + 30 > result[0] and
                        c[1] - 30 < result[1] and c[1] + 30 > result[1]):
                        result = None
                        break

        # if result was set append it to tree.centers
        if not result is None:
            self.tree.centers.append(result)
        return result


'''
    createLabeledImage
    size    - width/height of window
    tmap    - binary treeMap
    point   - starting point
    create a labeled window and filter all centers from each region by the size 
    of the area, if point -> center is crossing any vessels, and if area is a neighbour of the vessel
    returns labeled image, all centers, and all areasizes
'''
def createLabeledImage(size, tmap, point):

    img = np.zeros((300,300,4))                 # init labeled image
    centers = []                                # init list of centers
    areas = []                                  # init list of areas
    # get boundaries of the window
    bounds = (np.array([[-size/2, -size/2], [size/2, size/2]]) + point).astype(int)
    bounds[np.where(bounds < 0)] = 0            # set each entry below 0 to 0
    # get labels of all segments on the binary treeMap
    labels = measure.label(tmap[bounds[0, 0]: bounds[1, 0], 
                                bounds[0, 1]: bounds[1, 1], :3], background=255)

    # if labels were found, get centers and len(areas)
    if len(labels) > 0:
        for i in range(1, np.max(labels)):                  #iterate over labels

            # get all positions where the id == i
            ids = np.asarray(np.where(np.sum(labels, axis=2) / 3 == i), int)
            # move the ids to actual position in the image
            ids = (ids.T + point - [size/2, size/2]).astype(int)

            # if the distance of the nearest position of an id is smaller than 3 it is next to point
            # and if there are a minimum of 200 points labeled, calculate the center
            if np.min(np.linalg.norm(np.abs(ids - point), axis=1)) < 3 and ids.shape[0] > 200:
                
                center = np.mean(ids, axis=0)               # get center point

                # if point -> center dont crosses any vessel add center to centerlist
                # and add the length to list of areas
                if not crossingVessel(center, point, tmap):
                    centers.append(center)
                    areas.append(ids.shape[0])

            # apply a random color to all points on the id for displaying the image
            color = np.hstack((np.random.choice(range(255), (3,)), np.array(255)))
            for p in ids:
                # only add the color if the location is part of the image of (300x300px)
                if p[0] < 0 or p[0] > 299 or p[1] < 0 or p[1] > 299:
                    continue
                else:
                    img[p[0], p[1]] = color

    return img, centers, areas

'''
    crossingVessel
    center  - goal point
    point   - starting point
    tmap    - binary treemap
    check wheter on the path from point to center is another vessel.
'''
def crossingVessel(center, point, tmap):
    # if center is out of the image boundaries get largest point in image on
    # the path from point to center
    if center[0] > 299 or center[0] < 0 or center[1] > 299 or center[1] < 0:
        point2center = center - point               # vector point -> center
        for i in np.linspace(1, 0, 100):            # check 100 points on the path
            tmp = point2center * i + point          # get current point
            
            # if they are still out of bounds try next one
            # else use this point
            if tmp[0] > 299 or tmp[0] < 0 or tmp[1] > 299 or tmp[1] < 0:
                continue
            else:
                center = tmp

    # create a list of all points needed to check
    points_needed = np.ceil(np.max(np.abs(center - point)) + 1)
    x_spaced = np.linspace(point[0], center[0], points_needed, dtype=int)
    y_spaced = np.linspace(point[1], center[1], points_needed, dtype=int)
    points = np.vstack((x_spaced, y_spaced)).T

    # check all points in list if they are white (a vessel)
    for p in points[5:]:
        if np.array_equal(tmap[p[0], p[1]], [255, 255, 255, 255]):
            return True
    
    return False