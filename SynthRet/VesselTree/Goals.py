import numpy as np
from utils import showImage, saveImage
from skimage import measure

'''
    nextGoalPoint
    get the coordinates of the next goal point according to some heuristics.

    if the vessel level is 1 the next goal point should either point towards the fovea or 
        in the opposite direction.
    else cluster the area around point and get a cluster center as a new goal point.
'''
def nextGoalPoint(self, point):
    if self.level == 1:
        #showImage(self.tree.createTreeMap())
        pf = point - self.tree.fovea                # direction from fovea to the point
        overFovea = 1 if pf[1] < 0 else -1

        # if point is on the opposite site of the fovea according to the od,
        # not go to fovea
        # else go to fovea or in opposite direction
        if point[0] - self.tree.opticaldisc[0] > 0:
            goalVector = (self.goal - point) / np.linalg.norm(self.goal - point)
            length = np.random.randint(40, 100)
            alpha = np.random.randint(30, 70)
            left = -1 if np.random.rand() else 1
            x, y = np.dot(self.Rotate(left * alpha), goalVector) * length + point

        else:
            x = point[0] -  np.random.randint(20, 60)

            # if same amount goes to fovea and same goes not in fovea direction roll the dice else use the fraction of toFovea / notToFovea
            toFovea = np.random.rand()
            if not self.subBranchesFovea == self.subBranchesNotFovea:
                toFovea = 1 - toFovea / 3 if self.subBranchesFovea > self.subBranchesNotFovea else 0 + toFovea / 3

            if toFovea < 0.5:
                self.subBranchesFovea += 1
                y = self.tree.fovea[1] - overFovea * np.random.randint(10, 30)
            else:
                self.subBranchesNotFovea += 1
                y = point[1] - overFovea * np.random.randint(40, 100)
        return np.array((x, y))
    
    if self.level > 1:
        tmap = self.tree.createTreeMap()
        #imgs = [self.tree.createTreeImage()]
        size = 20
        centers = []
        new_centers = []
        old_ncenters = 0
        while len(centers) >= old_ncenters and size < 200:
            img, new_centers, _ = createLabeledImage(size, tmap, point)

            breakWhile = False
            for p in centers:
                if (len(new_centers) > 0 and 
                    np.min(np.abs(new_centers - p)) > 3):
                    size -= 5
                    breakWhile = True
            
            if breakWhile or (len(new_centers) == 0 and size > 50):
                break
            
            old_ncenters = len(centers)
            centers = new_centers
            new_centers = []
            size += 5
            #imgs.append(img.astype(int))
        
        result = None
        img, centers, areas = createLabeledImage(size, tmap, point)
        #imgs.append(img.astype(int))
        if len(areas) > 0:
            max_area = np.where(areas == np.max(areas))[0][0]
            
            result = centers[max_area]

            to_goal = self.goal - point
            to_result = result - point
            normed_g = to_goal / np.linalg.norm(to_goal)
            len_result = np.linalg.norm(to_result)
            normed_result = to_result / len_result
            if normed_g.dot(normed_result) > 0.94:
                alpha = np.random.randint(20,70)
                result = self.Rotate(alpha).dot(to_result) + point
        #saveImage(imgs, imagePath="./", png=True)
        return result
    
def createLabeledImage(size, tmap, point):

    img = np.zeros((300,300,4))
    centers = []
    areas = []
    bounds = (np.array([[-size/2, -size/2], [size/2, size/2]]) + point).astype(int)
    bounds[np.where(bounds < 0)] = 0
    labels = measure.label(tmap[bounds[0, 0]: bounds[1, 0], 
                                bounds[0, 1]: bounds[1, 1], :3], background=255)

    if len(labels) > 0:
        for i in range(1, np.max(labels)):

            ids = np.asarray(np.where(np.sum(labels, axis=2) / 3 == i), int)
            ids = (ids.T + point - [size/2, size/2]).astype(int)
            if np.min(np.linalg.norm(np.abs(ids - point), axis=1)) < 3 and ids.shape[0] > 200:
                
                center = np.mean(ids, axis=0)

                if not crossingVessel(center, point, tmap):
                    centers.append(center)
                    areas.append(ids.shape[0])

            color = np.hstack((np.random.choice(range(255), (3,)), np.array(255)))
            for p in ids:
                
                if p[0] < 0 or p[0] > 299 or p[1] < 0 or p[1] > 299:
                    continue
                else:
                    img[p[0], p[1]] = color

    return img, centers, areas

def crossingVessel(center, point, tmap):
    if center[0] > 299 or center[0] < 0 or center[1] > 299 or center[1] < 0:
        point2center = center - point
        for i in np.linspace(1, 0, 100):
            tmp = point2center * i + point
            if tmp[0] > 299 or tmp[0] < 0 or tmp[1] > 299 or tmp[1] < 0:
                continue
            else:
                center = tmp
    points_needed = np.ceil(np.max(np.abs(center - point)) + 1)
    x_spaced = np.linspace(point[0], center[0], points_needed, dtype=int)
    y_spaced = np.linspace(point[1], center[1], points_needed, dtype=int)
    points = np.vstack((x_spaced, y_spaced)).T

    for p in points[5:]:
        if np.array_equal(tmap[p[0], p[1]], [255, 255, 255, 255]):
            return True
    
    return False