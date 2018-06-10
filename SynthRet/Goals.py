import numpy as np
from utils import showImage
from skimage import measure

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
        # tmap = self.tree.createTreeMap(unicolor=True)
        # size = 30
        # centers = []
        # new_centers = []
        # old_ncenters = 0
        # while len(centers) >= old_ncenters and size < 300:
        #     img, new_centers, _ = createLabeledImage(size, tmap, point)

        #     breakWhile = False
        #     for p in centers:
        #         if (len(new_centers) > 0 and 
        #             np.min(np.abs(new_centers - p)) > 4):
        #             size -= 5
        #             breakWhile = True
            
        #     if breakWhile:
        #         break
            
        #     old_ncenters = len(centers)
        #     centers = new_centers
        #     new_centers = []
        #     size += 5
        
        # img, centers, areas = createLabeledImage(size, tmap, point)
        result = None
        # if len(areas) > 0:
        #     max_area = np.where(areas == np.max(areas))[0][0]
            
        #     result = centers[max_area]

        #     to_goal = self.goal - point
        #     to_result = result - point
        #     normed_g = to_goal / np.linalg.norm(to_goal)
        #     len_result = np.linalg.norm(to_result)
        #     normed_result = to_result / len_result
        #     if normed_g.dot(normed_result) > 0.94:
        #         alpha = np.random.randint(20,70)
        #         result = self.Rotate(alpha).dot(to_result) + point

        #     showImage(img.astype(int), [point, result, self.goal])
        return result
    
def createLabeledImage(size, tmap, point):

    img = np.zeros((300,300,4))
    centers = []
    areas = []
    bounds = (np.array([[-size/2, -size/2], [size/2, size/2]]) + point).astype(int)
    labels = measure.label(tmap[bounds[0, 0]: bounds[1, 0], 
                                bounds[0, 1]: bounds[1, 1], :3], background=255)

    for i in range(1, np.max(labels)):

        ids = np.asarray(np.where(np.sum(labels, axis=2) / 3 == i), int)
        ids = (ids.T + point - [size/2, size/2]).astype(int)
        if np.min(np.linalg.norm(np.abs(ids - point), axis=1)) < 5 and ids.shape[0] > 300:
            centers.append(np.mean(ids, axis=0))
            areas.append(ids.shape[0])
        color = np.hstack((np.random.choice(range(255), (3,)), np.array(255)))
        for p in ids:
            
            if p[0] < 0 or p[0] > 299 or p[1] < 0 or p[1] > 299:
                continue
            else:
                img[p[0], p[1]] = color
    
    # if len(centers) > 0:
    #     a = centers[:]
    #     a.append(point)
    #     showImage(img.astype(int), a)
    # else:
    #     showImage(img.astype(int))

    return img, centers, areas