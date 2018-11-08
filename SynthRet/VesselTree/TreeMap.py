from matplotlib.collections import LineCollection
from scipy import interpolate
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from utils import showImage, makeBinary, meanCoverage
from PIL import Image, ImageDraw

'''
    TreeMap
    this class manages the colored vessel image and the binary map
'''
class TreeMap:
    def __init__(self, sizeX, sizeY):
        self.veinColor = np.array((150. / 255, 30. / 255, 10. / 255))
        self.arteryColor = np.array((110. / 255, 10. / 255, 5. / 255))
        self.white = np.array((1., 1., 1.))
        self.vessels = []
        self.treeMap = np.zeros((sizeX,sizeY,4), dtype=np.float32)
        self.treeImage = np.zeros((sizeX,sizeY,4), dtype=np.float32)
        self.sizeX = sizeX
        self.sizeY = sizeY

        self.img = Image.new('RGBA', (self.sizeY, self.sizeX))
        self.mask = Image.new('L', (self.sizeY, self.sizeX))
        self.draw = ImageDraw.Draw(self.img)
        self.drawMask = ImageDraw.Draw(self.mask)

    '''
        addBranch
        branch  -  Branch to add to the image
        updates both images with the branch
    '''
    def addBranch(self, branch):
        # select colors according to branch
        color = self.arteryColor if branch.artery else self.veinColor
        x,y = np.array(zip(*branch.points))     # seperate x and y coordinates from branches.points
        
        s = 0                                   # smoothing condition (0 means passing all points) for interpolation
        k = 3 if x.shape[0] > 3 else x.shape[0]-1 # degree of spline
        if k == 0:
            return
        
        x_len = max(x) - min(x)                 # get length of the vessel in x direction
        y_len = max(y) - min(y)                 # get length of the vessel in y direction
        total_len = np.sqrt(x_len**2 + y_len**2)
        
        # sometimes it throws an error, this should fix it
        try:
            # calculate the splineinterpolation for the branch
            tck, t = interpolate.splprep([x, y], s=s, k=k)
        except:
            return
        # get all points on spline for total_len * 2 to cover each pixel
        xi, yi = interpolate.splev(np.linspace(t[0], t[-1], total_len * 2), tck)
        
        # calculate widths and colors for each point
        r = np.linspace(0, total_len * 2, total_len * 2)
        colors = np.repeat(color[None, :], total_len * 2, axis=0)
        if branch.level == 1:                   # for main vessels
            widths = 0.75 * r / self.sizeX + 2. * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.4, 0.9, total_len * 2)[:, None]))
        elif branch.level == 2:
            widths = 2. * r / self.sizeX + 1.2 * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.3, 0.7, total_len * 2)[:, None]))
        elif branch.level > 2:                                   # for each other vessel
            widths = 1. * r / self.sizeX + .5 * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.2, 0.6, total_len * 2)[:, None]))

        # put points together
        points = np.array([xi, yi]).T.reshape(-1, 1, 2)
        # create array of all lines from x_i to x_i+1
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]

        newVessels = [segments, widths, colors]
        self.vessels.append(newVessels)
        self.update(newVessels)
    
    '''
        update
        updates both images treeImage and treeMap
    '''
    def update(self, l):
        segments = l[0]
        for i in range(0, segments.shape[0]): # todo: draw.line also accepts lists of tuples
            self.draw.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(tuple((l[2][i]*255).astype(int))), width=int(l[1][i] + 0.5))
            self.drawMask.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(255), width=int(l[1][i] + 0.5))

        self.treeImage = np.array(self.img).astype(np.float32) / 255
        # print(self.treeImage.shape)
        treeMap = np.array(self.mask).astype(np.float32) / 255           # make image binary
        notransp = np.ones(treeMap.shape, dtype=np.float32)
        # update treeMap
        self.treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))

    '''
        update
        updates both images treeImage and treeMap
    '''
    def updateAliased(self):
        img = Image.new('RGBA', (self.sizeY * 3, self.sizeX * 3))
        mask = Image.new('L', (self.sizeY * 3, self.sizeX * 3))
        draw = ImageDraw.Draw(img)
        drawMask = ImageDraw.Draw(mask)

        for l in self.vessels:
            segments = l[0] * 3.
            for i in range(0, segments.shape[0]): # todo: draw.line also accepts lists of tuples
                draw.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(tuple((l[2][i]*255).astype(int))), width=int(l[1][i] * 3. + 0.5))
                drawMask.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(255), width=int(l[1][i] * 3. + 0.5))
        
        treeImage = np.array(img).astype(np.float32) / 255
        treeMap = np.array(mask).astype(np.float32) / 255           # make image binary
        notransp = np.ones(treeMap.shape, dtype=np.float32)
        # update treeMap
        treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))
        treeImage = resize(treeImage, (self.sizeX, self.sizeY, 4), anti_aliasing=True).astype(np.float32)
        treeMap = resize(treeMap, (self.sizeX, self.sizeY, 4), anti_aliasing=True).astype(np.float32)
        return treeImage, treeMap

    '''
        getImg
        return the current image
    '''
    def getImg(self):
        assert(self.treeImage.dtype == np.float32)
        return self.treeImage

    '''
        getMap
        return the current binary image
    '''
    def getMap(self):
        assert(self.treeMap.dtype == np.float32)
        return self.treeMap

    '''
        getAliasedImg
        return the current image
    '''
    def getAliasedImgs(self):
        aliasedImg, aliasedMap = self.updateAliased()
        assert(aliasedImg.dtype == np.float32)
        assert(aliasedMap.dtype == np.float32)
        return aliasedImg, aliasedMap