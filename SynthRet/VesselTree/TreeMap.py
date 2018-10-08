from matplotlib.collections import LineCollection
from scipy import interpolate
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from utils import showImage, makeBinary, meanCoverage

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
        self.treeMap = np.zeros((sizeX,sizeY,4), dtype=int)
        self.treeImage = np.zeros((sizeX,sizeY,4), dtype=int)
        self.sizeX = sizeX
        self.sizeY = sizeY

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
        black_white = np.repeat(self.white[None, :], total_len * 2, axis=0)
        black_white = np.hstack((black_white, np.linspace(1, 1, total_len * 2)[:, None]))
        if branch.level == 1:                   # for main vessels
            widths = 0.6 * r / self.sizeX + 1.5
            colors = np.hstack((colors, np.linspace(0.4, 0.9, total_len * 2)[:, None]))
        else:                                   # for each other vessel
            widths = 0.8 * r / self.sizeX + 0.3
            colors = np.hstack((colors, np.linspace(0.3, 0.7, total_len * 2)[:, None]))

        # put points together
        points = np.array([xi, yi]).T.reshape(-1, 1, 2)
        # create array of all lines from x_i to x_i+1
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]

        self.vessels.append([segments, widths, colors, black_white])
        self.updateImg()
        self.updateMap()
    
    '''
        updateImg
        updates both images treeImage and treeMap
    '''
    def updateImg(self):
        self.treeImage = self._update('color')

    def updateMap(self):
        self.treeMap = self._update('black_white')

    def _update(self, color):
        color_id = 3 if color == 'black_white' else 2
        fig, ax = plt.subplots(figsize=(self.sizeX/100,self.sizeY/100), dpi=100)       # init plt
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # set dimensions of plot
        ax.set_xlim(0,self.sizeX)
        ax.set_ylim(0,self.sizeY)

        for l in self.vessels:                              # add each vessel to plt
            lc = LineCollection(l[0], linewidths=l[1], color=l[color_id])
            ax.add_collection(lc)

        # convert plt to np.ndarray
        plt.show(block=False)                               # render plt
        fig.canvas.draw()                                   # draw the canveas
        w,h = fig.canvas.get_width_height()                 # get canvas properties
        
        # save canvas as numpy array in buf
        buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
        plt.close()                                         # close plt
        buf.shape = (w, h, 4)                               # set shape of buffer
        buf = np.roll(buf, 3, axis=2)
        buf = np.transpose(buf, (1,0,2))                    # transpose the image
        buf = transform.resize(buf, (self.sizeX,self.sizeY))              # resize image wished size
        buf = np.fliplr(buf)                                # correct orientation
        if buf.dtype == float:                              # if buf is of type float convert it to int
            buf = buf * 255

        return buf.astype(int)                              # return result buf

    '''
        getImg
        return the current image
    '''
    def getImg(self):
        return self.treeImage

    '''
        getMap
        return the current binary image
    '''
    def getMap(self):
        return self.treeMap