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
    def __init__(self):
        self.veinColor = np.array((221. / 255, 125. / 255, 94. / 255))
        self.arteryColor = np.array((209. / 255, 93. / 255, 74. / 255))
        self.vessels = []
        self.treeMap = np.zeros((300,300,4), dtype=int)
        self.treeImage = np.zeros((300,300,4), dtype=int)

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
        
        # calculate widths for each point
        r = np.linspace(0, total_len * 2, total_len * 2)
        colors = np.repeat(color[None, :], total_len * 2, axis=0)
        if branch.level == 1:                   # for main vessels
            widths = 0.001 * r + 1
            colors = np.hstack((colors, np.linspace(0.4, 0.7, total_len * 2)[:, None]))
        else:                                   # for each other vessel
            widths = 0.003 * r + 0.4
            colors = np.hstack((colors, np.linspace(0.2, 0.5, total_len * 2)[:, None]))

        # put points together
        points = np.array([xi, yi]).T.reshape(-1, 1, 2)
        # create array of all lines from x_i to x_i+1
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]

        self.vessels.append([segments, widths, colors])
        self.updateImg()
    
    '''
        updateImg
        updates both images treeImage and treeMap
    '''
    def updateImg(self):
        fig, ax = plt.subplots(figsize=(3,3), dpi=100)       # init plt
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # set dimensions of plot
        ax.set_xlim(0,300)
        ax.set_ylim(0,300)

        for l in self.vessels:                              # add each vessel to plt
            lc = LineCollection(l[0], linewidths=l[1], color=l[2])
            ax.add_collection(lc)

        # convert plt to np.ndarray
        plt.show(block=False)                               # render plt
        fig.canvas.draw()                                   # draw the canveas
        w,h = fig.canvas.get_width_height()                 # get canvas properties
        assert(w == h)                                      # make sure that resize wont change location of OD

        # save canvas as numpy array in buf
        buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
        plt.close()                                         # close plt
        buf.shape = (w, h, 4)                               # set shape of buffer
        buf = np.roll(buf, 3, axis=2)
        buf = np.transpose(buf, (1,0,2))                    # transpose the image
        buf = transform.resize(buf, (300,300))              # resize image to 300 x 300
        if buf.dtype == float:                              # if buf is of type float convert it to int
            buf = buf * 255

        self.treeImage = buf.astype(int)                    # set image to buf
        treeMap = makeBinary(self.treeImage, 10)            # make image binary
        notransp = np.ones(treeMap.shape) * 255             # convert to int image
        # update treeMap
        self.treeMap = np.dstack((treeMap, treeMap, treeMap, notransp)).astype(int)

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