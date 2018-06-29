from matplotlib.collections import LineCollection
from scipy import interpolate
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from utils import showImage, makeBinary, meanCoverage


class TreeMap:
    def __init__(self):
        #self.arteryColor = (161. / 255, 25. / 255, 44. / 255, 1)
        #self.veinColor = (97. / 255, 33. / 255, 43. / 255, 1)
        self.arteryColor = (161. / 255, 25. / 255, 44. / 255, 0.6)
        self.veinColor = (97. / 255, 33. / 255, 43. / 255, 0.5)
        self.lines = []
        self.treeMap = np.zeros((300,300,4), dtype=int)
        self.treeImage = np.zeros((300,300,4), dtype=int)

    def addBranch(self, branch):
        color = self.arteryColor if branch.artery else self.veinColor
        x,y = np.array(zip(*branch.points))     # seperate x and y coordinates from Branches
        
        # interpolate 
        s = 0   # smoothing condition (0 means passing all points)
        k = 3 if x.shape[0] > 3 else x.shape[0]-1
        if k == 0:
            return
        
        x_len = max(x) - min(x)
        y_len = max(y) - min(y)
        total_len = np.sqrt(x_len**2 + y_len**2)
        
        try:

            tck, t = interpolate.splprep([x, y], s=s, k=k)
        except:
            return
        xi, yi = interpolate.splev(np.linspace(t[0], t[-1], total_len * 2), tck)
        
        r = np.linspace(0, total_len * 2, total_len * 2)
        if branch.level == 1:
            widths = 0.003 * r + 0.7
        else:
            widths = 0.003 * r + 0.3
        points = np.array([xi, yi]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]
        self.lines.append([segments, widths, color])
        self.updateImg()
    
    def updateImg(self):
        fig, ax = plt.subplots(figsize=(3,3),dpi=100)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        # set dimensions of plot
        ax.set_xlim(0,300)
        ax.set_ylim(0,300)

        for l in self.lines:
            lc = LineCollection(l[0], linewidths=l[1], color=l[2])
            ax.add_collection(lc)

        plt.show(block=False)
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        plt.close()
        buf.shape = ( w, h, 4 )
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        buf = np.transpose(buf, (1,0,2))
        buf = transform.resize(buf, (300,300,4))
        #print buf.shape
        if buf.dtype == float:
            buf = buf * 255
        self.treeImage = buf.astype(int)
        treeMap = makeBinary(self.treeImage, 10)
        notransp = np.ones(treeMap.shape) * 255
        self.treeMap = np.dstack((treeMap, treeMap, treeMap, notransp)).astype(int)
        #showImage(self.treeImage, sec=0.01)
        #print meanCoverage(self.treeMap, None)

    def getImg(self):
        return self.treeImage

    def getMap(self):
        return self.treeMap