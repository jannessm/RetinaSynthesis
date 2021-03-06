from scipy import interpolate
import numpy as np
from PIL import Image, ImageDraw


class TreeMap:
    '''
        TreeMap
        this class manages the colored vessel image and the binary map
    '''
    
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

    def addBranch(self, branch):
        '''
            addBranch
            branch  -  Branch to add to the image
            updates both images with the branch
        '''
        # select colors according to branch
        color = self.arteryColor if branch.artery else self.veinColor
        
        # this replacement seems to do the same
        x_y = np.array(branch.points)
        x = x_y[:,0]
        y = x_y[:,1]
        
        s = 0                                   # smoothing condition (0 means passing all points) for interpolation
        k = 3 if x.shape[0] > 3 else x.shape[0]-1 # degree of spline
        if k == 0:
            return
        
        x_len = max(x) - min(x)                 # get length of the vessel in x direction
        y_len = max(y) - min(y)                 # get length of the vessel in y direction
        total_len = int(np.ceil(np.sqrt(x_len**2 + y_len**2) * 2))
        
        # sometimes it throws an error, this should fix it
        try:
            # calculate the splineinterpolation for the branch
            tck, t = interpolate.splprep([x, y], s=s, k=k)
        except:
            return
        
        # get all points on spline for total_len * 2 to cover each pixel
        xi, yi = interpolate.splev(np.linspace(t.astype(np.int)[0], t.astype(np.int)[-1], total_len), tck)
        
        # calculate widths and colors for each point
        r = np.linspace(0, total_len, total_len)
        colors = np.repeat(color[None, :], total_len, axis=0)
        if branch.level == 1:                   # for main vessels
            widths = 2 * r / self.sizeX + 2. * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.4, 0.9, total_len)[:, None]))
        elif branch.level == 2:
            widths = 2. * r / self.sizeX + 1.2 * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.3, 0.7, total_len)[:, None]))
        elif branch.level > 2:                                   # for each other vessel
            widths = 1. * r / self.sizeX + .5 * self.sizeX / 565.
            colors = np.hstack((colors, np.linspace(0.2, 0.6, total_len)[:, None]))

        # put points together
        points = np.array([xi, yi]).T.reshape(-1, 1, 2)
        # create array of all lines from x_i to x_i+1
        segments = np.concatenate([points[:-1], points[1:]], axis=1)[::-1]

        newVessels = [segments, widths, colors]
        self.vessels.append(newVessels)
        self.update(newVessels)
    
    def update(self, l):
        '''
            update
            updates both images treeImage and treeMap
        '''
        segments = l[0]
        for i in range(0, segments.shape[0]): # todo: draw.line also accepts lists of tuples
            self.draw.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(tuple((l[2][i]*255).astype(int))), width=int(l[1][i] + 0.5))
            self.drawMask.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(255), width=int(l[1][i] + 0.5))

        self.treeImage = np.array(self.img).astype(np.float32) / 255
        
        treeMap = np.array(self.mask).astype(np.float32) / 255           # make image binary
        notransp = np.ones(treeMap.shape, dtype=np.float32)
        
        # update treeMap
        self.treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))

    def createSupersampledImages(self, scalingFactor):
        '''
            update
            updates both images treeImage and treeMap
        '''
        img = Image.new('RGB', (self.sizeY * scalingFactor, self.sizeX * scalingFactor))
        imgAlpha = Image.new('L', (self.sizeY * scalingFactor, self.sizeX * scalingFactor))
        mask = Image.new('L', (self.sizeY * scalingFactor, self.sizeX * scalingFactor))
        draw = ImageDraw.Draw(img)
        drawAlpha = ImageDraw.Draw(imgAlpha)
        drawMask = ImageDraw.Draw(mask)

        for l in self.vessels:
            segments = l[0] * scalingFactor
            
            segmentDirections = segments[:,1,:] - segments[:,0,:]
            scale = 1.0 / np.sqrt(segmentDirections[:,0]*segmentDirections[:,0] + segmentDirections[:,1]*segmentDirections[:,1])
            segmentDirections[:,0] = segmentDirections[:,0] * scale
            segmentDirections[:,1] = segmentDirections[:,1] * scale
            
            segments[:,1,:] = segments[:,1,:] + segmentDirections * 3
            
            for i in range(0, segments.shape[0]):
                draw.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(tuple((l[2][i]*255).astype(int))), width=int(l[1][i] * scalingFactor + 0.5))
                alpha = 10+int(l[1][i] * 20)
                if alpha > 255:
                    alpha = 255
                drawAlpha.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=alpha, width=int(l[1][i] * scalingFactor + 0.5))
                if l[1][i]*scalingFactor >= 2.5:
                    drawAlpha.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=alpha//3, width=int(l[1][i] * scalingFactor/2.5 + 0.5))
                drawMask.line((segments[i, 0, 1], segments[i, 0, 0], segments[i, 1, 1], segments[i, 1, 0]), fill=(255), width=max(int(l[1][i] * scalingFactor*1.1 + 0.5), int(1.2*scalingFactor)))
        
        img = np.array(img).astype(np.float32) / 255
        imgAlpha = np.array(imgAlpha).astype(np.float32) / 255
        treeImage = np.dstack((img[:,:,0], img[:,:,1], img[:,:,2], imgAlpha))
        
        treeMap = np.array(mask).astype(np.float32) / 255           # make image binary
        notransp = np.ones(treeMap.shape, dtype=np.float32)
        # update treeMap
        treeMap = np.dstack((treeMap, treeMap, treeMap, notransp))
        return treeImage, treeMap

    def getImg(self):
        '''
            getImg
            return the current image
        '''
        assert(self.treeImage.dtype == np.float32)
        return self.treeImage
    
    def getMap(self):
        '''
            getMap
            return the current binary image
        '''
        assert(self.treeMap.dtype == np.float32)
        return self.treeMap
        
