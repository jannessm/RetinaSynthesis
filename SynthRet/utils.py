import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, transform, draw
from skimage.io import imread, imsave
import os
import math
import random

np.seterr(divide="ignore", invalid="ignore")
def mergeLayer(collect):
    '''
        mergeLayer
        collect - list of rgba integer images 
        merges all rgba images in collect. the first will be the lowest image
    '''

    # change each img to float
    for i in range(len(collect)):
        assert(collect[i].dtype == np.float32)

    dimg = collect[0]                               # init lowest layer
    for idx in range(1, len(collect)):
        img = collect[idx]            
        
        # a (img) over b (dimg) (porter-duff-algorithm)
        a_a = img[:, :, 3][:, :, np.newaxis]
        a_b = dimg[:, :, 3][:, :, np.newaxis]
        a_c = a_a + (1 - a_a) * a_b
        a_c[ a_c == 0 ] = 1                         # make sure no division by 0 is happening

        a_A = np.multiply(img, a_a)
        a_B = np.multiply(dimg, a_b)
        dimg = np.divide(a_A + np.multiply(a_B, 1 - a_a), a_c)
    return dimg

def makeBinary(img, threshold):
    '''
        makeBinary
        img     - image to make binary
        threshold - the threshold
        make an image binary by a given threshold
    '''
    # if image is rgba convert to rgb and split to r,g,b
    if img.shape[2] == 4:
        r = np.multiply(img[:, :, 0], img[:, :, 3]) 
        g = np.multiply(img[:, :, 1], img[:, :, 3]) 
        b = np.multiply(img[:, :, 2], img[:, :, 3])
        rgb = np.dstack((r, g, b)) # put them back together
    else:
        rgb = img[:]
    
    # convert to greyscale
    grey = np.multiply(rgb, [0.21, 0.72, 0.07])
    grey = np.sum(grey, axis=2)

    # apply threshold
    binary = np.ones(grey.shape, dtype=np.float32)
    binary[np.where(grey > threshold)] = 1
    binary[np.where(grey < threshold)] = 0

    return binary

def addIllumination(image, groundtruth): #detail adjustment
    '''
        addIllumination
        image - image to add illumination to
    '''

    # mirror by probability of 0.5
    if np.random.rand() < 0.5:
        image = np.flipud(image)
        groundtruth = np.flipud(groundtruth)
        
        
    chromaticAbberation = True
    if chromaticAbberation:
        # FU python
        image = image * 0.01

        center = np.array([[image.shape[0]/2, image.shape[1]/2]])
        scale = 1.0 / (image.shape[0]*image.shape[0])

        def computeDistortionMap(coeff, pairs):
            pairs = pairs - center
            sqrD = np.sum(pairs**2, axis=1)
            pairs = pairs * (1.0 + sqrD[:,np.newaxis] * scale * coeff) + center
            return pairs
            
        def radialDistortion(coeff):
            return lambda pairs : computeDistortionMap(coeff, pairs)
        
        kappa = random.uniform(-0.04, 0.04)
        
        image[:,:,0] = transform.warp(image[:,:,0], 
                             radialDistortion(-kappa))
        image[:,:,2] = transform.warp(image[:,:,2], 
                             radialDistortion(kappa))

        # FU python
        image = image * 100

    return np.clip(image*255, 0, 255), np.clip(groundtruth*255, 0, 255)

def showImage(img, pointsBlue=None, pointsYellow=None, sec=-1):
    '''
        showImage

        img - image or list of images to plot
        pointsBlue - list of points to plot with blue color
        pointsYellow - list of points to plot with yellow color
        sec - seconds the plot is shown. if sec == -1 it is show until it is closed
        plots images into a plot and 
    '''
    if type(img) == list:
        pointsBlue = pointsBlue if type(pointsBlue) == list else [None] * len(img)
        pointsYellow = pointsYellow if type(pointsYellow) == list else [None] * len(img)
        rows = np.floor(np.sqrt(len(img)))
        cols = np.ceil(np.sqrt(len(img))) 
        if not rows > 1:
            rows = 1
            cols = len(img)
        for i in range(len(img)):
            plt.subplot(int(rows), int(cols), i+1)
            _plotHelper(img[i], pointsBlue[i], pointsYellow[i])
    else:
        _plotHelper(img, pointsBlue, pointsYellow)
    if not sec == -1:
        plt.show(block=False)
        plt.pause(sec)
        plt.close()
    else:
        plt.show()

def _plotHelper(img, pointsBlue, pointsYellow):
    '''
        subfunction of showImage()
        img - Image to show
        pointsBlue - points that are displayed blue
        pointsYellow - points that are displayed yellow
        plots the given image and points into a plt
    '''
    plt.axis("off")
    plt.gcf().patch.set_alpha(0.0)
    plt.gca().patch.set_alpha(0.0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    if img.ndim == 3:
        plt.imshow(np.transpose(img, (1,0,2)))   #show transposed so x is horizontal and y is vertical
    else:
        plt.imshow(img)
    if pointsBlue is not None:
        x, y = zip(*pointsBlue)
        plt.scatter(x=x, y=y, c='b')
    if pointsYellow is not None:
        x, y = zip(*pointsYellow)
        plt.scatter(x=x, y=y, c='y')

def saveImage(imgs, j=None, groundtruth=None, maxId=None, groundtruthPath="./groundtruth/", imagePath="./images/"):
    '''
        saveImage
        imgs - image or list of images which should be saved
        j - current id of image
        groundtruth - is the image a groundtruth
        maxId - maximal id which is used to pad the id with 0s
        groundtruthPath - path where to save the groundtruths
        imagePath - path where images are saved which are not a groundtruth
        saves one image or all images with the id j to the given path
    '''
    if not type(imgs) == list:
        imgs = [imgs]
    
    if not os.path.exists(groundtruthPath):
        os.mkdir(groundtruthPath)
    if not os.path.exists(imagePath):
        os.mkdir(imagePath)

    for i in range(len(imgs)):
        if not j:
            id = i
        else:
            id = j
        if maxId:
            i_str = str(id).rjust(int(np.log10(maxId)) + 1, '0')
        else:
            i_str = str(id).rjust(int(np.log10(len(imgs))) + 1, '0')

        path = imagePath
        if groundtruth:
            path = groundtruthPath
        #print('%svessel%s.png'%(path,i_str))
        imsave(                                                 # save image
            '%svessel%s.png'%(path,i_str), 
            np.transpose(imgs[i], (1,0,2))[:,:,:3]
        )

def rgba2rgb(img):
    '''
        rgba2rgb
        convert an rgba image to a rgb image
    '''
    r = np.multiply(img[:, :, 0], img[:, :, 3]) 
    g = np.multiply(img[:, :, 1], img[:, :, 3]) 
    b = np.multiply(img[:, :, 2], img[:, :, 3])
    return np.dstack((r,g,b))


def addMaskSupersampled(image, sizeX, sizeY, supersample):
    '''
        add black mask on top of the image
    '''
    if not addMaskSupersampled.final_mask_initialized:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isfile(dir_path + '/maskSupersample.npy'):
            addMaskSupersampled.final_mask = prepareMask(dir_path, sizeX*supersample, sizeY*supersample, '/maskSupersample.npy')
        else:
            addMaskSupersampled.final_mask = np.load(dir_path + '/maskSupersample.npy').astype(np.float32)
            if not addMaskSupersampled.final_mask.shape == (sizeX*supersample,sizeY*supersample,4):
                addMaskSupersampled.final_mask = prepareMask(dir_path, sizeX*supersample, sizeY*supersample, '/maskSupersample.npy')
        addMaskSupersampled.final_mask_initialized = True
        return mergeLayer([image, addMaskSupersampled.final_mask])
    return mergeLayer([image, addMaskSupersampled.final_mask])

addMaskSupersampled.final_mask_initialized = False
addMaskSupersampled.final_mask = None

def addMask(image, sizeX, sizeY):
    '''
        add black mask on top of the image
    '''
    assert(image.dtype == np.float32)
    if not addMask.final_mask_initialized:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isfile(dir_path + '/mask.npy'):
            addMask.final_mask = prepareMask(dir_path, sizeX, sizeY, '/mask.npy')
        else:
            addMask.final_mask = np.load(dir_path + '/mask.npy').astype(np.float32)
            if not addMask.final_mask.shape == (sizeX,sizeY,4):
                addMask.final_mask = prepareMask(dir_path, sizeX, sizeY, '/mask.npy')
        addMask.final_mask_initialized = True
        return mergeLayer([image, addMask.final_mask])
    return mergeLayer([image, addMask.final_mask])

addMask.final_mask_initialized = False
addMask.final_mask = None


def prepareMask(dir_path, sizeX, sizeY, filename):
    '''
        prepareMask
        load a mask from the DRIVE dataset and bring it into the wanted format 
    '''
    mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
    mask = transform.resize(mask, (sizeY, sizeX))
    mask = mask.T
    final_mask = np.zeros((sizeX, sizeY, 4), dtype=np.float32)
    black = np.where(mask < 0.5)
    transparent = np.where(mask >= 0.5)
    final_mask[black] = [0,0,0,1.0]
    final_mask[transparent] = [0,0,0,0]
    np.save(dir_path + filename, final_mask)
    return final_mask

def calculateMeanCoverage(path, sizeX, sizeY):
    '''
        calculateMeanCoverage
        calculates the mean coverage of all groundtruths in a given path
    '''
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images: 
        binary = imread(path+f).astype(np.float32)
        binary = transform.resize(binary, (sizeY, sizeX), preserve_range=True, mode='edge', anti_aliasing=True).astype(np.float32)
        binary[np.where(binary > 0)] = 1
        if len(binary.shape) < 3:
            binary = np.dstack((binary, binary, binary, binary))
        else:
            binary = np.dstack((binary, np.ones((binary.shape[0], binary.shape[1]), dtype=np.float32)))
        binary = binary * 255
        binary = np.transpose(binary, (1,0,2))
        binary[:,:,3] = 255
        means.append(meanCoverage(binary, sizeX, sizeY))
    print("MEAN COVERAGE: " + str(np.mean(np.asarray(means))))
    print("STDDEV COVERAGE: " + str(np.std(np.asarray(means))))
    return np.mean(np.asarray(means))

def meanCoverage(image, sizeX, sizeY):
    '''
        meanCoverage
        calculates the mean coverage of a given groundtruth
    '''
    m = addMask(image, sizeX, sizeY)[:,:,0]
    cov = np.mean(m) / 255
    return cov

if __name__ == '__main__':
    sizeX = 565
    sizeY = 584
    paths = [
        '/../DRIVE/test/1st_manual/'
    ]
    means = []
    for p in paths:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mypath = dir_path + p
        means = calculateMeanCoverage(mypath, sizeX, sizeY)

    # result: dilation of 0 mean = 0.00044903899554091833; std = 3.860214848429707e-05
