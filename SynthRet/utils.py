import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, transform, draw
from skimage.io import imread, imsave
import os
np.seterr(divide="ignore", invalid="ignore")
'''
    mergeLayer
    collect - list of rgba integer images 
    merges all rgba images in collect. the first will be the lowest image
'''
def mergeLayer(collect):

    # change each img to float
    for i in range(len(collect)):
        collect[i] = collect[i].astype(float) / 255
        assert(not collect[i].dtype == np.float32)

    dimg = collect[0]                               # init lowest layer
    for idx in range(1, len(collect)):
        img = collect[idx]
        if img is None:
            continue
            
        # a (img) over b (dimg) (porter-duff-algorithm)
        a_a = img[:, :, 3][:, :, np.newaxis] / 255
        a_b = dimg[:, :, 3][:, :, np.newaxis] / 255
        a_c = a_a + (1 - a_a) * a_b
        a_c[ a_c == 0 ] = 1                         # make sure no division by 0 is happening

        a_A = np.multiply(img, a_a)
        a_B = np.multiply(dimg, a_b)
        dimg = np.divide(a_A + np.multiply(a_B, 1 - a_a), a_c)
        #zero = np.where(a_c == 0)
        #dimg[zero[0], zero[1], :] = [0,0,0,0]
    return dimg

'''
    makeBinary
    img     - image to make binary
    threshold - the threshold
    make an image binary by a given threshold
'''
def makeBinary(img, threshold):
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
    binary = np.ones(grey.shape) * 255
    binary[np.where(grey > threshold)] = 255
    binary[np.where(grey < threshold)] = 0

    return binary

'''
    addIllumination
    image - image to add illumination to
'''
def addIllumination(image, groundtruth): #detail addjustment
    
    # set parameters (random)
    #brightness = np.random.uniform(0.1,3)
    #low, high = np.random.randint(low=0,high=30), np.random.randint(low=225,high=255)

    # enhance brightness
    #image1 = exposure.adjust_gamma(image.astype(float) / 255, brightness)
    #image1 = (image1 * 255).astype(int)
    
    # enhance contrast 
    #img = exposure.rescale_intensity(image1,out_range=(low,high))

    # mirror by probability of 0.5
    if np.random.rand() < 0.5:
        image = np.flipud(image)
        groundtruth = np.flipud(groundtruth)

    # add gaussian noise
    #gauss = np.random.normal(0, 0.1, (300, 300, 3)) * 255 * np.random.rand() * 0.1
    #alpha = np.zeros((300,300))
    #gauss = np.dstack((gauss, alpha))
    #img += gauss.astype(int)

    return np.clip(image, 0, 255), np.clip(groundtruth, 0, 255)

'''
    showImage

    img - image or list of images to plot
    pointsBlue - list of points to plot with blue color
    pointsYellow - list of points to plot with yellow color
    sec - seconds the plot is shown. if sec == -1 it is show until it is closed
    plots images into a plot and 
'''
def showImage(img, pointsBlue=None, pointsYellow=None, sec=-1):
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

'''
    subfunction of showImage()
    img - Image to show
    pointsBlue - points that are displayed blue
    pointsYellow - points that are displayed yellow
    plots the given image and points into a plt
'''
def _plotHelper(img, pointsBlue, pointsYellow):
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
def saveImage(imgs, j=None, groundtruth=None, maxId=None, groundtruthPath="./groundtruth/", imagePath="./images/"):
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
        print('%svessel%s.png'%(path,i_str))
        imsave(                                                 # save image
            '%svessel%s.png'%(path,i_str), 
            np.transpose(imgs[i].astype(int), (1,0,2))[:,:,:3]
        )

'''
    rgba2rgb
    convert an rgba image to a rgb image
'''
def rgba2rgb(img):
    r = np.multiply(img[:, :, 0], img[:, :, 3]) 
    g = np.multiply(img[:, :, 1], img[:, :, 3]) 
    b = np.multiply(img[:, :, 2], img[:, :, 3])
    return np.dstack((r,g,b))


'''
    add black mask on top of the image
'''
def addMask(image, sizeX, sizeY):
    if not addMask.final_mask_initialized:
        print("loading mask")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isfile(dir_path + '/mask.npy'):
            addMask.final_mask = prepareMask(dir_path, sizeX, sizeY).astype(np.float32)
        else:
            addMask.final_mask = np.load(dir_path + '/mask.npy').astype(np.float32)
            if not addMask.final_mask.shape == (sizeX,sizeY,4):
                addMask.final_mask = prepareMask(dir_path, sizeX, sizeY).astype(np.float32)
        addMask.final_mask_initialized = True
        return mergeLayer([image, addMask.final_mask])
    return mergeLayer([image, addMask.final_mask])

addMask.final_mask_initialized = False
addMask.final_mask = None

'''
    prepareMask
    load a mask from the DRIVE dataset and bring it into the wanted format 
'''
def prepareMask(dir_path, sizeX, sizeY):
    mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
    mask = transform.resize(mask, (sizeY, sizeX))
    mask = mask.T
    final_mask = np.zeros((sizeX, sizeY, 4))
    black = np.where(mask < 0.5)
    transparent = np.where(mask >= 0.5)
    final_mask[black] = [0,0,0,255]
    final_mask[transparent] = [0,0,0,0]
    np.save(dir_path + '/mask.npy', final_mask)
    return final_mask

'''
    calculateMeanCoverage
    calculates the mean coverage of all groundtruths in a given path
'''
def calculateMeanCoverage(path, sizeX, sizeY):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images: 
        binary = imread(path+f)
        binary = transform.resize(binary, (sizeY, sizeX))
        binary[np.where(binary > 0)] = 1
        if len(binary.shape) < 3:
            binary = np.dstack((binary, binary, binary, binary))
        else:
            binary = np.dstack((binary, np.ones((binary.shape[0], binary.shape[1]))))
        binary = (binary * 255).astype(int)
        binary = np.transpose(binary, (1,0,2))
        binary[:,:,3] = 255
        means.append(meanCoverage(binary, sizeX, sizeY))
    print("MEAN COVERAGE: " + str(np.mean(np.asarray(means))))
    print("STDDEV COVERAGE: " + str(np.std(np.asarray(means))))
    return np.mean(np.asarray(means))

'''
    meanCoverage
    calculates the mean coverage of a given groundtruth
'''
def meanCoverage(image, sizeX, sizeY):
    return np.mean(addMask(image, sizeX, sizeY)[:,:,0]) / 255

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
