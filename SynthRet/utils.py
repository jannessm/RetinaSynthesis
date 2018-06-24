# imports
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, transform, draw
import scipy.misc 
from scipy.ndimage.morphology import binary_dilation
import os

imagePath = './images/'
groundtruthPath = './groundtruth/'

def measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP / (TP+FN), TN / (TN+FP), (TP+TN)/(TP+TN+FP+FN))

#merge 3-chanel RGB images
def merge3c(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial black image
    finalimage = np.zeros((ncol, nrow, 3),np.uint8)
    #merge layers
    for n in range(len(collect)):
        img = collect[n]
        for i in range(ncol):
            for j in range(nrow):
                if (img[i,j,:].max() > 0):
                    finalimage[i,j,:] = img[i,j,:]
    return finalimage

#merge 4-chanel RGBA images
def mergeLayer(collect):
    #***the size of images***
    ncol=300
    nrow=300
    #initial transparent image
    dimg = np.zeros((ncol, nrow, 4),np.uint8)
    #merge layers
    for img in collect:
        if img is None:
            continue
        ids = np.where(img[:,:,3] > 0)
        dimg[ids] = img[ids]
    return dimg

def makeBinary(img, threshold):
    if img.shape[2] == 4:
        r = np.multiply(img[:, :, 0], img[:, :, 3])
        g = np.multiply(img[:, :, 1], img[:, :, 3])
        b = np.multiply(img[:, :, 2], img[:, :, 3])
    else:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
    rgb = np.dstack((r, g, b))
    grey = np.multiply(rgb, [0.21, 0.72, 0.07])
    grey = np.sum(grey, axis=2)
    binary = np.ones(grey.shape) * 255
    binary[np.where(grey > threshold)] = 255
    binary[np.where(grey < threshold)] = 0
    return binary.astype(int)

#
def addIllumination(image): # rewrite with skimage
    
    # set parameters (random)
    brightness = np.random.uniform(0.1,3)
    low, high = np.random.randint(low=0,high=30), np.random.randint(low=225,high=255)

    # enhance brightness
    image1 = exposure.adjust_gamma(image, brightness) 
    #exposure.adjust_log(image)   
    
    # enhance contrast 
    img = exposure.rescale_intensity(image1,out_range=(low,high))

    return img

def showImage(img, pointsBlue=None, sec=-1, groundtruth=None, onlySave=False, pointsYellow=None, k=None):
    if type(img) == list:
        pointsBlue = pointsBlue if type(pointsBlue) == list else [None] * len(img)
        pointsYellow = pointsYellow if type(pointsYellow) == list else [None] * len(img)
        rows = np.floor(np.sqrt(len(img)))
        cols = np.ceil(np.sqrt(len(img))) 
        if not rows > 1:
            rows = 1
            cols = len(img)
        for i in range(len(img)):
            if not onlySave:
                plt.subplot(int(rows), int(cols), i+1)
            if k:
                i_str = str(i).rjust(int(np.log10(k)) + 1, '0')
            else:
                i_str = str(i).rjust(int(np.log10(len(img))) + 1, '0')
            _plotHelper(img[i], pointsBlue[i], pointsYellow[i], i_str, groundtruth, onlySave)
    else:
        _plotHelper(img, pointsBlue, pointsYellow, '', groundtruth, onlySave)
    if not sec == -1 and not onlySave:
        plt.show(block=False)
        plt.pause(sec)
        plt.close()
    elif not onlySave:
        plt.show()

def _plotHelper(img, pointsBlue, pointsYellow, i='', groundtruth=None, onlySave=False):
    if img.ndim == 3:
        if not onlySave:
            plt.imshow(np.transpose(img, (1,0,2)))   #show transposed so x is horizontal and y is vertical
        if i:
            rgb = rgba2rgb(np.transpose(img, (1,0,2)))
            path = imagePath
            if groundtruth:
                path = groundtruthPath
            print '%svessel%s.jpg'%(path,i)
            scipy.misc.imsave('%svessel%s.jpg'%(path,i), rgb)      # save images as jpg
    else:
        if not onlySave:
            plt.imshow(img)
        if i:
            path = imagePath
            if groundtruth:
                path = groundtruthPath
            scipy.misc.imsave('%svessel%s.jpg'%(path,i), img.T)
    if pointsBlue is not None and not onlySave:
        x, y = zip(*pointsBlue)
        plt.scatter(x=x, y=y, c='b')
    if pointsYellow is not None and not onlySave:
        x, y = zip(*pointsYellow)
        plt.scatter(x=x, y=y, c='y')

def rgba2rgb(img):
    a = img[:,:,3] / 255.
    return np.dstack(((img[:,:,0] + (255 * (1-a)), img[:,:,1] + (255 * (1-a)), img[:,:,2] + (255 * (1-a)))))

'''
    add black mask on top of the image
'''
def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy'):
        mask = scipy.misc.imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
        mask = transform.resize(mask, (300, 300))
        mask = mask.T
        final_mask = np.zeros((300,300,4))
        black = np.where(mask < 0.5)
        transparent = np.where(mask >= 0.5)
        final_mask[black] = [0,0,0,255]
        final_mask[transparent] = [255,255,255,0]
        np.save(dir_path + '/mask.npy', final_mask)
    else:
        final_mask = np.load(dir_path + '/mask.npy')
        if not final_mask.shape == (300,300,4):
            mask = scipy.misc.imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
            mask = transform.resize(mask, (300, 300))
            mask = mask.T
            final_mask = np.zeros((300,300,4))
            black = np.where(mask < 0.5)
            transparent = np.where(mask >= 0.5)
            final_mask[black] = [0,0,0,255]
            final_mask[transparent] = [255,255,255,0]
            np.save(dir_path + '/mask.npy', final_mask)
    return mergeLayer([image, final_mask])

def calculateMeanCoverage(path, k=10):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images:
        binary = scipy.misc.imread(path+f)
        binary = transform.resize(binary, (300, 300))
        means.append(meanCoverage(binary, [150,150], k))
    return np.mean(np.asarray(means))

def fig2ndarray(fig):
    fig.canvas.draw ()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h, 4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    buf = np.transpose(buf, (1,0,2))
    buf = transform.resize(buf, (300, 300))
    return buf

def coverage(binary, fovea, k=10):
    if binary.ndim == 3:
        binary = makeBinary(binary, 200)
    
    # dilation
    if k > 0:
        binary = binary_dilation(binary, iterations=k)

    rgba = np.zeros((300, 300, 4)) # make rgba image from binary
    rgba[np.where(binary)] = [0,0,0, 255] # draw binary on it

    # add fovea
    rr, cc = draw.circle(fovea[0], fovea[1], 15)
    draw.set_color(rgba, [rr,cc], [0,0,0,255])

    # add mask
    binary = addMask(rgba)    # add Mask
    binary = np.abs(binary - [255, 255, 255, 0])
    return makeBinary(binary, 200)

def meanCoverage(img, fovea, k=10):
    return np.mean(coverage(img, fovea, k)) / 255

if __name__ == '__main__':
    paths = [
        '/../DRIVE/test/1st_manual/',
        '/../DRIVE/test/2nd_manual/'
    ]
    means = []
    k = 0
    for p in paths:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mypath = dir_path + p
        means.append(calculateMeanCoverage(mypath, k))
    print("MEAN COVERAGE WITH DILATION OF " + str(k) + ": " + str(np.mean(np.asarray(means))))
    print("STDDEV COVERAGE WITH DILATION OF " + str(k) + ": " + str(np.std(np.asarray(means))))

    # result: dilation of 0 mean = 0.38799999; std = 0.0605
