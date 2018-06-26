import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, transform, draw
from scipy.misc import imread, imsave
import os

#merge 4-chanel RGBA images
def mergeLayer(collect, lastIsVessel=False):

    # change each img to float
    for i in range(len(collect)):
        collect[i] = collect[i].astype(float) / 255

    dimg = collect[0]
    #merge layers
    for img in collect:
        if img is None or np.array_equal(img, dimg):
            continue

        # a (img) over b (dimg)
        a_a = img[:, :, 3][:, :, np.newaxis]
        a_b = dimg[:, :, 3][:, :, np.newaxis]
        a_c = a_a + (1 - a_a) * a_b

        a_A = np.multiply(img, a_a)
        a_B = np.multiply(dimg, a_b)
        dimg = np.divide(a_A + np.multiply(a_B, 1 - a_a), a_c)
    return (dimg * 255).astype(int)

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
    image1 = exposure.adjust_gamma(image.astype(float) / 255, brightness)
    image1 = (image1 * 255).astype(int)
    
    # enhance contrast 
    img = exposure.rescale_intensity(image1,out_range=(low,high))
    return img

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

def _plotHelper(img, pointsBlue, pointsYellow):
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

def saveImage(imgs, groundtruth=None, maxId=None, groundtruthPath="./groundtruth/", imagePath="./images/"):
    if not type(imgs) == list:
        imgs = [imgs]

    for i in range(len(imgs)):
        if maxId:
            i_str = str(i).rjust(int(np.log10(maxId)) + 1, '0')
        else:
            i_str = str(i).rjust(int(np.log10(len(imgs))) + 1, '0')
        rgb = rgba2rgb(np.transpose(imgs[i], (1,0,2)))
        path = imagePath
        if groundtruth:
            path = groundtruthPath
        print '%svessel%s.jpg'%(path,i_str)
        imsave('%svessel%s.jpg'%(path,i_str), rgb)      # save images as jpg

def rgba2rgb(img):
    r = np.multiply(img[:, :, 0], img[:, :, 3]) 
    g = np.multiply(img[:, :, 1], img[:, :, 3]) 
    b = np.multiply(img[:, :, 2], img[:, :, 3])
    return np.dstack((r,g,b))


'''
    add black mask on top of the image
'''
def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy'):
        final_mask = prepareMask(dir_path)
    else:
        final_mask = np.load(dir_path + '/mask.npy')
        if not final_mask.shape == (300,300,4):
            final_mask = prepareMask(dir_path)
    return mergeLayer([image, final_mask])

def prepareMask(dir_path):
    mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
    mask = transform.resize(mask, (300, 300))
    mask = mask.T
    final_mask = np.zeros((300,300,4))
    black = np.where(mask < 0.5)
    transparent = np.where(mask >= 0.5)
    final_mask[black] = [0,0,0,255]
    final_mask[transparent] = [0,0,0,0]
    np.save(dir_path + '/mask.npy', final_mask)
    return final_mask

def calculateMeanCoverage(path):
    images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    means = []
    for f in images: 
        binary = imread(path+f)
        binary = transform.resize(binary, (300, 300))
        binary[np.where(binary > 0)] = 1
        if len(binary.shape) < 3:
            binary = np.dstack((binary, binary, binary, binary))
        else:
            binary = np.dstack((binary, np.ones((binary.shape[0], binary.shape[1]))))
        binary = (binary * 255).astype(int)
        means.append(meanCoverage(binary, [150,150]))
    return np.mean(np.asarray(means))

def coverage(binary, fovea):
    #invert
    binary = np.abs(binary - [255, 255, 255, 0])

    # add fovea
    rr, cc = draw.circle(fovea[0], fovea[1], 15)
    draw.set_color(binary, [rr,cc], [0,0,0,255])

    # add mask
    binary = addMask(binary)    # add Mask

    #invert
    binary = np.abs(binary - [255, 255, 255, 0])
    return binary

def meanCoverage(img, fovea):
    return np.mean(coverage(img, fovea)) / 255

if __name__ == '__main__':
    paths = [
        '/../DRIVE/test/1st_manual/',
        '/../DRIVE/test/2nd_manual/'
    ]
    means = []
    for p in paths:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mypath = dir_path + p
        means.append(calculateMeanCoverage(mypath))
    print("MEAN COVERAGE: " + str(np.mean(np.asarray(means))))
    print("STDDEV COVERAGE: " + str(np.std(np.asarray(means))))

    # result: dilation of 0 mean = 0.93107; std = 0.06892986
