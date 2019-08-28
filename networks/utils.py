import net_combined.net as net_combined
import net_synthetic.net as net_synthetic
import net_drive_large.net as net_drive_large
import net_drive_small.net as net_drive_small

from PIL import Image
import numpy as np


def getNetwork(net_name):
    if net_name == 'combined':
        return net_combined.Network()
    elif net_name == 'synthetic':
        return net_synthetic.Network()
    elif net_name == 'drive_large':
        return net_drive_large.Network()
    elif net_name == 'drive_small':
        return net_drive_small.Network()
    else:
        return None

def loadBinary(filename):
    im = Image.open(filename) 
    image = np.array(im)
    return image[:,:] > 127


def loadImg(filename):
    im = Image.open(filename) 
    image = np.array(im).astype(float)
    image = np.stack((image[:,:,0], image[:,:,1], image[:,:,2]), axis=0)
    return image / 255.0 * 6.0 - 3.0

def saveMaskImg(img, filename, scale):
    image = np.stack((img[:,:], img[:,:], img[:,:]), axis=-1)
    im = Image.fromarray(np.minimum(np.maximum(255/2.0 + image * scale, 0), 255).astype('uint8'), 'RGB')
    im.save(filename)


def padImageMultipleOf(image, multiple_of = 32):
    w = image.shape[2]   
    h = image.shape[1]

    new_w = ((w + multiple_of-1) // multiple_of) * multiple_of
    new_h = ((h + multiple_of-1) // multiple_of) * multiple_of

    return np.pad(image, ((0,0), (0,new_h-h), (0,new_w-w)), 'constant', constant_values = -3.0) # pad with black


def saveEvalImg(TP, TN, FP, FN, filename):
    eval_image = np.zeros((TP.shape[0], TP.shape[1], 3))

    eval_image[TP, 0] = 255
    eval_image[TP, 1] = 255
    eval_image[TP, 2] = 255

    eval_image[FP, 0] = 0
    eval_image[FP, 1] = 0
    eval_image[FP, 2] = 255

    eval_image[FN, 0] = 255
    eval_image[FN, 1] = 0
    eval_image[FN, 2] = 0

    im = Image.fromarray(eval_image.astype('uint8'), 'RGB')
    im.save(filename)

def loadDriveTuple(path, index):
    input = loadImg("{}/test/images/{:02d}_test.tif".format(path, index))
    GT = loadBinary("{}/test/1st_manual/{:02d}_manual1.gif".format(path, index))
    mask = loadBinary("{}/test/mask/{:02d}_test_mask.gif".format(path, index))
    
    return input, GT, mask


def count_P_N(GT, mask):
    P = np.sum(np.logical_and(GT, mask))
    N = np.sum(np.logical_and(np.logical_not(GT), mask))
    return P, N

def evaluate_TP_TN_FP_FN(prediction, GT, mask):
    TP = np.logical_and(np.logical_and(prediction, GT), mask)
    TN = np.logical_and(np.logical_and(np.logical_not(prediction), np.logical_not(GT)), mask)

    FP = np.logical_and(np.logical_and(prediction, np.logical_not(GT)), mask)
    FN = np.logical_and(np.logical_and(np.logical_not(prediction), GT), mask)

    return TP, TN, FP, FN

def accumulateROC(output, thresholds, TP_curve, TN_curve, GT, mask):
    for i,t in enumerate(thresholds):
        TP = np.logical_and(np.logical_and((output > t), GT), mask)
        TN = np.logical_and(np.logical_and((output <= t), np.logical_not(GT)), mask)

        TP_curve[i] = TP_curve[i] + np.sum(TP)
        TN_curve[i] = TN_curve[i] + np.sum(TN)





