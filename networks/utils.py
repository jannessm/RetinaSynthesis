import torch

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
    return (image / 128.0 - 1.0) * 3.0

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




def runNetwork(input, net):
    padded = padImageMultipleOf(input, 32)

    padded = torch.from_numpy(padded[np.newaxis,:,:,:]).float()
    output = net.forward(padded)
    output = output.detach().numpy()[0,:,:,:]

    return output[0, 0:input.shape[1], 0:input.shape[2]]

def runNetwork_tiled(input, net):

    crop_size = 128
    overlap = 32
    output_crop_size = crop_size - 2*overlap

    num_blocks1 = (input.shape[1] + output_crop_size-1)//output_crop_size
    num_blocks2 = (input.shape[2] + output_crop_size-1)//output_crop_size

    output = np.zeros((num_blocks1*output_crop_size, num_blocks2*output_crop_size))

    padded = np.zeros((3, num_blocks1*output_crop_size + overlap*2, num_blocks2*output_crop_size + overlap*2)) - 3.0

    padded[:, overlap:input.shape[1]+overlap, overlap:input.shape[2]+overlap] = input

    print(padded.shape)

    for block1 in range(0, num_blocks1):
        for block2 in range(0, num_blocks2):
            output_offset_1 = block1 * output_crop_size
            output_offset_2 = block2 * output_crop_size
            inputOffset1 = output_offset_1 - overlap + overlap
            inputOffset2 = output_offset_2 - overlap + overlap

            input_crop = padded[:, inputOffset1:inputOffset1+crop_size, inputOffset2:inputOffset2+crop_size]
            print(input_crop.shape)


            input_crop = torch.from_numpy(input_crop[np.newaxis,:,:,:]).float()
            output_crop = net.forward(input_crop)
            output_crop = output_crop.detach().numpy()[0,:,:,:]
            

            output[output_offset_1:output_offset_1+output_crop_size, output_offset_2:output_offset_2+output_crop_size] = output_crop[0,overlap:-overlap,overlap:-overlap]


    return output[0:input.shape[1], 0:input.shape[2]]


