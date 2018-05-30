import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage
import cv2
from skimage import io, transform, draw, data
from scipy.misc import imread
import math
import os 
from OpticalDisc import generateOpticalDisc

def main():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    return addMask(image), addMask(groundTruth)

# generate an image with the background and fovea
def generateBackgroundAndFovea(): #TODO where is the gradient and the red tissue?
    img=np.zeros((300, 300, 4),np.uint8)            
    img[:,:,]=[255,127,36,255]
    #macula
    change=np.random.randint(-20,20)
    for i in range(100):
        rr,cc=draw.circle(150+change,150+change,26-i/4.0)
        draw.set_color(img,[rr,cc],[190-i,190-i,190-i,255])
    
    #fovea
    PosFovea=(150+change,150+change)
    rr,cc=draw.circle(150+change,150+change,15)
    draw.set_color(img,[rr,cc],[139,126,102,255])
    return img, PosFovea

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

'''
    add black mask on top of the image
'''
def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy') or True:
        mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
        mask = transform.resize(mask, (300, 300))
        mask = mask.T
        final_mask = np.zeros((300,300,4))
        black = np.where(mask < 0.5)
        transparent = np.where(mask >= 0.5)
        final_mask[black] = [0,0,0,255]
        final_mask[transparent] = [255,255,255,0]
        np.save(dir_path + '/mask.npy', mask)
    else:
        final_mask = np.load(dir_path + '/mask.npy')
    return mergeLayer([image, final_mask])


if __name__ == '__main__':
    img, gt = main()
    showImage(img)
    showImage(gt)
