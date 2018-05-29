import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage, odr, odb, odg
import cv2 
from skimage import io, transform,draw,data
from scipy.misc import imread
import math
import os 

def main():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    return addMask(image), addMask(groundTruth)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    #all black background
    img=np.zeros((300, 300, 4),np.uint8)
    #outline
    rr,cc=draw.circle(150,150,125)
    draw.set_color(img,[rr,cc],[255,127,36,255])
    #small part
    rr,cc=draw.circle(150-125*math.cos(45),180+125*math.cos(45),22)
    draw.set_color(img,[rr,cc],[255,127,36,255])
    #macula
    rr,cc=draw.circle(150,150,25)
    draw.set_color(img,[rr,cc],[205,186,150,255])
    #fovea
    change=random.randint(-8,8)
    PosFovea=(150+change,150+change)
    rr,cc=draw.circle(150+change,150+change,15)
    draw.set_color(img,[rr,cc],[139,126,102,255])
    return img,PosFovea

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

# generate an image with the optical disc
def generateOpticalDisc():
    odimg = np.zeros((300, 300, 4),np.uint8)
    cv2.ellipse(odimg,(240,150),(22,26),0,0,360,(255,255,255,255),-1,8,0) 
    for i in range(217,263): 
        for j in range(123,177): 
            if np.array_equal(odimg[j,i], [255,255,255,255]): 
                odimg[j,i,0] = odr(i,j) 
                odimg[j,i,1] = odg(i,j) 
                odimg[j,i,2] = odb(i,j) 
    return np.transpose(odimg, (1,0,2)), [240, 150] #TODO select random point according to fovea pos.

'''
    add black mask on top of the image
'''
def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy'):
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
