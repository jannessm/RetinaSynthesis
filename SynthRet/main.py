import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage, odr, odb, odg
import cv2 
from skimage import io, transform
from scipy.misc import imread
import math
import os 

def main():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    return addMask(image), groundTruth

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    #all black background
    #img = Image.new('RGBA',(300,300),0) #TODO remove image object because of merge USE skimage

    #eye=ImageDraw.Draw(img)
    #small=ImageDraw.Draw(img)
    #fovea=ImageDraw.Draw(img)
    #macula=ImageDraw.Draw(img)

    #position of fovea macula and small part
    #PosSmall=(150+125*math.cos(45)-30,150-125*math.cos(45)-30,150+125*math.cos(45)+45,150-125*math.cos(45)+45)
    #randomly change the position of fovea
    #change=np.random.randint(-5,5)
    #PosFovea=(135-change,135-change,165-change,165-change)
    #PosMacula=(115,115,185,185) #TODO add random

    #draw
    #eye.ellipse((25,25,275,275),fill=(255,255,255))
    #small.ellipse(PosSmall,fill=(255,255,255))
    #macula.ellipse(PosMacula,fill=(277,207,87))
    #fovea.ellipse(PosFovea,fill=(237,145,33))
    
    return None, [150, 150] #TODO return mean value

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

# generate an image with the optical disc
def generateOpticalDisc():
    odimg = np.zeros((300, 300, 4),np.uint8) 
    odimg[::,::,3] = 255 
    cv2.ellipse(odimg,(240,150),(22,26),0,0,360,(255,255,255,255),-1,8,0) 
    for i in range(217,263): 
        for j in range(123,177): 
            if (odimg[j,i,::] == [255,255,255,255]).all(): 
                odimg[j,i,0] = odr(i,j) 
                odimg[j,i,1] = odg(i,j) 
                odimg[j,i,2] = odb(i,j) 
    return odimg, [240, 150] #TODO select random point according to fovea pos.

def addMask(image):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + '/mask.npy'):
        mask = imread(dir_path + '/../DRIVE/test/mask/01_test_mask.gif')
        mask = transform.resize(mask, (300, 300))
        mask = mask.T
        final_mask = np.zeros((300,300,4))
        black = np.where(mask < 0.5)
        transparent = np.where(mask >= 0.5)
        final_mask[black] = [0,0,0,1]
        final_mask[transparent] = [255,255,255,0]
        showImage(final_mask, None)
        np.save(dir_path + '/mask.npy', mask)
    else:
        mask = np.load(dir_path + '/mask.npy')
    
    return image


if __name__ == '__main__':
    #img, gt = main()
    #showImage(img, None)
    #showImage(gt, None)
    addMask(None)
