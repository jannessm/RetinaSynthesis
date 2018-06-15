import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree

from utils import mergeLayer, addIllumination, showImage, odr, odb, odg
import cv2 
import skimage.io as io 

def main():
    bkg, fovea = generateBackgroundAndFovea()
    io.imsave("./1.png",bkg)
    od_img, od = generateOpticalDisc()
    io.imsave("./2.png",vt)
    vt, groundTruth = generateVesselsTree(fovea, od)
    io.imsave("./3.png",od)
    collect = io.ImageCollection("./*.png")
    merged = mergeLayer(collect)
    io.imshow(merged)
    return addIllumination(merged), groundTruth


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
    return odimg


if __name__ == '__main__':
    main()
