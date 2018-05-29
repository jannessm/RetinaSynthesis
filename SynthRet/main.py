import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.io as io

from utils import merge, addIllumination,odr,odb,odg

def main():
    bkg = generateBackgroundAndFovea()
    io.imsave("./1.png",bkg)
    vt = generateVesselsTree()
    io.imsave("./2.png",vt)
    od = generateOpticalDisc()
    io.imsave("./3.png",od)
    collect = io.ImageCollection("./*.png")
    merged = merge(collect)
    io.imshow(merged)
    return addIllumination(merged)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    img = [300, 300, 4]
    return img

# generate an image containing the vessels tree
def generateVesselsTree():
    img = [300, 300, 4]
    #do somethings
    return img

# generate an image with the optical disc
#vers. 1: set alpha to 0; choose parameters by try different values  
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


#if __name__ == '__main__':
#    main()

#od = generateOpticalDisc()
#io.imshow(od)
#io.imsave("./2.png",od)
