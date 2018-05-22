import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Point import Point

from utils import merge, addIllumination

def main():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt = generateVesselsTree(fovea, od)
    merged = merge(bkg, vt, od_img)
    return addIllumination(merged)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    img = [300, 300, 4]
    return img, Point(0,0)

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    img = [300, 300, 4]
    #do some things
    return img

# generate an image with the optical disc
def generateOpticalDisc():
    img = [300, 300, 4]
    # do something here
    return img, Point(0,0)

if __name__ == '__main__':
    main()