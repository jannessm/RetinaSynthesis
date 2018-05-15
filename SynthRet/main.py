import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import merge, addIllumination

def main():
    bkg = generateBackgroundAndFovea()
    vt = generateVesselsTree()
    od = generateOpticalDisc()
    merged = merge(bkg, vt, od)
    return addIllumination(merged)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    img = [300, 300, 4]
    return img

# generate an image containing the vessels tree
def generateVesselsTree():
    img = [300, 300, 4]
    return img

# generate an image with the optical disc
def generateOpticalDisc():
    img = [300, 300, 4]
    # do something here
    return img

if __name__ == '__main__':
    main()