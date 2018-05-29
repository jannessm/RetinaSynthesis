import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree

from utils import mergeLayer, addIllumination, showImage

def main():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, vt, od_img])
    return addIllumination(merged), groundTruth

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    img = [300, 300, 4]
    return img, [0,0]

# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

# generate an image with the optical disc
def generateOpticalDisc():
    img = [300, 300, 4]
    # do something here
    return img, [0,0]

if __name__ == '__main__':
    img, groundTruth = main()
    showImage(img, None)
    showImage(groundTruth, None)