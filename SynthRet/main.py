import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree

from utils import mergeLayer, addIllumination, showImage

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


#if __name__ == '__main__':
#    main()

#od = generateOpticalDisc()
#io.imshow(od)
#io.imsave("./2.png",od)

#code for merge test
collect = io.ImageCollection("./*.png")
d=merge(collect)
io.imshow(d)
