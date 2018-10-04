from VesselTree import Tree
from utils import mergeLayer, addIllumination, showImage, addMask, saveImage
from OpticDisc import generateOpticDisc
from Background import generateBackgroundAndFovea
from IPython.display import clear_output

import numpy as np
import sys

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def _generateImage(imageX, imageY):
    bkg, fovea = generateBackgroundAndFovea(imageX, imageY)
    od_img, od = generateOpticDisc(imageX, imageY, fovea)
    vt, groundTruth = generateVesselsTree(imageX, imageY, fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    return addMask(image), addMask(groundTruth)


# generate an image containing the vessels tree
def generateVesselsTree(imageX, imageY, fovea, od):
    tree = Tree(imageX, imageY, od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

def generateImages(i=0, total=1, imageX=300, imageY=300, showImages=True, save=False, groundTruthPath="./groundtruths/", imagesPath="./images/"):    
    im, gt = _generateImage(imageX, imageY)
    if save:
        saveImage(im, i, False, total, groundTruthPath, imagesPath)
        saveImage(gt, i, True,  total, groundTruthPath, imagesPath)
 
    if showImages:
        showImage(im)
        showImage(gt)

if __name__ == '__main__':
    i = int(sys.argv[1])
    total = int(sys.argv[2])
    image_x = int(sys.argv[3])
    image_y = int(sys.argv[4])
    generateImages(i, total, image_x, image_y, showImages=False, save=True)