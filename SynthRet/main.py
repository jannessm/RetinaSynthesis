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
def _generateImage(sizeX, sizeY):
    bkg, fovea = generateBackgroundAndFovea(sizeX,sizeY)
    od_img, od = generateOpticDisc(fovea,sizeX,sizeY)
    vt, groundTruth = generateVesselsTree(sizeX, sizeY, fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image, groundTruth = addIllumination(merged, groundTruth)
    return addMask(image, sizeX, sizeY), addMask(groundTruth, sizeX, sizeY)


# generate an image containing the vessels tree
def generateVesselsTree(sizeX, sizeY, fovea, od):
    tree = Tree(sizeX, sizeY, od, fovea)
    tree.growTree()
    return tree.createAliasedImages()

def generateImages(i=0, total=1, sizeX=300, sizeY=300, showImages=True, save=False, groundTruthPath="./groundtruths/", imagesPath="./images/"):    
    im, gt = _generateImage(sizeX, sizeY)
    if save:
        saveImage(im, i, False, total, groundTruthPath, imagesPath)
        saveImage(gt, i, True,  total, groundTruthPath, imagesPath)
 
    if showImages:
        showImage(im)
        showImage(gt)

if __name__ == '__main__':
    i = int(sys.argv[1])
    total = int(sys.argv[2])
    sizeX = int(sys.argv[3])
    sizeY = int(sys.argv[4])
    generateImages(i, total, sizeX, sizeY, showImages=False, save=True)