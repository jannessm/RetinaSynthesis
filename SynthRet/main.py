from VesselTree import Tree
from utils import mergeLayer, addIllumination, showImage, addMask, saveImage
from OpticalDisc import generateOpticalDisc
from Background import generateBackgroundAndFovea

import numpy as np
import time
import tqdm

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def _generateImage():
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc(fovea)
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, od_img, vt])
    image = addIllumination(merged)
    return addMask(image), addMask(groundTruth)


# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

def generateImages(k=1, showImages=True, save=False, groundTruthPath="", imagesPath=""):    
    print("\nStart generating "+ str(k) +" images")
    start = time.time()                 # start timer

    imgs = []
    gt = []

    for j in tqdm.tqdm(range(k), total=k):
        print("generate Image: ",j)
        i, g = _generateImage()
        imgs.append(i)
        gt.append(g)
        if save:
            saveImage(i, j, groundtruth=False, maxId=k)
            saveImage(g, j, groundtruth=True,  maxId=k)

    print("\n" + str(k) + " pictures needed " + str(time.time() - start) + " sec!\n")

    if showImages:
        showImage(imgs)
        showImage(gt)

if __name__ == '__main__':
    generateImages()