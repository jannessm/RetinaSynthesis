from VesselTree import Tree
from utils import mergeLayer, addIllumination, showImage, addMask, saveImage
from OpticalDisc import generateOpticalDisc
from Background import generateBackgroundAndFovea
from IPython.display import clear_output

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

def generateImages(k=1, start=0, showImages=True, save=False, groundTruthPath="./groundtruths/", imagesPath="./images/"):    
    print("\nStart generating "+ str(k) +" images")
    start_time = time.time()                 # start timer

    imgs = []
    gt = []

    for j in tqdm.tqdm(range(start, k), total=k):
        print(j)
        if not j == 0:
            print("\nlast image took: " + str(time.time() - start_time) + " sec")
        i, g = _generateImage()
        imgs.append(i)
        gt.append(g)
        if save:
            saveImage(i, j, False, k, groundTruthPath, imagesPath)
            saveImage(g, j, True,  k, groundTruthPath, imagesPath)

    print("\n" + str(k) + " pictures needed " + str(time.time() - start_time) + " sec!\n")
 
    if showImages:
        showImage(imgs)
        showImage(gt)

if __name__ == '__main__':
    images = 200
    generateImages(images, showImages=False, save=True)