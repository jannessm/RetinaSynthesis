import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from utils import mergeLayer, addIllumination, showImage, addMask, makeBinary
from skimage import io, draw, data
import scipy.misc 
#from scipy.misc import imsave
import math
import os 
from OpticalDisc import generateOpticalDisc
from multiprocessing.dummy import Pool
import time
from Fovea import generateBackgroundAndFovea
import tqdm

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def generateImages(i=0):
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

if __name__ == '__main__':
    k = 1                              # amount of pictures to generate

    if k > 20:                           # limit threads to upper boundary 20
        nthreads = 20
    else:
        nthreads = k
    
    print("\nStart generating "+ str(k) +" images")
    start = time.time()                 # start timer

    for _ in tqdm.tqdm(range(k), total=k):
        i, g = generateImages()
        showImage([i])
        showImage([g])
#        showImage([i], groundtruth=False, onlySave=True)
 #       showImage([g], groundtruth=True, onlySave=True)

    print("\n" + str(k) + " pict1ures needed " + str(time.time() - start) + " sec!\n")

    # bild = io.imread("bild.png")
    # showImage(bild)
    # showImage(makeBinary(bild, 10))