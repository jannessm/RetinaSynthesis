import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Tree import Tree
from PIL import Image, ImageDraw
from utils import mergeLayer, addIllumination, showImage, addMask
import cv2
from skimage import io, draw, data
from scipy.misc import imsave
import math
import os 
from OpticalDisc import generateOpticalDisc
from multiprocessing.dummy import Pool
import time
from multiprocessing.dummy import Pool
from OpticalDisc import generateOpticalDisc
from Fovea import generateBackgroundAndFovea
import tqdm

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def generateImages(i=0):
    bkg, fovea = generateBackgroundAndFovea()
    od_img, od = generateOpticalDisc()
    vt, groundTruth = generateVesselsTree(fovea, od)
    merged = mergeLayer([bkg, np.transpose(od_img,(1,0,2)), vt])
    image = addIllumination(merged)
    return addMask(image), addMask(groundTruth)


# generate an image containing the vessels tree
def generateVesselsTree(fovea, od):
    tree = Tree(od, fovea)
    tree.growTree()
    return tree.createTreeImage(), tree.createTreeMap()

if __name__ == '__main__':
    k = 200                              # amount of pictures to generate

    if k > 20:                           # limit threads to upper boundary 20
        nthreads = 20
    else:
        nthreads = k
    
    print("\nStart generating "+ str(k) +" images")
    start = time.time()                 # start timer

    #threads = Pool(nthreads)            # generate k images in parallel
    #res = threads.map(generateImages, range(k))
    #for _ in tqdm.tqdm(res, total=k):
     #   pass
    #threads.close()
    #threads.join()

    im = []
    gt = []
    for _ in tqdm.tqdm(range(k), total=k):
        i, g = generateImages()
        im.append(i)
        gt.append(g)

    print("\n" + str(k) + " pictures needed " + str(time.time() - start) + " sec!\n")
    
    print("\n saving groundtruths")
    showImage(gt, groundtruth=True, onlySave=True)
    print("\n saving images")
    showImage(im, groundtruth=False, onlySave=True)
