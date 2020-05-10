from VesselTree import Tree
from utils import mergeLayer, addIllumination, showImage, addMaskSupersampled, saveImage
from OpticDisc import generateOpticDisc
from Background import generateBackgroundAndFovea
from IPython.display import clear_output
from skimage.transform import resize
from skimage import io

import cProfile

import numpy as np
import sys

'''
    generate synthetic images. if you want to save the generated files adjust save and path to your needs.
    path is relative to this file.
'''
def _generateImage(sizeX, sizeY, supersample):
    bkg, fovea = generateBackgroundAndFovea(sizeX,sizeY,supersample)
    assert(bkg.dtype == np.float32)
    
    od_img, od = generateOpticDisc(fovea,sizeX,sizeY,supersample)
    assert(od_img.dtype == np.float32)
    
#    print(np.mean(bkg[:,:,3]))
#    io.imshow(bkg)
#    io.show()
#    print(np.mean(od_img[:,:,3]))
#    io.imshow(od_img)
#    io.show()
    
    vt, groundTruth = generateVesselsTree(sizeX, sizeY, fovea, od, supersample)
    assert(vt.dtype == np.float32)
    assert(groundTruth.dtype == np.float32)
    
#    print(np.mean(vt[:,:,3]))
#    io.imshow(vt)
#    io.show()
    
    merged = mergeLayer([bkg, od_img, vt])
    assert(merged.dtype == np.float32)

#    print(np.mean(merged[:,:,3]))
#    io.imshow(merged)
#    io.show()
    
    
    image = addMaskSupersampled(merged, sizeX, sizeY, supersample)
    groundTruth = addMaskSupersampled(groundTruth, sizeX, sizeY, supersample)
    assert(image.dtype == np.float32)
    assert(groundTruth.dtype == np.float32)

#    io.imshow(image)
#    io.show()

    image, groundTruth = addIllumination(image, groundTruth)
    assert(image.dtype == np.float32)
    assert(groundTruth.dtype == np.float32)

#    print(np.mean(image[:,:,3]))
#    io.imshow(image / 255)
#    io.show()

    image = resize(image, (sizeX, sizeY, 4), anti_aliasing=True)
    groundTruth = resize(groundTruth, (sizeX, sizeY, 4), anti_aliasing=True)
    
#    io.imshow(image / 255)
#    io.show()

    return image.astype('uint8'), groundTruth.astype('uint8')


# generate an image containing the vessels tree
def generateVesselsTree(sizeX, sizeY, fovea, od, supersample):
    
    tree = Tree(sizeX, sizeY, od, fovea)
    tree.growTree()
    return tree.createSupersampledImages(supersample)

def generateImages(i=0, total=1, sizeX=300, sizeY=300, showImages=True, save=False, groundTruthPath="./groundtruths/", imagesPath="./images/"):    
    im, gt = _generateImage(sizeX, sizeY, 4)
    if save:
        saveImage(im, i, False, total, groundTruthPath, imagesPath)
        saveImage(gt, i, True,  total, groundTruthPath, imagesPath)
 
    if showImages:
        showImage(im)
        showImage(gt)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Ignoring parameters!")
        i = 0
        total = 1
        sizeX = 565
        sizeY = 565
    else:
        i = int(sys.argv[1])
        total = int(sys.argv[2])
        sizeX = int(sys.argv[3])
        sizeY = int(sys.argv[4])

    generateImages(i, total, sizeX, sizeY, showImages=False, save=True)