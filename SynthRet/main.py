from VesselTree import Tree
from utils import mergeLayer, addIllumination, showImage, addMaskSupersampled, saveImage
from OpticDisc import generateOpticDisc
from Background import generateBackgroundAndFovea
from skimage.transform import resize
from skimage import io

import cProfile

import numpy as np
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='Generate a synthetic retinal image.')
parser.add_argument('--imageIndex', type=int, dest='imageIndex', help='Index of the image to save')
parser.add_argument('--showImage', type=bool, dest='showImage', default=False, help='Show the generated image at the end.')
parser.add_argument('--lastImageIndex', type=int, dest='lastImageIndex', help='last image index is used for 0 padding the image name')
parser.add_argument('--sizeX', type=int, dest='sizeX', help='x dimension of the final image', default=565)
parser.add_argument('--sizeY', type=int, dest='sizeY', help='y dimension of the final image', default=565)
parser.add_argument('--dest', dest='dest', default='.', help='output path for the generated images')

args = parser.parse_args()

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
    generateImages(
        args.imageIndex,
        args.lastImageIndex,
        args.sizeX,
        args.sizeY,
        args.showImage,
        True,
        osp.join(args.dest, 'groundtruths', ''),
        osp.join(args.dest, 'images', '')
    )