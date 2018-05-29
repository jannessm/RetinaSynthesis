import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.io as io

from utils import merge, addIllumination,odr,odb,odg

def main():
    bkg = generateBackgroundAndFovea()
    vt = generateVesselsTree()
    od = generateOpticalDisc()
    merged = merge(bkg, vt, od)
    return addIllumination(merged)

# generate an image with the background and fovea
def generateBackgroundAndFovea():
    #all black background
    img = Image.new('RGBA',(300,300),0)

    eye=ImageDraw.Draw(img)
    small=ImageDraw.Draw(img)
    fovea=ImageDraw.Draw(img)
    macula=ImageDraw.Draw(img)

    #position of fovea macula and small part
    PosSmall=(150+125*math.cos(45)-30,150-125*math.cos(45)-30,150+125*math.cos(45)+45,150-125*math.cos(45)+45)
    #randomly change the position of fovea
    change=random.randint(-5,5)
    PosFovea=(135-change,135-change,165-change,165-change)
    PosMacula=(115,115,185,185)

    #draw
    eye.ellipse((25,25,275,275),fill=(255,255,255))
    small.ellipse(PosSmall,fill=(255,255,255))
    macula.ellipse(PosMacula,fill=(277,207,87))
    fovea.ellipse(PosFovea,fill=(237,145,33))
    
    return img,PosFovea

# generate an image containing the vessels tree
def generateVesselsTree():
    img = [300, 300, 4]
    #do somethings
    return img

# generate an image with the optical disc
#vers. 1: set alpha to 0; choose parameters by try different values  
def generateOpticalDisc():
    odimg = np.zeros((300, 300, 4),np.uint8)
    odimg[::,::,3] = 255
    cv2.ellipse(odimg,(240,150),(22,26),0,0,360,(255,255,255,255),-1,8,0)
    for i in range(217,263):
        for j in range(123,177):
            if (odimg[j,i,::] == [255,255,255,255]).all():
                odimg[j,i,0] = odr(i,j)
                odimg[j,i,1] = odg(i,j)
                odimg[j,i,2] = odb(i,j)
    return odimg


#if __name__ == '__main__':
#    main()

#od = generateOpticalDisc()
#io.imshow(od)
#io.imsave("./2.png",od)

#code for merge test
collect = io.ImageCollection("./*.png")
d=merge(collect)
io.imshow(d)
