import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import merge, addIllumination

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
def generateOpticalDisc():
    img = [300, 300, 4]
    # do something here
    return img

if __name__ == '__main__':
    main()