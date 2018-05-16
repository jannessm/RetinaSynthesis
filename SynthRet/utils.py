# imports
from PIL import Image, ImageEnhance

# functions

# 
def eval():
    return 0.0

#
def merge(*args):
    finalImage = [300, 300, 4]
    # TODO merge all arguments. first is the lowest layer
    for i in args:
        # TODO
        continue
    return finalImage

#
def addIllumination(image):
    
    # set parameters
    brightness = 2
    color = 3  
    contrast = 3  
    sharpness = 3.0

    # enhance brightness
    image1 = ImageEnhance.Brightness(image).enhance(brightness)  

    # enhance color
    image2 = ImageEnhance.Color(image1).enhance(color)   
    
    # enhance contrase 
    image3 = ImageEnhance.Contrast(image2).enhance(contrast)   
    
    # enhance sharpness 
    img = ImageEnhance.Sharpness(image3).enhance(sharpness)  

    return img