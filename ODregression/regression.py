from skimage import io,transform,color,img_as_ubyte,draw
import numpy as np
from scipy import optimize


#read the real image
img = io.imread("25_training.tif")
nimg = img_as_ubyte(transform.resize(img,(300,300)))

#segmentation OD area in a binary image
gimg = img_as_ubyte(color.rgb2gray(nimg))
biimg = gimg >134
corp = np.zeros((300,300),np.bool)
rr,cc = draw.circle(150,240,50)
draw.set_color(corp,[rr,cc],1)
biimg = biimg & corp

#select OD area points, input is the coordinates of points
#output is the real color chanel value in each point
input = np.array(np.where(biimg>0))
outputr = nimg[input[0],input[1],0]
outputg = nimg[input[0],input[1],1] 
outputb = nimg[input[0],input[1],2]

#mathematical models of three color channels 
def odr(x,zr,xr,yr,a,sr):
    exponentr = -((x[1]-xr)/sr)**2 - ((x[0]-yr)/sr)**2
    red = zr - 1/(a+np.exp(exponentr)) 
    return red

def odg(x,zr,xr,yr,a,sr,kg,xg,yg,sg):
    exponentr = -((x[1]-xr)/sr)**2 - ((x[0]-yr)/sr)**2
    r =  zr - 1/(a+np.exp(exponentr))
    exponentg = -((x[1]-xg)/sg)**2 - ((x[0]-yg)/sg)**2
    green = r+kg*np.exp(exponentg)
    return green

def odb(x,zr,xr,yr,a,sr,kb,xb,yb,sb):
    exponentr = -((x[1]-xr)/sr)**2 - ((x[0]-yr)/sr)**2
    r =  zr - 1/(a+np.exp(exponentr))
    exponentb = -((x[1]-xb)/sb)**2 - ((x[0]-yb)/sb)**2
    blue = r+kb*np.exp(exponentb)
    return blue

#do the regression to get best parameters
poptr, pcovr = optimize.curve_fit(odr,input,outputr,
                                 bounds=([150,150,100,0,0],[255,300,200,0.5,50]))
poptg, pcovg = optimize.curve_fit(odg,input,outputg,
                                 bounds=([0,150,100,0,5,0,150,100,0],[255,300,200,0.5,50,100,300,200,50]))
poptb, pcovb = optimize.curve_fit(odb,input,outputb,
                                 bounds=([0,150,100,0,5,0,150,100,0],[255,300,200,0.5,50,100,300,200,50]))

print(poptr, poptg, poptb)