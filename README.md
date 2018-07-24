# Synthesizing Retinal Fundus Photographies

This repository is structured in 4 important parts. 

- /
- /Retina Unet/              evaluation by a U-Net from [Retina Unet](https://github.com/orobix/retina-unet).
- /SynthRet/                 scripts for generating synthesized images
- /SynthRetSet/              the generated images for training porpuses
- /ODregression/             find the wanted parameters for generating od
- /DRIVE/			         [DRIVE dataset](https://www.isi.uu.nl/Research/Databases/DRIVE/)
- /High-Resolution Fundus/   [HRF dataset](https://www5.cs.fau.de/research/data/fundus-images/)
- /tex/                      latex documents
- /who_did_what.txt          who did what document
- /plot.py                   plot fig. 9

## SynthRet

Images can be generated by running the following script:

    python SynthRet/generate_images.py <total number of images>

or if an offset is used:

    python SynthRet/generate_images.py <start index> <end index>
    
    
## meanCoverage
To get the mean coverage of binary groundtruths from the drive dataset run

    python SynthRet/utils.py
    
It will output the mean coverage of vessels at the groundtruths which is used as
a termination criterion when growing the vessel tree.

## ODregression

Run regression.py to get the best-fit parameters of the mathematical model for generating optic disc. 

    python ODregression/regression.py
    
    #The first output list is the parameters [zr,xr,yr,a,sr] for Red channel. 
    #The second output list is the parameters [z1,x1,y1,a1,s1,kg,xg,yg,sg] for Green channel. 
    #The third output list is the parameters [z2,x2,y2,a2,s2,kb,xb,yb,sb] for Blue channel.

ODr.py and ODb.py can plot the color distribution of the mathematical models. colordistribution.py can plot the color distribution of the real image.

## Evaluation

To run the evaluation run the following:

    python RetinaUnet/prepare_datasets_SynthRet_Patches.py
    python RetinaUnet/run_training_SynthRet_Patches.py
    python RetinaUnet/run_testing_SynthRet_Patches.py

These scripts are adjusted scipts from [Retina Unet](https://github.com/orobix/retina-unet) to run all 3 wanted experiments.