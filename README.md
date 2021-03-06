# Synthesizing Fundus Photographies for Training Segmentation Networks

This repository maintains all relevant code for reproducing the results of this [paper](). It presents a novel method of generating synthetic fundus images. These images can be used for data augmentation to train segmentation networks like the U-net. Although the generated photographies look unrealistic, state-of-the art performance is achived by combining realistic and synthetic images for training. The proposed network is evaluated on DRIVE and STARE data.

* [Usage](#usage)
* [Generating images](#generating-images)
* [Mean Coverage](#mean-coverage)
* [Optic Disc Regression](#optic-disc-regression)
* [Evaluation](#evaluation)
* [Results](#results)
* [Citation](#citation)


## Usage
For running the code in this repo install all dependencies by:

```python
pip install -r requirements.txt
```

## Generating images

Images can be generated by the script `SynthRet/generate_images.py`. The following arguments can be used:

|Flag | Default | Description |
| -- | -- | -- |
-N | 2000 | number of images to generate
--start | 0 | start numbering with index i
--sizeX | 565 | x dimension of final images
--sizeY | 565 | y dimension of final images', default=565)
--dest | ./data | output path for the generated images
--processes | 8 | use N processes
--help | | display help
    
    
## Mean Coverage
To get the mean coverage of binary groundtruths from the drive dataset run

    python SynthRet/utils.py
    
It will output the mean coverage of vessels at the groundtruths which is used as
a termination criterion when growing the vessel tree.

## Optic Disc Regression

Run `regression.py` to get the best-fit parameters of the mathematical model for generating optic disc. 

```python
python ODregression/regression.py
    
#The first output list is the parameters [zr,xr,yr,a,sr] for Red channel. 
#The second output list is the parameters [z1,x1,y1,a1,s1,kg,xg,yg,sg] for Green channel. 
#The third output list is the parameters [z2,x2,y2,a2,s2,kb,xb,yb,sb] for Blue channel.
```

ODr.py and ODb.py can plot the color distribution of the mathematical models. colordistribution.py can plot the color distribution of the real image.

## Evaluation

All trained networks are given in the directory networks. The models are implemented in pytorch. For an evaluation, no GPU is required. The following scripts are provided:

* `eval_single.py` can be used to evaluate a specific sample for a given threshold
* `eval.py` is used to evaluate the model for a range of thresholds for generating an ROC curve.
* `run.py` predicts the segmentation for a given image and saves it under a given path

## Results
The networks are evaluated on DRIVE and STARE data.

### DRIVE
|Experiment | Se | Sp | Acc | AUC|
| -- | -- | -- | -- | -- |
DRIVE 1 | 0.7448 | 0.9796 | 0.9493 | 0.9609
DRIVE 2 | 0.56 | 0.977 | 0.9237 | 0.7865
synthetic | 0.667 | 0.9858 | 0.9449 | 0.9541
combined | 0.8334 | 0.9715 | 0.9554 | 0.9784

### STARE
|Experiment | Se | Sp | Acc | AUC|
| -- | -- | -- | -- | -- |
combined | 0.818 | 0.9705 | 0.9589 | 0.9421


## Citation

Please cite our paper if you use this code in your own work:

```
Magnusson, J.; Afifi, A.; Zhang, S.; Ley, A. and Hellwich, O. (2021). Synthesizing Fundus Photographies for Training Segmentation Networks.  In Proceedings of the 2nd International Conference on Deep Learning Theory and Applications, ISBN 978-989-758-526-5, ISSN 2184-9277, pages 67-78.   
```

```bibtex
@inproceedings{Magnusson_Synthesizing_2021,
    title={Synthesizing Fundus Photographies for Training Segmentation Networks},
    author={Magnusson, J. and Afifi, A. and Zhang, S. and Ley, A. and Hellwich, O.},
    booktitle={Proceedings of the 2nd International Conference on Deep Learning Theory and Applications},
    year={2021},
    isbn={978-989-758-526-5},
    pages={67--78}
}
```
