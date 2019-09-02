import torch
import numpy as np
import scipy.signal as signal
from PIL import Image

import matplotlib.pyplot as plt
import argparse


import utils




parser = argparse.ArgumentParser(description='Evaluate across entire DRIVE test dataset.')
parser.add_argument('--drive', dest='drive', help='Path to DRIVE dataset')
parser.add_argument('--network', dest='network', default='combined', help='Which network to use (combined, synthetic, drive_large, drive_small)')
parser.add_argument('--curveCSV', dest='curveCSV', help='Filename of output .csv file for ROC curve')

parser.print_help()

args = parser.parse_args()


net = utils.getNetwork(args.network)

if args.drive != None and net != None:
    drive_images = range(1, 21)

    P_total = 0
    N_total = 0

    thresholds = np.arange(-20, 20, 0.1)
    TP_curve = np.zeros(thresholds.shape, int)
    TN_curve = np.zeros(thresholds.shape, int)

    for img_idx in drive_images:
        print("Running image {}".format(img_idx))
        input, GT, mask = utils.loadDriveTuple(args.drive, img_idx)

        padded = utils.padImageMultipleOf(input, 32)

        padded = torch.from_numpy(padded[np.newaxis,:,:,:]).float()
        output = net.forward(padded)
        output = output.detach().numpy()[0,:,:,:]

        output = output[0, 0:input.shape[1], 0:input.shape[2]]

        P, N = utils.count_P_N(GT, mask)
        P_total = P_total + P
        N_total = N_total + N

        utils.accumulateROC(output, thresholds, TP_curve, TN_curve, GT, mask)

    TP_curve = np.array(TP_curve).astype(float) / P_total
    TN_curve = np.array(TN_curve).astype(float) / N_total
        
    plt.plot(TP_curve, TN_curve)
    plt.xlabel('True Positive')
    plt.xlabel('True Negative')
    plt.show()

    if args.curveCSV != None:
        f = open(args.curveCSV, 'w')
        f.write("{};Evaluation on DRIVE test set;;\n".format(args.network))
        f.write("Threshold;True positive rate;True negative rate;\n")
        for t,TP,TN in zip(thresholds,TP_curve,TN_curve):
            f.write("{};{};{};\n".format(t, TP, TN))
        f.close()

