import torch
import numpy as np
import scipy.signal as signal
from PIL import Image

import matplotlib.pyplot as plt
import argparse


import utils




parser = argparse.ArgumentParser(description='Evaluate across entire DRIVE test dataset.')
parser.add_argument('--data', dest='data', help='Path to dataset')
parser.add_argument('--data_type', dest='data_type', help='Dataset type that determines the specific load paths of the data. One of (drive, chase, hrf, iostar, stare). Default is drive.', default='drive')
parser.add_argument('--hrf_type', dest='hrf_type', help='Type of hrf data. This argument is only used if data_type is hrf. One of (dr, g, h). Default is h.', default='h')
parser.add_argument('--network', dest='network', default='combined', help='Which network to use (combined, synthetic, drive_large, drive_small)')
parser.add_argument('--curveCSV', dest='curveCSV', help='Filename of output .csv file for ROC curve')

args = parser.parse_args()


net = utils.getNetwork(args.network)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if args.data != None and net != None and args.data_type != None:
    images = utils.getIndices(args.data_type)

    P_total = 0
    N_total = 0

    thresholds = np.arange(-20, 20, 0.1)
    TP_curve = np.zeros(thresholds.shape, int)
    TN_curve = np.zeros(thresholds.shape, int)

    for img_idx, kargs in images:
        print("Running image {}".format(img_idx))
        input, GT, mask = utils.loadImageTuple(args.data, args.data_type, img_idx, hrf_type=args.hrf_type, **kargs)

        padded = utils.padImageMultipleOf(input, 32)


        padded = torch.from_numpy(padded[np.newaxis,:,:,:]).float()
        output = net.to(device).forward(padded.to(device))
        output = output.cpu().detach().numpy()[0,:,:,:]

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

