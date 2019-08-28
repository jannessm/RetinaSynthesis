import torch
import numpy as np
import scipy.signal as signal
from PIL import Image

import matplotlib.pyplot as plt
import argparse

import utils


parser = argparse.ArgumentParser(description='Evaluate a single retina image.')
parser.add_argument('--drive', dest='drive', help='Path to DRIVE dataset')
parser.add_argument('--imageIndex', type=int, dest='imageIndex', help='1..20 (index of image in DRIVE)')
parser.add_argument('--threshold', type=float, dest='threshold', help='Threshold for prediction', default=0.0)
parser.add_argument('--outputImg', dest='outputImg', help='Filename of generated evaluation image', default='eval.png')
parser.add_argument('--network', dest='network', default='combined', help='Which network to use (combined, synthetic, drive_large, drive_small)')

parser.print_help()

args = parser.parse_args()


net = utils.getNetwork(args.network)


if args.drive != None and args.imageIndex != None and net != None:

    input, GT, mask = utils.loadDriveTuple(args.drive, args.imageIndex)


    padded = utils.padImageMultipleOf(input, 32)

    padded = torch.from_numpy(padded[np.newaxis,:,:,:]).float()
    output = net.forward(padded)
    output = output.detach().numpy()[0,:,:,:]

    output = output[0, 0:input.shape[1], 0:input.shape[2]]





    prediction = output > args.threshold

    TP, TN, FP, FN = utils.evaluate_TP_TN_FP_FN(prediction, GT, mask)

    P, N = utils.count_P_N(GT, mask)

    print("At threshold {}:".format(args.threshold))
    print("True positive rate: {}%".format(np.sum(TP) * 100.0 / P))
    print("True negative rate: {}%".format(np.sum(TN) * 100.0 / N))

    utils.saveEvalImg(TP, TN, FP, FN, args.outputImg)


    thresholds = np.arange(-20, 20, 0.1)
    TP_curve = np.zeros(thresholds.shape, int)
    TN_curve = np.zeros(thresholds.shape, int)

    utils.accumulateROC(output, thresholds, TP_curve, TN_curve, GT, mask)

    TP_curve = np.array(TP_curve).astype(float) / P
    TN_curve = np.array(TN_curve).astype(float) / N
        
    print("TP/TN curve:")
    print(TP_curve, TN_curve)

    plt.plot(TP_curve, TN_curve)
    plt.xlabel('True Positive')
    plt.xlabel('True Negative')
    plt.show()


