import torch
import numpy as np
import scipy.signal as signal
from PIL import Image

import matplotlib.pyplot as plt
import argparse

import utils


parser = argparse.ArgumentParser(description='Evaluate a single retina image.')
parser.add_argument('--data', dest='data', help='Path to dataset')
parser.add_argument('--data_type', dest='data_type', help='Dataset type that determines the specific load paths of the data. One of (drive, chase, hrf, iostar, stare). Default is drive.', default='drive')
parser.add_argument('--hrf_type', dest='hrf_type', help='Type of hrf data. This argument is only used if data_type is hrf. One of (dr, g, h). Default is h.', default='h')
parser.add_argument('--imageIndex', nargs="+", dest='imageIndex', help='list of indecies 1..20 (index of image in dataset)')
parser.add_argument('--threshold', type=float, dest='threshold', help='Threshold for prediction', default=0.0)
parser.add_argument('--outputImg', dest='outputImg', help='Filename of generated evaluation image', default='eval.png')
parser.add_argument('--network', dest='network', default='combined', help='Which network to use (combined, synthetic, drive_large, drive_small)')
args = parser.parse_args()


net = utils.getNetwork(args.network)


if args.data != None and args.imageIndex != None and net != None:
    for i in [int(j) for j in args.imageIndex]:
        input, GT, mask = utils.loadImageTuple(args.data, args.data_type, i, hrf_type=args.hrf_type)

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

        utils.saveEvalImg(TP, TN, FP, FN, '%d_' % i + args.outputImg)