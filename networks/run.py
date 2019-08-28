import torch
import numpy as np
import scipy.signal as signal
from PIL import Image

import argparse

import utils


parser = argparse.ArgumentParser(description='Run a single retina image.')
parser.add_argument('--input', dest='input', help='Filename of input image')
parser.add_argument('--output', dest='output', help='Filename of output image')
parser.add_argument('--network', dest='network', default='combined', help='Which network to use (combined, synthetic, drive_large, drive_small)')

parser.print_help()

args = parser.parse_args()


net = utils.getNetwork(args.network)


if args.input != None and args.output != None and net != None:
    input = utils.loadImg(args.input)

    padded = utils.padImageMultipleOf(input, 32)

    padded = torch.from_numpy(padded[np.newaxis,:,:,:]).float()
    output = net.forward(padded)
    output = output.detach().numpy()[0,:,:,:]

    output = output[0, 0:input.shape[1], 0:input.shape[2]]

    utils.saveMaskImg(output, args.output, 100)

