import numpy as np

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

    output = utils.runNetwork(input, net)

    utils.saveMaskImg(output, args.output, 50)

