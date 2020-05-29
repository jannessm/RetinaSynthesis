from tqdm import tqdm
import sys
import os
import subprocess
import time
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser(description='Generate a bunch of synthetic retinal images.')
parser.add_argument('-N', type=int, dest='N', help='amount of images to generate')
parser.add_argument('--start', dest='start', default=0, help='start by image number')
parser.add_argument('--sizeX', type=int, dest='sizeX', help='x dimension of final images', default=565)
parser.add_argument('--sizeY', type=int, dest='sizeY', help='y dimension of final images', default=565)
parser.add_argument('--dest', dest='dest', default='.', help='output path for the generated images')
parser.add_argument('--processes', dest='processes', default=8, help='use N processes')

args = parser.parse_args()

def generateImage(id):
    path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/main.py")
    subprocess.check_call("python3 -W ignore \"" + path + \
        "\" --imageIndex " + str(id) + \
        " --lastImageIndex " + str(args.N) + \
        " --sizeX " + str(args.sizeX) + \
        " --sizeY " + str(args.sizeY) + \
        " --dest " + args.dest, shell=True)

if __name__ == '__main__':

    pool = Pool(processes=args.processes)
    with tqdm(total=args.N) as pbar:
        for _ in enumerate(pool.imap_unordered(generateImage, range(args.start, args.N))):
            pbar.update()
    pool.close()
