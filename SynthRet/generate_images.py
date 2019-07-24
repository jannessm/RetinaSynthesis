from tqdm import tqdm
import sys
import os
import subprocess
import time
from multiprocessing import Pool

start = 0
images = 1
sizeX = 500
sizeY = 500

def generateImage(id):
    path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/main.py")
    subprocess.check_call("python3 -W ignore \"" + path + "\" " + str(id) + " " + str(images) + " " + str(sizeX) + " " + str(sizeY), shell=True)

if __name__ == '__main__':

    if len(sys.argv) > 4:
        start = int(sys.argv[1])
        images = int(sys.argv[2])
        sizeX = int(sys.argv[3])
        sizeY = int(sys.argv[4])
    else:
        images = int(sys.argv[1])

    pool = Pool(processes=8)
    with tqdm(total=images) as pbar:
        for i, _ in tqdm(enumerate(pool.imap_unordered(generateImage, range(start, images)))):
            pbar.update()
    pool.close()
