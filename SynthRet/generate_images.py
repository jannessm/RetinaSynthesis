from tqdm import tqdm
import sys
import os
import subprocess
import time

if len(sys.argv) > 4:
    start = int(sys.argv[1])
    images = int(sys.argv[2])
    sizeX = int(sys.argv[3])
    sizeY = int(sys.argv[4])
else:
    start = 0
    images = int(sys.argv[1])
    sizeX = 300
    sizeY = 300

for i in tqdm(range(start, images), total=images - start):
    t = time.time()
    path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/main.py")
    subprocess.check_call("python -W ignore \"" + path + "\" " + str(i) + " " + str(images) + " " + str(sizeX) + " " + str(sizeY), shell=True)
    print (str(i) + " needed " + str(time.time() - t) + "sec")