from tqdm import tqdm
import sys
import os
import subprocess
import time

if len(sys.argv) > 2:
    start = int(sys.argv[1])
    images = int(sys.argv[2])
else:
    start = 0
    images = int(sys.argv[1])

for i in tqdm(range(start, images), total=images - start):
    t = time.time()
    path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/main.py")
    subprocess.check_call("python \"" + path + "\" " + str(i) + " " + str(images), shell=True)
    print (str(i) + " needed " + str(time.time() - t) + "sec")