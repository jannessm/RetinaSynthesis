from tqdm import tqdm
import sys
import os
import subprocess
import time

if len(sys.argv) > 4:
    start = int(sys.argv[1])
    images = int(sys.argv[2])
    image_x = int(sys.argv[3])
    image_y = int(sys.argv[4])
else:
    start = 0
    images = int(sys.argv[1])
    image_x = 300
    image_y = 300

for i in tqdm(range(start, images), total=images - start):
    t = time.time()
    path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/main.py")
    subprocess.check_call("python \"" + path + "\" " + str(i) + " " + str(images) + " " + str(image_x) + " " + str(image_y), shell=True)
    print (str(i) + " needed " + str(time.time() - t) + "sec")