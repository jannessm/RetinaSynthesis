import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_, ax = plt.subplots()

fovea = (150,150)
od = (250,150)

img = patches.Rectangle((0,0), 300, 300, fill=False, edgecolor="red")
ul = patches.Rectangle((-150, 200), 150, 150, facecolor="blue")
ll = patches.Rectangle((-150, -50), 150, 150, facecolor="blue")
ur = patches.Rectangle((300, 170), 150, 180, facecolor="orange")
lr = patches.Rectangle((300, -50), 150, 180, facecolor="orange")

i = ax.add_patch(img)
ax.add_patch(ul)
ax.add_patch(ll)
ax.add_patch(ur)
ax.add_patch(lr)

f = plt.scatter(fovea[0], fovea[1], c="blue")
o = plt.scatter(od[0], od[1], c="orange")
plt.legend((f, o, i), ("Fovea", "Optical Disc", "image boundaries"), loc="upper center")
plt.xlabel("x position in px")
plt.ylabel("y position in px")

plt.show()