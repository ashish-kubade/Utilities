#!/home/ashj/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys

fig = plt.figure()
ax = fig.add_subplot(111)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)

inp = sys.argv[1]
a = np.load(inp)


# plt.imshow(a, cmap='gray')
plt.imshow(a)
out_path = inp[:-4] + 'temp.png'
plt.savefig(out_path, dpi=1000, bbox_inches='tight', pad_inches=0)
