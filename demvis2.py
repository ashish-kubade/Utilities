#!/home/ashj/anaconda3/bin/python
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import math
dem_name = sys.argv[1]
if dem_name[-4:]==".npy":
	img = np.load(dem_name)
else:
	img = np.loadtxt(dem_name, delimiter=',')
#img_mean = np.mean(img)

minval = np.min(img)
img = img - minval
maxval = np.max(img)

img = img /maxval


fig,ax = plt.subplots()
# if len(img.shape) == 3:
# 	mig = img[0]
cax = plt.imshow(img)
plt.colorbar(cax)
ax.set(title = os.path.basename('Normalized '+dem_name))

plt.show()
