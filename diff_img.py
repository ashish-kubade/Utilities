#!/home/ashj/anaconda3/bin/python
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
def get_diff_img(arr1, arr2):
    # img1 and img2 have range [0, 255]
    # img1 and img2 have range [0, 255]
    img1 = arr1.astype(np.float64)
    img2 = arr2.astype(np.float64)
    diff = img1 - img2
    return diff

img1 = sys.argv[1]
img2 = sys.argv[2]

name_1 = img1.split('/')[-1]
name_2 = img2.split('/')[-1]

if(img1[-4:] == '.npy'):
	arr1 = np.load(img1)
	arr2 = np.load(img2)
elif(img1[-4:] == '.dem'):
    arr1 = np.loadtxt(img1)
    arr2 = np.loadtxt(img2)
else:
	arr1 = plt.imread(img1)
	arr2 = plt.imread(img2)

diff_img = get_diff_img(arr1, arr2)

print('Min and max values in the diff img', np.min(diff_img), np.max(diff_img))
# fig,ax = plt.subplots()
# # cax = plt.imshow(arr1, cmap='gray')
# cax = plt.imshow(arr1)
# cbar = fig.colorbar(cax)

# ax.set(title=name_1)
# #plt.savefig('LR.png', bbox_inches='tight')


# fig,ax = plt.subplots()
# # cax = plt.imshow(arr2, cmap='gray')
# cax = plt.imshow(arr2)
# cbar = fig.colorbar(cax)
# ax.set(title=name_2)
#plt.savefig('Output_SR.png', bbox_inches='tight')


fig,ax = plt.subplots()

# cax = plt.imshow(diff_img, cmap='gray')
cax = plt.imshow(diff_img)

cbar = fig.colorbar(cax)
ax.set(title='{} - {}'.format(name_1, name_2))

plt.show()