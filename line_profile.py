#!/home/ashj/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import math

def calculate_psnr_dem(arr1, arr2):
    img1 = arr1.astype(np.float64)
    img2 = arr2.astype(np.float64)

    pick_factor = np.max(img1) - np.min(img1)
    print(pick_factor)
    mse = np.mean((img1 - img2)**2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')

    return 20 * math.log10((pick_factor) / rmse), rmse


inp_1 = sys.argv[1]
inp_2 = sys.argv[2]
inp_3 = sys.argv[3]

n1 = inp_1[-8:]
n2 = inp_2[-8:]
n3 = inp_3[-8:]
d1 = np.load(inp_1)
d2 = np.load(inp_2)
d3 = np.load(inp_3)

row_number = int(input())

# print(offset_GT[:10, :10])

# fig,ax = plt.subplots()
# cax = plt.imshow(d1, cmap='gray')
# cbar = fig.colorbar(cax)
# ax.set(title='GT HR')
# #plt.savefig('GT_HR.png', bbox_inches='tight')

# fig,ax = plt.subplots()
# cax = plt.imshow(d2, cmap='gray')
# cbar = fig.colorbar(cax)
# ax.set(title='LR')
# #plt.savefig('LR.png', bbox_inches='tight')


# fig,ax = plt.subplots()
# cax = plt.imshow(d3, cmap='gray')
# cbar = fig.colorbar(cax)
# ax.set(title='Output SR')
#plt.savefig('Output_SR.png', bbox_inches='tight')


# fig,ax = plt.subplots()
# cax = plt.imshow(offset_GT-offset_output, cmap='gray')
# cbar = fig.colorbar(cax)
# ax.set(title='Difference from GT(HR) and Output(SR)')
#plt.savefig('Difference_Image.png', bbox_inches='tight')


# print(diff_layer[0].shape)
# splits = np.array([0, 	50, 100, 150, 200])
# for i in splits:
# 	fig,ax = plt.subplots()
# 	row = diff_layer[0]
# 	row = np.reshape(row , [1,200])
# 	cax = plt.imshow(row, cmap='gray')
# 	cbar = fig.colorbar(cax)

c1 = 'r'
c2 = 'g'
c3 = 'k'

img_width = np.shape(d1)[0]
t = np.arange(0.0, img_width)
fig, ax = plt.subplots()
ax.plot(t, d1[row_number], label=n1, color = c1)
ax.plot(t, d2[row_number], label=n2, color = c2)
ax.plot(t, d3[row_number], label=n3, color=c3)
ax.set(xlabel='Pixel', ylabel='Elevation',
       title='DEM profiles: {}_{}_{}'.format(n1,n2,n3))

ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
#plt.savefig('HR_LR_SR.png', bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(t, d1[row_number], label=n1, color = c1)
ax.plot(t, d2[row_number], label=n2, color = c2)
ax.set(xlabel='Pixel', ylabel='Elevation',
       title='DEM profile: {} Vs {}'.format(n1,n2))
ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
#plt.savefig('GT_SR.png', bbox_inches='tight')


fig, ax = plt.subplots()
ax.plot(t, d2[row_number], label=n2, color = c2)
ax.plot(t, d3[row_number], label=n3, color = c3)
ax.set(xlabel='Pixel', ylabel='Elevation',
       title='DEM profiles: Input {} Vs {}'.format(n2, n3))
ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='small')

fig, ax = plt.subplots()
ax.plot(t, d1[row_number], label=n1, color = c1)
ax.plot(t, d3[row_number], label=n3, color = c3)
ax.set(xlabel='Pixel', ylabel='Elevation',
       title='DEM profiles: Input {} Vs {}'.format(n1, n3))
ax.grid()
legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
#plt.savefig('LR_SR.png', bbox_inches='tight')

plt.show()

