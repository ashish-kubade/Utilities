#!/home/ashj/anaconda3/bin/python
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
def calculate_psnr_dem(arr1, arr2):
    # img1 and img2 have range [0, 255]
    # img1 and img2 have range [0, 255]
    img1 = arr1.astype(np.float64)
    img2 = arr2.astype(np.float64)

    pick_factor = np.max(img1) - np.min(img1)
    print(pick_factor)
    mse = np.mean((img1 - img2)**2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')

    return 20 * math.log10((pick_factor) / rmse), rmse

def calculate_psnr_img(img1, img2):
    # img1 and img2 have range [0, 255]


    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / rmse), rmse


img1 = sys.argv[1]
img2 = sys.argv[2]

if(img1[-4:] == '.npy'):
	arr1 = np.load(img1)
	arr2 = np.load(img2)
elif(img1[-4:] == '.dem'):
    arr1 = np.loadtxt(img1)
    arr2 = np.loadtxt(img2)
else:
	arr1 = plt.imread(img1)
	arr2 = plt.imread(img2)

psnr, rmse = calculate_psnr_img(arr1, arr2)


print('images: ', img1.split('/')[-1])
print('images: ', img2.split('/')[-1])
print('RMSE', rmse)
print('PSNR', psnr)
input()

