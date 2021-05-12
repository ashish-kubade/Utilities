import torch
from chamferdist import ChamferDistance
import numpy as np
import sys

def convert_to_pc(dem_arr):
	pc_length = dem_arr.shape[0] * dem_arr.shape[1]
	pc = np.zeros([1, pc_length, 3])
	i = 0
	for idx, row in enumerate(dem_arr):
		for idy, col in enumerate(row):
			pc_point = [idx, idy, col]
			pc[0][i] = pc_point
			i += 1
	return pc

GT_PATH = sys.argv[1]
TEST_IMG_PATH = sys.argv[2]

img_GT = np.load(GT_PATH)[0:200, 0:200]

img_test = np.load(TEST_IMG_PATH)[0:200, 0:200]

pc_GT = torch.Tensor(convert_to_pc(img_GT)).cuda()
pc_test = torch.Tensor(convert_to_pc(img_test)).cuda()

chamferDist = ChamferDistance()
dist_forward = chamferDist(pc_GT, pc_test)
print(dist_forward.detach().cpu().item())