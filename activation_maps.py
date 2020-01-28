#!/home/ashj/anaconda3/bin/python

import numpy as np
import sys
import os

inp = sys.argv[1]
out_name_prefix = inp.split('/')[-1][:-4]
inp = np.load(inp, allow_pickle=True)
#inp = np.load(inp)
no_of_grps = len(inp)
print(no_of_grps)
def get_act_map(arr):
	arr = np.sum(arr, 0)
#	arr = np.sum(arr, 0)
	return arr

for i in range(no_of_grps):
	acvt_map = get_act_map(inp[i])
	np.save('{}_act_map_{}'.format(out_name_prefix,i), acvt_map)

