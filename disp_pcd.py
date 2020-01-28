#!/home/ashj/anaconda3/bin/python
import numpy as np
from open3d import * 
import argparse
import sys

FILE_NAME = sys.argv[1]

disp_name = FILE_NAME.split('/')[-1]

def display_cloud(points):
    pc = PointCloud()
    pc.points = Vector3dVector(points)
    pc.colors = Vector3dVector(np.array([[1,0,0]] * len(pc.points)))
    draw_geometries([pc], window_name = disp_name)

file = np.load(FILE_NAME)

# data_points = file.points
# points_aug = np.vstack((data_points[0], data_points[1], data_points[2], data_points[5])).T   #dont forget the T for transpose
# points = np.vstack((data_points[0], data_points[1], data_points[2])).T

points = np.vstack((file[0,:], file[1,:], file[2,:])).T
# print(points.shape)
display_cloud(points)


