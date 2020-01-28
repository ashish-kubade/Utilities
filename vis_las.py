from laspy.file import File
import numpy as np
from open3d import * 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file_name', type = str, help = 'Please provide an inpput .las or .laz file')

FLAGS = parser.parse_args()

FILE_NAME = FLAGS.file_name

def display_cloud(points):
    pc = PointCloud()
    pc.points = Vector3dVector(points)
    pc.colors = Vector3dVector(np.array([[1,0,0]] * len(pc.points)))
    draw_geometries([pc])

file = File(FILE_NAME, mode = 'r')

# data_points = file.points
# points_aug = np.vstack((data_points[0], data_points[1], data_points[2], data_points[5])).T   #dont forget the T for transpose
# points = np.vstack((data_points[0], data_points[1], data_points[2])).T

points = np.vstack((file.x, file.y, file.z)).T
print(points.shape)
display_cloud(points)


