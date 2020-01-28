import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Ellipse, Rectangle
root = '/home/ashj/Documents/NeuralNetworks/fcn-terrains/ContourData/RawData/terrains/orthos/test'
regions = os.listdir(root)

region = regions[3]
dem_root = '/home/ashj/Documents/Research/DEM_Experiments/All_res/ref_results'

img = cv2.imread(os.path.join(root, region))[:,:,::-1]
models = ['2m', '15m', 'fcn', 'fcnd', 'fb', 'fbo']
dems_names = [region[:-11] + '{}.npy'.format(i) for i in models]
print(dems_names)

# img = cv2.imread('/home/ashj/Pictures/1.png')
coords = []

# Make some example data
# x = np.random.rand(5)*img.shape[1]
# y = np.random.rand(5)*img.shape[0]

# Create a figure. Equal aspect so circles look circular
fig, ax = plt.subplots(1)
# ax.set_aspect('equal')

# Show the image
ax.imshow(img)
# x = np.array([575,1118,800])
# y = np.array([332,627,523])
# ecl = Ellipse((745, 372), 40, 250, angle=0, linewidth=2, fill=False, zorder=2)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)


# def onclick(event):
# 	print("entered here")
# 	ix, iy = event.xdata, event.ydata
# 	print('x = {}, y = {}'.format(ix,iy))
	
# 	coords.append((ix, iy))
# 	if len(coords) == 2:
# 		fig.canvas.mpl_disconnect(cid)
	
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
pts = plt.ginput(2, timeout = -1)
print(pts)
x = int(pts[0][0]) # x is column 
y = int(pts[0][1]) # y is row 
print(x)

height = int( pts[1][1] - pts[0][1] ) # height will be the difference of rows, so difference of ys
width = int( pts[1][0] - pts[0][0] ) # width will be the difference of columns, so difference of xs

print('width', width)

name = region[:-4] + '_r{}_c{}_w{}_h{}'.format(y,x,width,height) + '.jpg'


cv2.imwrite(name, img[y:y+height,x:x+width,::-1])

dx = x //2 
dy = y//2

dw = width // 2
dh = height // 2



for d  in dems_names:
	dem_out = d[:-4] + '_r{}_c{}_w{}_h{}'.format(dy,dx,dw,dh)
	dem = np.load(os.path.join(dem_root, d))
	np.save(dem_out, dem[dy:dy+dh, dx:dx+width])


# y = [400]
# x = [4400]

# width = [800]
# height = [800]


# x = [2200]
# y = [200]
# width = [400]
# height = [400]
color = ['black', 'fuchsia']
plt.close()


rect = Rectangle((x,y), width, height, fill = None, color = color[0], linewidth = 2 )
fig,ax = plt.subplots(1)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
ax.imshow(img)
# plt.savefig('test', dpi=100, bbox_inches='tight', pad_inches=0)
# np.save('test', img[x[idx]:x[idx]+width[idx], y[idx]:y[idx]+height[idx], :3])

ax.add_patch(rect)

out_path = 'dyna_rect_' + name
plt.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0)
# # ax.add_patch(ecl)
# # Show the image

# plt.show()

plt.close()