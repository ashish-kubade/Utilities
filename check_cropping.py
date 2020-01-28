import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
import os


img = np.loadtxt('/home/ashj/Documents/NeuralNetworks/fcn-terrains/ContourData/RawData/terrains/2m/bassiero_2m.dem', delimiter=',')

np.save('cropped_bas', img[2200:2600,200:600])
np.savetxt('cropped_bas_txt', img[2200:2600,200:600], fmt='%.4f')
print(img.shape)

# Make some example data
# x = np.random.rand(5)*img.shape[1]
# y = np.random.rand(5)*img.shape[0]

# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
# ax.set_aspect('equal')

# Show the image
ax.imshow(img)
# x = np.array([575,1118,800])
# y = np.array([332,627,523])
# ecl = Ellipse((745, 372), 40, 250, angle=0, linewidth=2, fill=False, zorder=2)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)



# y = [400]
# x = [4400]

# width = [800]
# height = [800]


x = [2200]
y = [200]
width = [400]
height = [400]
color = ['black', 'fuchsia']

for idx in range(len(x)):
	rect = Rectangle((x[idx],y[idx]), width[idx], height[idx], fill = None, color = color[idx], linewidth = 2 )
	# fig,ax2 = plt.subplots(1)
	# ax2.axes.get_xaxis().set_visible(False)
	# ax2.axes.get_yaxis().set_visible(False)
	# ax2.set_frame_on(False)
	# ax2.imshow(img[x[idx]:x[idx]+width[idx], y[idx]:y[idx]+height[idx], :3])
	# plt.savefig('test', dpi=100, bbox_inches='tight', pad_inches=0)
	# np.save('test', img[x[idx]:x[idx]+width[idx], y[idx]:y[idx]+height[idx], :3])

	ax.add_patch(rect)


# ax.add_patch(ecl)
# Show the image

# plt.show()
out_path = 'rect_out_bas_dem'
plt.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0)
plt.close()