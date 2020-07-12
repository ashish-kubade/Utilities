import os
from os.path import join
# Enthought library imports
from mayavi import mlab
from mayavi.mlab import *
import mayavi
from mayavi.api import Engine
import time
temp_file = argv[1]
# temp_file = '/home/ashj/DEM_results/FBRGB/fbIN128/thesis_plots/GT_0_10.ply'

# Render the dragon ply file
from mayavi.api import Engine
engine = Engine()
engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()

mlab.pipeline.surface(mlab.pipeline.open(temp_file))
mlab.points3d([0], [0], [1973.9], color=(0,1,0),scale_factor=5)
mlab.points3d([0], [398], [1829.6], color=(1,0,0),scale_factor=5)


import numpy as np
from mayavi import mlab
# x, y = np.mgrid[0:3:1,0:3:1]
s = mlab.pipeline.surface(mlab.pipeline.open(temp_file))
print(mlab.view())


@mlab.animate
def anim():
	scene = engine.scenes[0]
	camera_light1 = engine.scenes[0].scene.light_manager.lights[1]
	camera_light2 = engine.scenes[0].scene.light_manager.lights[2]
	camera_light3 = engine.scenes[0].scene.light_manager.lights[3]

	scene.scene.background = (0.0, 0.0, 0.0)
	scene.scene.foreground = (0,0,0.8)
	scene.scene.full_screen = 0	
	scene.scene.camera.elevation(-45)
	scene.scene.camera.orthogonalize_view_up()
	mlab.points3d([0], [0], [1973.9], color=(0,1,0),scale_factor=5)
	mlab.points3d([0], [398], [1829.6], color=(1,0,0),scale_factor=5)
	# scene.scene.camera.view_up = [0.0, 1.0, 0.0]
	# camera_light1.activate = False
	camera_light2.activate = False
	camera_light3.activate = False
	scene.scene.show_axes = True
	for i in range(130):		
		scene.scene.camera.azimuth(-3)
		
		scene.scene.render()
		time.sleep(0.1)
		mlab.savefig('newGT/view_{}.jpg'.format(i))

		yield
anim()

mlab.show()