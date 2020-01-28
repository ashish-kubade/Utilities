#!/home/ashj/anaconda3/bin/python
import copy
import numpy as np
import open3d as o3d
import sys

inp_mesh = sys.argv[1]

if __name__ == "__main__":

    print("Testing mesh in open3d ...")
    mesh = o3d.io.read_triangle_mesh(inp_mesh)
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("")

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    # o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])

    print("We make a partial mesh of only the first half triangles.")
    
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])