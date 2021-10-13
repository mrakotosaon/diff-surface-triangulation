import polyscope as ps
import trimesh
import numpy as np
mesh=trimesh.load('../lion-head.off', process=False)
mesh2=trimesh.load('../build/my_mesh.obj', process=False)
boundary = np.loadtxt("boundary.txt").astype(int)
ps.init()
ps.register_surface_mesh("my mesh", mesh.vertices, mesh.faces, smooth_shade=True)
for i,b in enumerate(boundary):
    print(b)
    ps.register_point_cloud("my points{}".format(i), mesh.vertices[b])
ps.register_surface_mesh("my mesh2", mesh2.vertices, mesh2.faces, smooth_shade=True)

ps.show()
