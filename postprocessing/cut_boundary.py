from sklearn.neighbors import KDTree
import subprocess
import numpy as np
import trimesh
import polyscope as ps
from shapely.ops import cascaded_union
import time
import os
import sys
from scipy.spatial import ConvexHull
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon,LinearRing

def flip_boundary(boundary_file, output_file, input_file):
    args = ("flip-geodesics-demo/build/bin/flip_geodesics", boundary_file, output_file, input_file)
    print(args)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    #args = ("flip-geodesics-demo/build/bin/flip_geodesics", "flip-geodesics-demo/src/boundary.txt", "flip-geodesics-demo/build/my_mesh.obj", "flip-geodesics-demo/lion-head.off")
    popen.wait()
    output = popen.stdout.read()
    print(output)

def add_boundary_points(mesh, boundary):
    print(np.unique(np.unique(np.sort(mesh.edges, axis = 1), axis = 0, return_counts=True)[1]))
    print(len(mesh.vertices), np.max(boundary))
    point_exists = np.array([boundary[i] in np.unique(mesh.faces) for i in range(len(boundary))])
    to_add_vertex_idx = np.array(range(len(mesh.vertices)))[boundary[~point_exists]]
    to_add_vertex = mesh.vertices[boundary[~point_exists]]
    for i in range(len(to_add_vertex)):
        try:
            to_change_face = trimesh.proximity.closest_point(mesh,  to_add_vertex[i:i+1])[2]

        except Exception as e:
            to_change_face = trimesh.proximity.closest_point_naive(mesh,  to_add_vertex[i:i+1])[2]


        faces = [Polygon(mesh.triangles[i,:,:2]) for i in range(len(mesh.triangles))]
        shape = cascaded_union(faces)
        point = Point(to_add_vertex[i,:2])
        is_within =point.within(shape)
        if not is_within:
            triangle = mesh.triangles[to_change_face]
            edges_idx = np.array([[0, 1], [1, 2], [2, 0]])
            e1 = LineString([Point(triangle[0, 0,:2]), Point(triangle[0, 1,:2])])
            e2 = LineString([Point(triangle[0, 1,:2]), Point(triangle[0, 2,:2])])
            e3 = LineString([Point(triangle[0, 2,:2]), Point(triangle[0, 0,:2])])
            closest_edge = edges_idx[np.argmin([point.distance(e1),point.distance(e2),point.distance(e3)])]

        old_face = mesh.faces[to_change_face[0]]
        new_point = to_add_vertex_idx[i]
        if is_within:
            new_faces =np.array([np.append( old_face[:2], new_point),  np.append( old_face[1:], new_point), np.array( [old_face[2], old_face[0], new_point])])
            new_faces= np.concatenate([new_faces, np.delete(mesh.faces,to_change_face, axis = 0)])
        else:
            new_face = np.concatenate([mesh.faces[to_change_face[0]][closest_edge],np.array([to_add_vertex_idx[i]])])
            new_faces= np.concatenate([mesh.faces, [new_face]])
        mesh = trimesh.Trimesh(mesh.vertices, new_faces, process=False)

        print(i, len(to_add_vertex))

    trimesh.repair.fix_winding(mesh)
    print(np.unique(np.unique(np.sort(mesh.edges, axis = 1), axis = 0, return_counts=True)[1]))

    return mesh





def cut_outside_init(mesh, boundary_edges):

    polygon = Polygon(mesh.vertices[np.append(boundary_edges[:,0],boundary_edges[0,0])][:,:2]) # create polygon
    triangle_centers = np.mean(mesh.triangles, axis = 1)
    triangle_centers = [Point(p[0], p[1]) for p in triangle_centers] # create point

    triangles = [Polygon(t) for t in mesh.triangles]
    valid_faces =  np.array([polygon.intersects(t) and ((not polygon.touches(t)) and (polygon.intersection(t).area>1e-8)) for t in triangles])

    exterior_mesh = mesh.copy()
    exterior_mesh.update_faces(valid_faces)
    exterior_mesh_poly = [Polygon(t) for t in exterior_mesh.triangles]
    exterior_mesh_poly = cascaded_union(exterior_mesh_poly)
    if exterior_mesh_poly.geom_type == 'MultiPolygon':
        polys = list(exterior_mesh_poly)
        polys_area = [p.area for p in polys]
        exterior_mesh_poly = polys[np.argmax(np.array(polys_area))]


    exterior_mesh_poly = Polygon(exterior_mesh_poly.exterior.coords)
    valid_faces =  np.array([exterior_mesh_poly.contains(t) for t in triangles])

    mesh.update_faces(valid_faces)

    return mesh

def cut_outside(mesh, boundary_edges):
    polygon = Polygon(mesh.vertices[np.append(boundary_edges[:,0],boundary_edges[0,0])][:,:2]) # create polygon
    triangle_centers = np.mean(mesh.triangles, axis = 1)
    triangle_centers = [Point(p[0], p[1]) for p in triangle_centers] # create point
    valid_faces = np.array([polygon.contains(p) or polygon.touches(p) for p in triangle_centers])
    mesh.update_faces(valid_faces)
    return mesh


def combine_patches(patches):
    points = [patch.vertices for patch in patches]
    offset = np.cumsum(np.array([len(point) for point in points]))
    offset = np.concatenate([np.array([0]), offset])
    faces = [patches[i].faces + offset[i] for i in range(len(patches))]

    points = np.concatenate(points)
    faces = np.concatenate(faces)
    mesh=trimesh.Trimesh(points, faces)
    trimesh.repair.fix_winding(mesh)
    return mesh

def WeightedDelaunay(points,weights):
    num, dim = np.shape(points)
    lifted = np.zeros((num,dim+1))
    for i in range(num):
        p = points[i,:]
        lifted[i,:] = np.append(p,np.sum(p**2) - weights[i])
    pinf = np.append(np.zeros((1,dim)),1e10);
    lifted = np.vstack((lifted, pinf))
    hull = ConvexHull(lifted)
    delaunay = []
    for simplex in hull.simplices:
        if num not in simplex:
            delaunay.append(simplex.tolist())
    return delaunay

def cut(input_file,input_3D_file,boundary, weights=None):
    boundary_edges = np.array([boundary[i:i+2] for i in range(len(boundary) -1)])
    boundary_edges = np.concatenate([boundary_edges, np.array([[boundary[-1], boundary[0]]])])
    init_mesh = trimesh.load(input_file, process=False)
    init_3D_mesh = trimesh.load(input_3D_file, process=False)

    # if the patch is not manifold use the weights to recompute the weighted delaunay
    print("is manifold : ",np.unique(np.unique(np.sort(init_mesh.edges, axis = 1), axis = 0, return_counts=True)[1]))
    if np.unique(np.unique(np.sort(init_mesh.edges, axis = 1), axis = 0, return_counts=True)[1])[-1]>=3:
        print('non manifold!')
        if weights is None:
            weights = np.zeros(init_mesh.vertices.shape[0])
        weighted_faces = WeightedDelaunay(init_mesh.vertices[:,:2], weights)
        init_mesh.faces = weighted_faces
        print("is manifold : ",np.unique(np.unique(np.sort(init_mesh.edges, axis = 1), axis = 0, return_counts=True)[1]))
        init_3D_mesh.faces = weighted_faces
    add_boundary_mesh = add_boundary_points(init_mesh, boundary)
    add_boundary_mesh = cut_outside_init(add_boundary_mesh, boundary_edges)

    referenced_vertices = np.unique(add_boundary_mesh.faces)
    tree = KDTree(add_boundary_mesh.vertices[referenced_vertices])
    boundary_points = np.unique(boundary_edges)
    unreferenced_boundary = list(set(boundary_points) - set(referenced_vertices))
    referenced = np.array([v in referenced_vertices for v in range(len(add_boundary_mesh.vertices))])
    if len(unreferenced_boundary)>0:
        inverse_map = dict(zip(np.array(range(sum(referenced))),np.array(range(len(add_boundary_mesh.vertices)))[referenced]))

        closest_referenced_vertex = tree.query(add_boundary_mesh.vertices[unreferenced_boundary], k =2)[1][:, 1]
        for k in range(len(unreferenced_boundary)):
            print(unreferenced_boundary[k], closest_referenced_vertex[k])
            boundary_edges[boundary_edges==unreferenced_boundary[k]] = inverse_map[closest_referenced_vertex[k]]
    map = dict(zip(np.array(range(len(add_boundary_mesh.vertices)))[referenced], np.array(range(sum(referenced)))))

    add_boundary_mesh.remove_unreferenced_vertices()

    boundary_edges = np.vectorize(map.get)(boundary_edges)
    boundary_edges = boundary_edges[boundary_edges[:,0]!=boundary_edges[:,1]]

    tmp_add_boundary = "tmp_add_boundary{}.obj".format(unique_label)
    tmp_boundary = 'tmp_boundary_{}.txt'.format(unique_label)
    add_boundary_mesh.export(tmp_add_boundary)
    np.savetxt(tmp_boundary, boundary_edges,fmt='%i')
    time.sleep(1)
    print("n trig points",len(np.unique(add_boundary_mesh.faces)))
    print("n points",len(add_boundary_mesh.vertices))

    output_file = "tmp_output_{}.obj".format(unique_label)
    flip_boundary(tmp_boundary, output_file, tmp_add_boundary)
    time.sleep(1)

    mesh = trimesh.load(output_file, process=False)
    mesh = cut_outside(mesh, boundary_edges)
    new_3D_mesh = trimesh.Trimesh(init_3D_mesh.vertices[referenced], mesh.faces)
    return new_3D_mesh

if __name__ == '__main__':
    ps.init()
    unique_label = "label0" # change this label if the code in run multiple time in parralel because we save tmp files
    shapes = ["100349"]
    task = "curvature"
    list_n_patches = [10]*len(shapes)
    for shape, n_patches in list(zip(shapes, list_n_patches)):
        param_path = os.path.join(ROOT_DIR, "data/{}/param_{}".format(shape, shape))
        optim_path = os.path.join(ROOT_DIR, "data",task, shape)
        out_path = os.path.join(ROOT_DIR,"data", task)

        patches = []
        init_patches = []
        for n_patch in range(n_patches): #9
            input_file = os.path.join(optim_path,"opt_mesh2D_{}.ply".format( n_patch))
            input_3D_file = os.path.join(optim_path,'opt_mesh_{}.ply'.format(n_patch))

            boundary = np.load(os.path.join(param_path,"boundary_{}.npy".format(n_patch)))
            weights = np.load(os.path.join(optim_path, 'weights_{}.npy'.format(n_patch)))

            patches.append(cut(input_file,input_3D_file,boundary, weights=weights))

            input_file = os.path.join(optim_path,"opt_mesh_init_{}.ply".format(n_patch))
            input_3D_file =  os.path.join(optim_path,"opt_mesh_init3D_{}.ply".format(n_patch))

            init_patches.append(cut(input_file,input_3D_file,boundary))
        mesh = combine_patches(patches)
        mesh.export(os.path.join(out_path,"optim_{}.ply".format(shape)))
        mesh = combine_patches(init_patches)
        mesh.export(os.path.join(out_path,"init_{}.ply".format(shape)))
