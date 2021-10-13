import os
import trimesh
import polyscope as ps
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
seen_vertices = np.array([])

def separate_patch(component, mesh):
    # separate mesh in patches
    PQ = trimesh.proximity.ProximityQuery(component)
    sd = PQ.signed_distance(mesh.vertices)

    is_inside = np.where(sd<0, np.zeros_like(sd), np.ones_like(sd))
    inside_points = np.array(range(len(mesh.vertices)))[is_inside>0]
    global seen_vertices
    seen_vertices = np.concatenate([inside_points, seen_vertices])
    inside_faces = []
    inside_faces_idx = []
    for i, t in enumerate(mesh.faces):
        c1 = (t[0] in inside_points) or (t[1] in inside_points) or (t[2] in inside_points)
        c2 = (t[0] in seen_vertices) and (t[1] in seen_vertices) and (t[2] in seen_vertices)
        if c1 and c2:
            inside_faces.append(t)
            inside_faces_idx.append(i)
    inside_faces = np.array(inside_faces)
    inside_faces_idx = np.array(inside_faces_idx)

    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[inside_faces_idx] = True
    patch = mesh.copy()
    patch.update_faces(mask)
    patch.remove_unreferenced_vertices()

    return patch


def separate_patch_spectral(mesh, n_patches):
    # separate mesh in patches
    adj = nx.to_numpy_matrix(mesh.vertex_adjacency_graph)
    adj = np.zeros([len(mesh.vertices),len(mesh.vertices)])

    for edge, edge_length in zip(mesh.edges_unique, mesh.edges_unique_length):
        adj[edge[0], edge[1]] = edge_length*10
        adj[edge[1], edge[0]] = edge_length*10
        #adj[edge[0], edge[1]] = 1#np.max(mesh.edges_unique_length)-edge_length#edge_length*10
        #adj[edge[1], edge[0]] = 1#np.max(mesh.edges_unique_length)-edge_length#edge_length*10

        #adj[edge[0], edge[1]] = 0.006/edge_length
        #adj[edge[1], edge[0]] = 0.006/edge_length
    sc = SpectralClustering(n_patches, affinity='precomputed', n_init=100)
    sc.fit(adj)
    all_label_faces = []
    faces_to_allocate = set(list(range(len(mesh.faces))))
    for label in range(n_patches):
        label_faces = []
        inside_vertices = np.array(range(len(mesh.vertices)))[sc.labels_ == label]
        for f in faces_to_allocate:
            if (mesh.faces[f][0] in inside_vertices) or (mesh.faces[f][1] in inside_vertices) or (mesh.faces[f][2] in inside_vertices):
                label_faces.append(f)
        faces_to_allocate = faces_to_allocate - set(label_faces)
        all_label_faces.append(label_faces)
    patches = []
    for patch_faces in all_label_faces:
        mask = np.zeros(len(mesh.faces), dtype=np.bool)
        mask[patch_faces] = True
        patch = mesh.copy()
        patch.update_faces(mask)
        patch.remove_unreferenced_vertices()
        patches.append(patch)
    return patches, sc.labels_



def spectral(mesh, n_patches):
    patches, labels = separate_patch_spectral(mesh, n_patches)
    largest_patch_size = np.max([len(p.vertices) for p in patches])
    print("largest patch size", largest_patch_size)
    if largest_patch_size<1000:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mesh.export(os.path.join(save_path, "{}.ply".format(shape)))

        if not os.path.exists(os.path.join(save_path,"patches")):
            os.makedirs(os.path.join(save_path,"patches"))

        for i in range(n_patches):
            patches[i].export(os.path.join(save_path,"patches/patch_{}.off".format(i)))

        for i,patch in enumerate(patches):
            ps.register_surface_mesh("or mesh {}".format(i), patch.vertices, patch.faces)
        ps.show()
    else:
        print('largest patch size should be <1000')



ps.init()
with open('shapes.txt') as f:
    shapes = f.read().splitlines()
list_n_patches = [10]*len(shapes)
seeds = [2]*len(shapes)
for i, shape, n_patches, seed in list(zip(range(len(shapes)),shapes, list_n_patches, seeds)):
    print(shape, i, "/", len(shapes))
    np.random.seed(seed=seed)
    save_path = "preprocessing_data/{}".format(shape)
    mesh = trimesh.load("preprocessing_data/{}.ply".format(shape), process=False)
    mesh.vertices -=np.mean(mesh.vertices, axis = 0)
    mesh.vertices/=np.sqrt(mesh.area)
    spectral(mesh, n_patches)
