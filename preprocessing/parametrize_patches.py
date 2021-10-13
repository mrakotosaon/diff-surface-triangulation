import shutil
import networkx as nx
import os
import trimesh
import numpy as np
import igl
from scipy.spatial import KDTree
from shapely import geometry
import sys
import shapely

def compute_boundaries(patch):
    unique_edges, counts = np.unique(patch.edges_unique_inverse, return_counts = True)
    boundary_edges = patch.edges_unique[unique_edges[counts==1]]
    G = nx.Graph()
    G.add_nodes_from(range(len(patch.vertices)))
    G.add_edges_from(boundary_edges)
    boundaries = [list(x) for x in sorted(nx.connected_components(G), key = len, reverse=True) if len(x)>1]
    ordered_boundaries = []
    for boundary in boundaries:
        ordered_boundaries.append([x[0] for x in nx.find_cycle(G, boundary[0])])
    return ordered_boundaries


def recut_patch(patch, cut_node, dist_nn):
    boundaries = compute_boundaries(patch)
    count=1
    while cut_node in boundaries[0]:
        cut_node = np.argsort(np.mean(dist_nn, axis = 1))[count]
        count+=1

    G= init_graph(patch)
    path = find_shortest_path(G, boundaries[0],  [cut_node])
    new_patch = cut_path_in_mesh(path, patch)
    return new_patch

def init_graph(patch):
    G = nx.Graph()
    edges = [(e[0][0], e[0][1], {"weight": e[1]}) for e in zip(patch.edges_unique, patch.edges_unique_length)]
    G.add_nodes_from(range(len(patch.vertices)))
    G.add_edges_from(edges)
    return G

def find_shortest_path(G, boundaryA, boundaryB):
    shortest_path = 1e8
    for p1 in boundaryA:
        for p2 in boundaryB:
            path_length = nx.shortest_path_length(G,source=p1,target=p2, weight='weight')
            if path_length<shortest_path:
                shortest_path = path_length
                shortest_path_couple = [p1, p2]

    path = nx.shortest_path(G,source=shortest_path_couple[0],target=shortest_path_couple[1], weight='weight')
    return path

def cut_path_in_mesh(path, patch):
    to_remove_edges = []
    # find faces that are adjacent to the separation path
    for i in range(len(path)-1):
        path_edge = path[i:i+2]
        to_remove_edges.append([j for (j, e) in enumerate(patch.edges) if (e[0]==path_edge[0]) and (e[1]==path_edge[1])])
    for i in range(len(path)-1):
        path_edge = np.concatenate([[path[i+1]], [path[i]]])
        to_remove_edges.append([j for (j, e) in enumerate(patch.edges) if (e[0]==path_edge[0]) and (e[1]==path_edge[1])])
    to_remove_edges = np.array(to_remove_edges).reshape([-1])
    to_remove_faces = patch.edges_face[to_remove_edges]
    to_remove_faces_idx = patch.faces[to_remove_faces]
    to_remove_faces2 = []
    for i in range(len(to_remove_edges)):
        for k in range(3):
            if to_remove_faces_idx[i,k] == patch.edges[to_remove_edges][i, 0]:
                to_remove_faces2.append(k)
    face_mask = np.zeros_like(patch.faces)
    face_mask[to_remove_faces, to_remove_faces2] =1
    cuts = igl.cut_to_disk(patch.faces)
    v, f = igl.cut_mesh(patch.vertices, patch.faces,face_mask )

    new_patch = trimesh.Trimesh(v, f, process=False)
    return new_patch

def cut_patch(patch):
    boundaries = compute_boundaries(patch)
    while len(boundaries)>1:
        G = init_graph(patch)
        path = find_shortest_path(G, boundaries[0], boundaries[1])
        trimesh.repair.fix_winding(patch)
        new_patch = cut_path_in_mesh(path, patch)
        boundaries = compute_boundaries(new_patch)
        patch = new_patch
    return patch, boundaries


def parametrize(patch, bnd=None):
    b = np.array([2, 1])
    if bnd is None:
        bnd = igl.boundary_loop(patch.faces)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    # LSCM parametrization
    _, uv = igl.lscm(patch.vertices, patch.faces, b, bc)
    return uv

def compute_mean_curvature(gt_mesh):
    gt_pd1, gt_pd2, gt_v1, gt_v2 = igl.principal_curvature(gt_mesh.vertices, gt_mesh.faces)
    gt_normals = gt_mesh.vertex_normals
    gt_mean_curv = np.abs(gt_v1) + np.abs(gt_v2)
    M = np.quantile(gt_mean_curv, 0.9)
    m = np.quantile(gt_mean_curv, 0.1)
    gt_mean_curv = np.clip(gt_mean_curv, m , M)
    gt_mean_curv = (gt_mean_curv- m) /(M - m)
    return gt_pd1, gt_pd2, gt_mean_curv


if __name__ == '__main__':
    with open('shapes.txt') as f:
        shapes = f.read().splitlines()
    list_n_patches = [10]*len(shapes)
    valid_shapes = []
    count = 0
    count_valid = 0
    for shape, n_patches in list(zip(shapes,list_n_patches)):
        #try:
            print('attempting to parametrize shape', shape)
            patches = []
            mean_curvatures = []
            boundaries = []
            patches_2D = []
            principal_curvatures = []
            normals = []
            save_path = "preprocessing_data"
            gt_mesh = trimesh.load(os.path.join(save_path,"{}/{}.ply".format(shape,shape)), process=False)
            gt_pd1, gt_pd2, gt_mean_curv = compute_mean_curvature(gt_mesh)
            gt_normals = gt_mesh.vertex_normals
            gt_tree = KDTree(gt_mesh.vertices)
            valid = True
            for i in range(n_patches):

                print("patch",i)
                valid_patch =True
                patch = trimesh.load(os.path.join(save_path,'{}/patches/patch_{}.off'.format(shape, i)), process=False)
                trimesh.repair.fix_winding(patch)
                patch, boundary = cut_patch(patch)
                patch_2D = parametrize(patch, np.array(boundary[0]))

                tree = KDTree(patch_2D)
                dist_nn, _ = tree.query(patch_2D, k=3)
                closeness_ratio = np.min(np.mean(dist_nn[:,1:],axis = 1))/np.mean(np.mean(dist_nn[:,1:],axis = 1))

                limit = 0.15
                k = 0
                # cut patch if distortion is high
                while closeness_ratio<limit and k<20:
                    k+=1
                    cut_node =  np.argmin(np.mean(dist_nn, axis = 1))
                    patch = recut_patch(patch, cut_node, dist_nn)
                    boundary = compute_boundaries(patch)
                    patch_2D = parametrize(patch)
                    tree = KDTree(patch_2D)
                    dist_nn, _ = tree.query(patch_2D, k=5)
                    closeness_ratio = np.min(np.mean(dist_nn[:,1:],axis = 1))/np.mean(np.mean(dist_nn[:,1:],axis = 1))


                # if distortion is still high the shape is invalid.
                if closeness_ratio<limit:
                    print("Distortion is too high, please consider building smaller patches", closeness_ratio)
                    valid = False
                    valid_patch=False
                    shutil.rmtree(os.path.join(save_path,shape))
                    print("delete", os.path.join(save_path,shape))
                    break
                query = gt_tree.query(patch.vertices, k=1)
                mean_curvature = gt_mean_curv[query[1]]
                pd1 = gt_pd1[query[1]]
                pd2 = gt_pd2[query[1]]
                patch_normals = gt_normals[query[1]]
                mean_curvatures.append(mean_curvature)
                normals.append(patch_normals)
                boundaries.append(boundary[0])
                patches.append(patch)

                # check for overlappings in the parametrization
                patch_mesh= trimesh.Trimesh(patch_2D, patch.faces)
                faces = [geometry.Polygon(patch_mesh.triangles[i]) for i in range(len(patch_mesh.triangles))]
                total_area = sum([f.area for f in faces])
                union = shapely.ops.cascaded_union(faces)
                union_area = union.area
                if total_area>union_area + 1e-8:
                    print("There are overlappings in the parametrization, please consider building smaller patches", closeness_ratio)
                    valid=False
                    valid_patch = False
                    shutil.rmtree(os.path.join(save_path,shape))
                    print("delete", os.path.join(save_path,shape))
                    break
                principal_curvature = np.concatenate([pd1, pd2],axis = 1)
                if not os.path.exists(os.path.join(save_path,"{}/param_{}".format(shape,shape))):
                    os.makedirs(os.path.join(save_path,"{}/param_{}".format(shape,shape)))

                patches_2D.append(patch_2D)
                principal_curvatures.append(principal_curvature)

                print('valid patch', valid_patch)

            count +=1
            print("shape:",shape, "valid: ", valid)
            if valid:
                for i in range(n_patches):
                    np.save(os.path.join(save_path,"{}/param_{}/pc_{}.npy".format(shape, shape, i)), principal_curvatures[i])
                    np.save(os.path.join(save_path,"{}/param_{}/param_{}.npy".format(shape,shape,i)), patches_2D[i])
                    patches[i].export(os.path.join(save_path,"{}/param_{}/cut_patch_{}.ply".format(shape,shape,i)))
                    np.save(os.path.join(save_path,"{}/param_{}/boundary_{}.npy".format(shape,shape,i)), np.array(boundaries[i]))
                    np.save(os.path.join(save_path,"{}/param_{}/mean_curvature_{}.npy".format(shape, shape, i)), mean_curvatures[i])
                    np.save(os.path.join(save_path,"{}/param_{}/normals_{}.npy".format(shape, shape, i)),normals[i])
                count_valid+=1
                valid_shapes.append(shape)
                with open(os.path.join(save_path,"valid_shapes.txt"), "w") as output:
                    output.write("\n".join(valid_shapes))
            else:
                import shutil
                shutil.rmtree(os.path.join(save_path,"{}/param_{}".format(shape, shape)))
                print("delete", os.path.join(save_path,shape))
            print('total shapes:', count, 'valid', count_valid)
        # except Exception as e:
        #     try:
        #         print('parametrization failed')
        #         shutil.rmtree(os.path.join(save_path,"{}/param_{}".format(shape, shape)))
        #         print("delete",os.path.join(save_path,shape))
        #     except Exception as e:
        #         pass
