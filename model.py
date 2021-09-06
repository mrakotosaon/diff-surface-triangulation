import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'diff_triangles'))
import tensorflow as tf
import numpy as np
import weighted_triangulation2D as w_tri
import numpy as np
import config


def safe_norm(x, epsilon=config.EPS, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(x ** 2, axis=axis), epsilon))


def compute_state_per_point_sparse_3D_full_batches(nn_tf, coord,weights,  couples, n_neighbours,   normals=None, compute_normals="exact",normalized_normals=None, full_index=False, n_trigs = 1000, consistent_nn_weight=None, gdist=None, first_index=None, sigmoid_weight=None,is_boundary_center=None, is_boundary_B=None):
    n_points = coord.shape[0].value
    nn_coord = tf.gather(coord, nn_tf)
    nn_coord_normal = tf.gather(normals, nn_tf)
    nn_coord = gdist[:,1:]
    center_point = gdist[:,0]
    nn_weights = weights[:,1:]
    w_center_point = weights[:,0]
    center_point_normal = normals[:center_point.shape[0]]
    exact_triangles, approx_triangles, local_indices, intersections= w_tri.compute_triangles_local_geodesic_distances(nn_coord, nn_weights, center_point,  center_point_normal,nn_coord_normal, w_center_point, couples,exact=False, compute_normals=compute_normals,normalized_normals=normalized_normals, n_trigs = n_trigs, sigmoid_weight=sigmoid_weight,is_boundary_center=is_boundary_center, is_boundary_B=is_boundary_B)

    global_indices = tf.gather(nn_tf, local_indices, batch_dims=1)
    first_index = tf.tile(first_index[:,tf.newaxis,tf.newaxis],[1, global_indices.shape[1], 1])
    global_indices = tf.concat([first_index, global_indices], axis = 2)
    return exact_triangles, approx_triangles, global_indices,  intersections, local_indices

def get_couples_matrix_sparse(shape):
    couples = []
    for i in range(1,shape):
        for j in range(i):
            couples.append([i,j])
    couples = np.array(couples)
    return couples

def get_triangles_geo_batches(points, normals, weights, n_neighbors=60, n_trigs=500, exact=False, gt_trigs = False, gdist=None, gdist_neighbors=None, first_index=None, sigmoid_weight=None,is_boundary_center=None, is_boundary_B=None):
    n_points = points.shape[0]
    couples = tf.constant(get_couples_matrix_sparse(n_neighbors), dtype = tf.int32)
    compute_normals =  "exact"
    weights0 = tf.abs(weights)
    exact_triangles, approx_triangles,  indices,  intersections, local_index= compute_state_per_point_sparse_3D_full_batches(gdist_neighbors,points,weights0, couples,n_neighbors,  normals=normals, compute_normals=compute_normals, normalized_normals = None, n_trigs=n_trigs, gdist=gdist, first_index=first_index, sigmoid_weight=sigmoid_weight,is_boundary_center=is_boundary_center, is_boundary_B=is_boundary_B)
    return approx_triangles, indices,  intersections, exact_triangles, local_index
