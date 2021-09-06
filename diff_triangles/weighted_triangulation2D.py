import tensorflow as tf
import numpy as np
import config

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))


def get_middle_points_weighted(center_point, B, square_rB, w_center_point, is_boundary_center=None, is_boundary_B=None):
    n_points = center_point.shape[0]
    n_neighbors = B.shape[1]
    A = tf.tile(center_point[:,tf.newaxis,:], [1,n_neighbors, 1])
    square_rA = tf.tile(w_center_point[:,tf.newaxis], [1, n_neighbors])
    AB_square = tf.square(A - B)
    alpha = tf.where(tf.reduce_sum(AB_square, -1)<1e-8, tf.ones_like(square_rA)*0.5, 0.5 - tf.divide(square_rA - square_rB, 2.0*(tf.reduce_sum(AB_square, -1))))
    alpha = tf.tile(alpha[:,:,tf.newaxis], [1,1,2])
    middle_points = tf.multiply(A, alpha) + tf.multiply(1-alpha, B)
    return middle_points

def comp_half_planes(nn_coord, weights,center_point, w_center_point,is_boundary_center=None, is_boundary_B=None):
    # compute the equations of the half planes
    n_points = nn_coord.shape[0]
    n_neighbors = nn_coord.shape[1]
    middle_points = get_middle_points_weighted(center_point, nn_coord, weights, w_center_point,is_boundary_center=is_boundary_center, is_boundary_B=is_boundary_B)
    dir_vec=  nn_coord - center_point[:,tf.newaxis,:]
    dir_vec = gradient_clipping(dir_vec)
    half_planes_normal =  tf.divide(dir_vec,tf.maximum(tf.tile(safe_norm(dir_vec, axis = -1)[:,:,tf.newaxis],[1,1, 2]),config.EPS))

    col3 = - (middle_points[:,:,0]*half_planes_normal[:,:,0] + middle_points[:,:,1]*half_planes_normal[:,:,1] )
    half_planes = tf.concat([half_planes_normal, col3[:,:,tf.newaxis]], axis=-1)
    return half_planes


def is_inside(signed_dist, half_planes_normal):
  min_signed_distance = tf.reduce_min(signed_dist, 2)
  is_triangle =tf.math.sigmoid(min_signed_distance*1000)
  return is_triangle, signed_dist

def select_approx_triangles_distance(inter_dist0, n_neighbors, half_planes_normal):
    n_points = inter_dist0.shape[0]
    beta = 30.0#5.0#5.0#500.0#500.0#100.0#300#300.0#50.0
    inter_dist = -inter_dist0
    inter_dist = gradient_clipping(inter_dist)

    is_triangle, test = is_inside(inter_dist, half_planes_normal)
    return is_triangle

def get_is_trig_exact(inter_dist, n_neighbors):
    n_points = inter_dist.shape[0]
    inter_dist = -tf.sign(inter_dist)
    is_triangle = tf.reduce_sum(inter_dist, axis = 2)
    is_triangle = tf.where(is_triangle<n_neighbors, tf.zeros_like(is_triangle), tf.ones_like(is_triangle))
    return is_triangle

@tf.custom_gradient
def gradient_clipping(x):
  return x, lambda dy: tf.clip_by_norm(dy, 0.5)

def compute_intersections(half_planes, couples):
    # compute the intersections between the couples of half planes
    inter00 = tf.linalg.cross(tf.gather(half_planes,couples[:,0], axis=1), tf.gather(half_planes,couples[:,1], axis=1))
    inter0 = gradient_clipping(inter00)
    mask = tf.abs(inter0[:,:,2])<config.EPS
    inter1 = tf.divide(inter0,tf.tile(tf.expand_dims(tf.where(mask, tf.ones_like(inter0[:,:,2]),inter0[:,:,2]), 2),[1, 1,3]))
    inter = tf.where(tf.tile(mask[:,:,tf.newaxis], [1, 1, 3]), tf.ones_like(inter1)*1e7,inter1)
    return inter



def compute_triangles_local_geodesic_distances(nn_coord, nn_weights, center_point,  center_point_normal, nn_coord_normal,w_center_point, couples,normalized_normals=None, exact = True, compute_normals="exact", n_trigs = 1000, sigmoid_weight=None,is_boundary_center=None, is_boundary_B=None):
    n_neighbors = nn_coord.shape[1].value
    # project neighbors using geodesic distances

    nn_coord = nn_coord[:,:,:2]
    center_point = center_point[:,:2]
    half_planes1 =  comp_half_planes(nn_coord, nn_weights,center_point, w_center_point,is_boundary_center=is_boundary_center, is_boundary_B=is_boundary_B)
    half_planes = gradient_clipping(half_planes1)
    intersections=compute_intersections(half_planes, couples)
    # TODO couples should be half
    direction = intersections[:,:,:2] - tf.tile(center_point[:,tf.newaxis,:], [1, intersections.shape[1],1])
    direction1 = tf.where((direction[:,:,0]<0) & (direction[:,:,1]<0),tf.ones_like(direction[:,:,0], dtype=tf.bool), tf.zeros_like(direction[:,:,0], dtype=tf.bool))
    direction2 = tf.where((direction[:,:,0]<0) & (direction[:,:,1]>0),tf.ones_like(direction[:,:,0], dtype=tf.bool), tf.zeros_like(direction[:,:,0], dtype=tf.bool))
    direction3 = tf.where((direction[:,:,0]>0) & (direction[:,:,1]<0),tf.ones_like(direction[:,:,0], dtype=tf.bool), tf.zeros_like(direction[:,:,0], dtype=tf.bool))
    direction4 = tf.where((direction[:,:,0]>0) & (direction[:,:,1]>0),tf.ones_like(direction[:,:,0], dtype=tf.bool), tf.zeros_like(direction[:,:,0], dtype=tf.bool))
    distances = tf.reduce_sum(tf.square(intersections[:,:,:2] - tf.tile(center_point[:,tf.newaxis,:], [1, intersections.shape[1],1])), axis = -1)

    _, closest_intersections_idx1 = tf.math.top_k(-tf.where(direction1, distances, distances*1000), n_trigs//4)
    _, closest_intersections_idx2 = tf.math.top_k(-tf.where(direction2, distances, distances*1000), n_trigs//4)
    _, closest_intersections_idx3 = tf.math.top_k(-tf.where(direction3, distances, distances*1000), n_trigs//4)
    _, closest_intersections_idx4 = tf.math.top_k(-tf.where(direction4, distances, distances*1000), n_trigs//4)
    closest_intersections_idx = tf.concat([closest_intersections_idx1, closest_intersections_idx2, closest_intersections_idx3, closest_intersections_idx4], axis = 1)

    intersection_couples = tf.gather(tf.tile(couples[tf.newaxis,:,:],[center_point.shape[0], 1, 1]), closest_intersections_idx, batch_dims=1)
    intersections = tf.gather(intersections, closest_intersections_idx, batch_dims=1)
    # compute the distance between the intersection points (N**2 points) and the half planes (N)
    inter_dist00 = tf.reduce_sum(tf.multiply(tf.tile(half_planes[:,tf.newaxis,:,:],[1, n_trigs, 1, 1]) ,tf.tile(intersections[:,:,tf.newaxis,:],[1,1,n_neighbors, 1])), axis=-1)
    inter_dist00 = tf.where(tf.tile(intersections[:,:,0:1]>1e6,[1, 1, inter_dist00.shape[2]]), tf.ones_like(inter_dist00)*1e6, inter_dist00)
    index_couples_a = tf.tile(tf.range(center_point.shape[0])[:,tf.newaxis,tf.newaxis], [1, n_trigs, 2])
    index_couples_b = tf.tile(tf.range(n_trigs)[tf.newaxis,:,tf.newaxis], [center_point.shape[0], 1, 2])
    index_couples = tf.stack([index_couples_a, index_couples_b, intersection_couples], axis = -1)
    # for each triangle we want to ignore the current couple to compute the distance to a "virtual" voronoi cell
    to_ignore = tf.scatter_nd(tf.reshape(index_couples, [-1, 3]), tf.ones([center_point.shape[0]*n_trigs*2]), inter_dist00.shape)
    inter_dist0 = tf.where(to_ignore>0.5, -tf.ones_like(inter_dist00)*1e10,inter_dist00)
    inter_dist = tf.where(tf.abs(inter_dist0)<config.EPS,-tf.ones_like(inter_dist0)*1e10, inter_dist0)
    is_triangle_exact = get_is_trig_exact(inter_dist, n_neighbors)
    half_planes0 = half_planes[:,:,:2]
    is_triangle_approx = select_approx_triangles_distance(inter_dist0, n_neighbors,half_planes0)

    return is_triangle_exact, is_triangle_approx, intersection_couples,   center_point_normal
