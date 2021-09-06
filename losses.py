import tensorflow as tf
import numpy as  np

def safe_norm(x, epsilon=1e-8, axis=None):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x) , axis=axis), epsilon))

def curvature_loss_min(curvature_direction, approx_triangles, indices,points, n_trigs):
    coordA = tf.gather(points,indices[:,:,0], batch_dims=1)
    coordB = tf.gather(points,indices[:,:,1], batch_dims=1)
    coordC = tf.gather(points,indices[:,:,2], batch_dims=1)
    AB = coordB - coordA
    AC = coordC - coordA
    vectors = tf.concat([AB,AC], axis = 1)
    normalized_AB = tf.divide(AB, tf.maximum(safe_norm( AB, axis = -1)[:,:,tf.newaxis], 1e-6))
    normalized_AC = tf.divide(AC, tf.maximum(safe_norm( AC, axis = -1)[:,:,tf.newaxis], 1e-6))
    normalized_vectors = tf.concat([normalized_AB,normalized_AC], axis = 1)
    curvature_direction = tf.tile(curvature_direction[:,tf.newaxis] , [1, n_trigs*2,1])
    approx_triangles = tf.tile(approx_triangles , [1, 2])
    dot_product = tf.reduce_sum(tf.multiply(curvature_direction, normalized_vectors), axis = -1)
    dot_product = tf.multiply(dot_product, approx_triangles)
    tmp = -tf.math.reduce_logsumexp(dot_product*1000, axis = -1) - tf.math.reduce_logsumexp(-dot_product*1000, axis = -1)
    tmp*=1e-3
    curv_loss_old = tf.divide(tf.reduce_sum(tmp), tf.reduce_sum(approx_triangles))
    return curv_loss_old


def curvature_loss_measure(curvature_direction, approx_triangles, indices,points, n_trigs):
    coordA = tf.gather(points,indices[:,:,0], batch_dims=1)
    coordB = tf.gather(points,indices[:,:,1], batch_dims=1)
    coordC = tf.gather(points,indices[:,:,2], batch_dims=1)
    AB = coordB - coordA
    AC = coordC - coordA
    vectors = tf.concat([AB,AC], axis = 2)
    normalized_AB = tf.divide(AB, tf.maximum(safe_norm( AB, axis = -1)[:,:,tf.newaxis], 1e-6))
    normalized_AC = tf.divide(AC, tf.maximum(safe_norm( AC, axis = -1)[:,:,tf.newaxis], 1e-6))
    normalized_vectors = tf.concat([normalized_AB,normalized_AC], axis = 1)
    curvature_direction = tf.tile(curvature_direction[:,tf.newaxis] , [1, n_trigs*2,1])
    approx_triangles = tf.tile(approx_triangles , [1, 2])

    dot_product = tf.reduce_sum(tf.multiply(curvature_direction, normalized_vectors), axis = -1)
    dot_product = tf.where(approx_triangles>0.5, dot_product, tf.zeros_like(dot_product))
    ignored_points = tf.reduce_sum(tf.where(approx_triangles>0.5, tf.ones_like(approx_triangles), tf.zeros_like(approx_triangles)), axis= -1)
    ignored_points = tf.where(ignored_points<0.5, tf.zeros_like(ignored_points), tf.ones_like(ignored_points))
    measure = tf.reduce_sum(- tf.reduce_min(dot_product, axis = 1))/tf.reduce_sum(ignored_points)
    measure+= tf.reduce_sum(tf.reduce_max(dot_product, axis = 1))/tf.reduce_sum(ignored_points)
    measure*=0.5
    return measure


def quad_angles_loss( approx_triangles, indices,points):
    coordA = tf.gather(points,indices[:,:,0], batch_dims=1)
    coordB = tf.gather(points,indices[:,:,1], batch_dims=1)
    coordC = tf.gather(points,indices[:,:,2], batch_dims=1)
    dot_productA = compute_angle(coordA, coordB, coordC)
    dot_productB = compute_angle(coordB, coordA, coordC)
    dot_productC = compute_angle(coordC, coordA, coordB)
    A = 45.0*np.pi/180.0
    B = 60.0*np.pi/180.0
    target_angle1 =  tf.ones_like(coordA[:,:,0])*B
    target_angle2 = np.pi - target_angle1*2
    target_angle = target_angle1
    target_angle1 = tf.math.cos(target_angle)
    target_angle2 = tf.math.cos(target_angle)
    diff = tf.abs(tf.abs(dot_productA) - target_angle2) + tf.abs(tf.abs(dot_productB) - target_angle1) + tf.abs(tf.abs(dot_productC) - target_angle1)

    angles_loss = tf.divide(tf.reduce_sum(tf.multiply(diff, approx_triangles)),tf.reduce_sum(approx_triangles))
    return angles_loss



def compute_face_areas(points_3D, prob, indices, neighbors):
    global_indices = tf.gather(neighbors, indices, batch_dims=1)
    triangle_points = tf.gather(points_3D, global_indices)
    normals = tf.linalg.cross(triangle_points[:,:,1] -triangle_points[:,:,0], triangle_points[:,:,2] - triangle_points[:,:,0])
    face_area = safe_norm(normals, axis = -1)
    #norm = tf.tile(safe_norm(normals, axis = -1)[:,:,tf.newaxis],[1, 1, 3])
    #div = tf.where(norm<1e-7, tf.ones_like(normals), norm)
    #normals = tf.divide(normals, div)
    return face_area


def quad_angles_measure( approx_triangles, indices,points):
    coordA = tf.gather(points,indices[:,:,0], batch_dims=1)
    coordB = tf.gather(points,indices[:,:,1], batch_dims=1)
    coordC = tf.gather(points,indices[:,:,2], batch_dims=1)
    AB = coordB - coordA
    AC = coordC - coordA
    normalized_AB = tf.divide(AB, tf.maximum(safe_norm( AB, axis = -1)[:,:,tf.newaxis], 1e-7))
    normalized_AC = tf.divide(AC, tf.maximum(safe_norm( AC, axis = -1)[:,:,tf.newaxis], 1e-7))
    dot_product = tf.abs(tf.reduce_sum(tf.multiply(normalized_AB, normalized_AC), axis = -1))
    diff_quad = tf.abs(tf.abs(dot_product) - 0.5)

    filtered_diff_quad = tf.where(approx_triangles>0.5, diff_quad, tf.zeros_like(diff_quad))
    count_filtered_diff_quad = tf.where(approx_triangles>0.5, tf.ones_like(diff_quad), tf.zeros_like(diff_quad))
    div = tf.reduce_sum(count_filtered_diff_quad, axis = -1)
    div = tf.where(div>0, div, tf.ones_like(div))
    angles_measure = tf.reduce_mean(tf.divide(tf.reduce_sum(filtered_diff_quad, axis = -1),div))
    return angles_measure

def compute_angle(coordA, coordB, coordC):
    AB = coordB - coordA
    AC = coordC - coordA
    normalized_AB = tf.divide(AB, tf.maximum(safe_norm( AB, axis = -1)[:,:,tf.newaxis], 1e-6))
    normalized_AC = tf.divide(AC, tf.maximum(safe_norm( AC, axis = -1)[:,:,tf.newaxis], 1e-6))
    dot_product = tf.abs(tf.reduce_sum(tf.multiply(normalized_AB, normalized_AC), axis = -1))
    return dot_product


def maximize_face_areas(face_areas, prob,  points_curvature, target_indices, neighbors):
    target_size = 0.001#0.0003
    global_indices = tf.gather(neighbors, target_indices, batch_dims=1)
    curvature= tf.minimum(tf.maximum(1- tf.gather(points_curvature, global_indices[:,:,0]),0.0),1.0)#1.0 - tf.gather(points_curvature, global_indices[:,:,0])
    target_size = 0.00005 + curvature*0.001
    size_loss = tf.square(face_areas-target_size)*10000
    target_size2 = 0.002#0.001
    size_loss2 = tf.square(tf.maximum(face_areas-target_size2, 0))*10000
    loss = tf.divide(tf.reduce_sum(tf.multiply(size_loss, prob)), tf.reduce_sum(prob))
    loss2 = tf.divide(tf.reduce_sum(tf.multiply(size_loss2, prob)), tf.reduce_sum(prob))

    return loss + loss2

def boundary_repulsion_loss(points, boundary, non_boundary, normal,n_boundary ):
    boundary_points = tf.gather(points, boundary[:,0])
    non_boundary_points = tf.gather(points, non_boundary)
    tmp_points = tf.tile(non_boundary_points[:,tf.newaxis],[1, n_boundary,1] )
    tmp_boundary_points = tf.tile(boundary_points[tf.newaxis], [tmp_points.shape[0], 1, 1] )
    closest_boundary_dist, closest_boundary_index = tf.math.top_k(-safe_norm(tmp_points - tmp_boundary_points, axis = -1))
    closest_boundary_dist = tf.abs(closest_boundary_dist)
    closest_boundary = tf.gather(boundary,tf.squeeze(closest_boundary_index))
    closest_boundary = tf.gather(points, closest_boundary)
    closest_boundary_normal = tf.gather(normal,tf.squeeze(closest_boundary_index))
    v1 =closest_boundary[:, 1]-closest_boundary[:, 0]
    v2 =closest_boundary[:, 2]-closest_boundary[:, 0]
    norm_v1 = safe_norm(v1, axis = -1)
    norm_v2 = safe_norm(v2, axis = -1)
    v1_normalized = tf.divide(v1, norm_v1[:,tf.newaxis])
    v2_normalized = tf.divide(v2, norm_v2[:,tf.newaxis])
    point_vector = non_boundary_points - closest_boundary[:, 0 ]
    dot_v1 = tf.reduce_sum(tf.multiply(point_vector , v1_normalized), axis = -1)
    dot_v2 = tf.reduce_sum(tf.multiply(point_vector , v2_normalized), axis = -1)

    distance_to_v1 = safe_norm(point_vector - tf.multiply(v1_normalized, dot_v1[:, tf.newaxis]), axis = -1)
    distance_to_v2 = safe_norm(point_vector - tf.multiply(v2_normalized, dot_v2[:, tf.newaxis]), axis = -1)
    dist_sign = tf.sign(tf.reduce_sum(tf.multiply(point_vector, closest_boundary_normal), axis = -1))
    distance_to_v1 = tf.multiply(distance_to_v1, dist_sign)
    distance_to_v2 = tf.multiply(distance_to_v2, dist_sign)

    # point projection is on the segment if dot_v1 < norm(v1)
    threshold = 0.01
    l1 = tf.where((dot_v1<norm_v1+0.001) & (dot_v1>-0.001), tf.exp(threshold-tf.minimum(distance_to_v1, threshold))-1 , tf.zeros_like(distance_to_v1))
    l2 = tf.where((dot_v2<norm_v2+0.001) & (dot_v2>-0.001), tf.exp(threshold-tf.minimum(distance_to_v2, threshold))-1 , tf.zeros_like(distance_to_v2))
    loss = tf.reduce_mean(tf.maximum(l1, l2))
    return loss
