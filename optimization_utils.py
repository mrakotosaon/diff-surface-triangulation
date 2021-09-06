import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_to_3D(w_u, w_v, w_w, points_3D, faces_3D):
    n_points = points_3D.shape[0]
    n_faces = w_u.shape[0]
    weights = tf.stack([w_u, w_v, w_w], axis = 2)
    face_vertices = tf.gather(points_3D, faces_3D)
    face_vertices = tf.tile(face_vertices[:, tf.newaxis], [1, n_points, 1, 1])
    coordinates = tf.multiply(face_vertices, tf.tile(weights[:,:,:,tf.newaxis], [1, 1, 1, points_3D.shape[-1]]))
    denom = tf.reduce_sum(tf.reduce_sum(weights, axis = -1), axis = 0)[:,tf.newaxis]
    denom = tf.where(denom<1e-8, tf.ones_like(denom), denom)
    coordinates = tf.reduce_sum(tf.reduce_sum(coordinates, axis = 0), axis =-2)
    coordinates = tf.divide(coordinates, denom)
    return coordinates


def resize_parametrization(parametrization, cut_mesh):
    mean_edge3D = np.sqrt(np.sum(np.square(cut_mesh.vertices[cut_mesh.faces][:,0] -  cut_mesh.vertices[cut_mesh.faces][:,1] ), axis = -1))
    mean_edge3D+= np.sqrt(np.sum(np.square(cut_mesh.vertices[cut_mesh.faces][:,0] -  cut_mesh.vertices[cut_mesh.faces][:,2] ), axis = -1))
    mean_edge3D+= np.sqrt(np.sum(np.square(cut_mesh.vertices[cut_mesh.faces][:,1] -  cut_mesh.vertices[cut_mesh.faces][:,2] ), axis = -1))
    mean_edge3D/=3
    mean_edge3D = np.quantile(mean_edge3D, 0.2)


    mean_edge = np.sqrt(np.sum(np.square(parametrization[cut_mesh.faces][:,0] -  parametrization[cut_mesh.faces][:,1] ), axis = -1))
    mean_edge+= np.sqrt(np.sum(np.square(parametrization[cut_mesh.faces][:,0] -  parametrization[cut_mesh.faces][:,2] ), axis = -1))
    mean_edge+= np.sqrt(np.sum(np.square(parametrization[cut_mesh.faces][:,1] -  parametrization[cut_mesh.faces][:,2] ), axis = -1))
    mean_edge/=3
    mean_edge = np.quantile(mean_edge, 0.2)


    parametrization*=mean_edge3D/mean_edge
    return parametrization

def entry_stop_gradients(points, mask):
    mask = mask[:,tf.newaxis]
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask * points) + mask_h * points



def compute_nearest_neighbors(points,  n_nearest_neighbors):
    n_points = points.shape[0]
    distances = tf.tile(points[tf.newaxis], [n_points, 1, 1])
    distances = tf.reduce_sum(tf.square(tf.transpose(distances, [1, 0, 2]) - distances),axis = -1)
    nn = tf.math.top_k(-tf.abs(distances), k=n_nearest_neighbors+1)[1]
    return nn



def compute_barycentric_coordinates(points, fixed_mesh_vertices, fixed_mesh_faces):
    n_points = points.shape[0]
    n_faces = fixed_mesh_faces.shape[0]
    fixed_mesh_vertices = tf.concat([fixed_mesh_vertices, tf.zeros([n_points, 1])], axis = 1)
    points = tf.concat([points, tf.zeros([n_points, 1])], axis = 1)
    p_a = tf.tile(tf.gather(fixed_mesh_vertices, fixed_mesh_faces[:, 0])[:, tf.newaxis,:], [1, n_points, 1])
    p_b = tf.tile(tf.gather(fixed_mesh_vertices, fixed_mesh_faces[:, 1])[:, tf.newaxis,:], [1, n_points, 1])
    p_c = tf.tile(tf.gather(fixed_mesh_vertices, fixed_mesh_faces[:, 2])[:, tf.newaxis,:], [1, n_points, 1])
    points = tf.tile(points[tf.newaxis,: ,:], [n_faces, 1, 1])

    area_ABC = tf.abs(tf.linalg.cross(p_b -p_a, p_c - p_a)[:, :, 2])
    area_PBC = tf.abs(tf.linalg.cross(p_b -points, p_c - points)[:, :, 2])
    area_PAB = tf.abs(tf.linalg.cross(p_b -points, p_a - points)[:, :, 2])
    area_PCA = tf.abs(tf.linalg.cross(p_c -points, p_a - points)[:, :, 2])

    u= tf.divide(area_PBC, area_ABC)
    v= tf.divide(area_PCA, area_ABC)
    w= tf.divide(area_PAB, area_ABC)
    mask = u + w + v>1+ 1e-6
    u = tf.where(mask, tf.zeros_like(u), u)
    v = tf.where(mask, tf.zeros_like(v), v)
    w = tf.where(mask, tf.zeros_like(w), w)

    return v, w, u


def make_boundary(boundary, vertices):
    boundary = np.stack([boundary, np.concatenate([boundary[1:],boundary[0:1]]), np.concatenate([boundary[-1:],boundary[0:-1]])], axis = 1)
    boundary_pc = vertices[boundary]
    v1 = boundary_pc[:, 1] - boundary_pc[:, 0]
    v2 = boundary_pc[:, 2] - boundary_pc[:, 0]
    v1 = np.divide(v1, np.linalg.norm(v1, axis = -1)[:, np.newaxis])
    v2 = np.divide(v2, np.linalg.norm(v2, axis = -1)[:, np.newaxis])
    sinus = np.cross(v1, v2)
    cosinus = np.sum(np.multiply(v1, v2), axis = -1)
    cosinus = np.clip(cosinus, -1, 1)
    sinus = np.clip(sinus, -1, 1)
    sign = np.sign(sinus)
    sign = np.where(sign==0, np.ones_like(sign), sign)
    angle = np.arccos(cosinus)*sign/2
    rotation = R.from_euler('z', angle, degrees=False).as_matrix()[:,:2, :2]
    normal = np.squeeze(np.matmul(rotation, v1[:,:,np.newaxis]))
    sign = np.sign(np.cross(v1, normal)[:,np.newaxis])
    sign = np.where(sign==0, np.ones_like(sign), sign)

    normal*=sign
    sign_check = np.mean(np.sign(np.sum(np.multiply(np.mean(vertices, axis = 0) - boundary_pc[:, 0], normal), axis = -1)))
    if sign_check<0:
        normal*=-1
    return boundary, normal




def init_config():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return config
