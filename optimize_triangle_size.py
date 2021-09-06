import os
import sys
import trimesh
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from scipy import interpolate
DEBUG = False # sets result saving
from sklearn.neighbors import KDTree
import tensorflow as tf
import importlib
import losses
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.join(BASE_DIR, 'models'))
import optimization_utils as utils
import model
n_patches = 10000
N_NEAREST_NEIGHBORS =80
n_trigs=2800




def init_graph(curvature_direction, parametrization, points_3D, faces_3D, boundary, non_boundary,boundary_normal, gt_normals, mean_curvature):
    config = utils.init_config()
    with tf.device('/gpu:'+str(0)):
        batch = tf.Variable(0)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        boundary = tf.constant(boundary, tf.int32)
        non_boundary = tf.constant(non_boundary, tf.int32)
        boundary_normal = tf.constant(boundary_normal, tf.float32)
        boundary_mask = tf.scatter_nd(boundary[:,0][:,tf.newaxis], tf.ones([N_BOUNDARY]), [n_points])
        points_2D = tf.constant(parametrization, dtype = tf.float32)
        parametrization = tf.Variable(parametrization, dtype = tf.float32)
        gt_normals = tf.constant(gt_normals, dtype = tf.float32)
        gt_mean_curvature = tf.constant(mean_curvature, dtype = tf.float32)
        full_points = utils.entry_stop_gradients(parametrization, boundary_mask)
        weights = tf.Variable(np.random.uniform(size=n_points)*0.00005 + 0.0001, dtype=tf.float32)
        use_weights = tf.minimum(tf.abs(weights), 0.002)
        neighbors = utils.compute_nearest_neighbors(full_points,N_NEAREST_NEIGHBORS)
        points_3D = tf.constant(points_3D, dtype = tf.float32)
        faces_3D = tf.constant(faces_3D, dtype = tf.int32)
        neighbor_points =tf.gather(full_points, neighbors)
        neighbors_boundary_mask = tf.gather(boundary_mask, neighbors)
        neighbor_points = neighbor_points - neighbor_points[:,0:1]
        neighbor_points = tf.concat([neighbor_points, tf.zeros([BATCH_SIZE,  N_NEAREST_NEIGHBORS+1, 1])], axis = 2)
        neighbor_weights = tf.gather(use_weights, neighbors)
        curvature_direction = tf.constant(curvature_direction, dtype=tf.float32)
        normals = np.zeros([BATCH_SIZE, 3])
        normals[:, 2] = 1
        normals = tf.constant(normals, dtype=tf.float32)
        points_idx = tf.constant(np.array(range(N_NEAREST_NEIGHBORS+1)))
        points_idx  = tf.tile(points_idx[tf.newaxis, :], [BATCH_SIZE, 1])
        target_approx_triangles, target_indices, _, exact_triangles, local_indices = model.get_triangles_geo_batches(neighbor_points,
                                                                                                                            normals,
                                                                                                                            tf.abs(neighbor_weights),
                                                                                                                            n_neighbors=N_NEAREST_NEIGHBORS,
                                                                                                                            n_trigs=n_trigs,
                                                                                                                            gdist = neighbor_points,
                                                                                                                            gdist_neighbors =points_idx[:,1:],
                                                                                                                            first_index =tf.zeros([BATCH_SIZE], dtype=tf.int64),
                                                                                                                            is_boundary_center=boundary_mask, is_boundary_B=neighbors_boundary_mask[:,1:])



        v, w, u = utils.compute_barycentric_coordinates(full_points, points_2D, faces_3D)
        converted_parametrization = utils.convert_to_3D(u, v, w, points_3D, faces_3D)
        target_curvature  = utils.convert_to_3D(u, v, w, curvature_direction, faces_3D)
        target_curvature = tf.divide(target_curvature, losses.safe_norm(target_curvature, axis = -1)[:, tf.newaxis])

        points_curvature = utils.convert_to_3D(u, v, w, gt_mean_curvature[:, tf.newaxis], faces_3D)
        points_curvature = tf.squeeze(points_curvature)

        neighbor_points_3D =tf.gather(converted_parametrization, neighbors)
        neighbor_points_3D = neighbor_points_3D - neighbor_points_3D[:,0:1]


        face_areas = losses.compute_face_areas(converted_parametrization, target_approx_triangles, target_indices, neighbors)
        face_area_loss = losses.maximize_face_areas(face_areas, target_approx_triangles, points_curvature, target_indices, neighbors)*500.0#*100000.0


        angles_loss = losses.quad_angles_loss( target_approx_triangles, target_indices,neighbor_points_3D)*0.5#*1000#*0.05#*1000.0
        angles_measure = losses.quad_angles_measure( target_approx_triangles, target_indices,neighbor_points_3D)
        point_distance2edge = losses.boundary_repulsion_loss(full_points, boundary, non_boundary, boundary_normal, N_BOUNDARY)
        point_distance2edge *=100.0

        loss =  point_distance2edge  + angles_loss + face_area_loss*10.0


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(loss,var_list = [parametrization, weights])
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

        train = optimizer.apply_gradients(capped_gvs,global_step=batch)


        init = tf.global_variables_initializer()
    session = tf.Session(config=config)
    session.run(init)



    ops = {"train": train,
            "loss": loss,
            "triangles":target_approx_triangles,
            "indices":target_indices,
            "use_weights": use_weights,
            "face_area_loss":face_area_loss,
            "neighbors": neighbors,
            "faces_3D": faces_3D,
            "points_3D": points_3D,
            "target_curvature": target_curvature,
            "converted_parametrization": converted_parametrization,
            "boundary": boundary,
            "parametrization":parametrization,
            "boundary_normal": boundary_normal,
            "curvature_direction": curvature_direction,
            "step": batch,
            "neighbor_points": neighbor_points,
            "point_distance2edge": point_distance2edge,
            "angles_loss":angles_loss,
            "weights": weights,
            "points": full_points,
            'learning_rate' : learning_rate,
            }
    return session, ops




def optimize(session, ops, boundary, original_mesh, save_path):
    l_loss= []
    n_loss= []
    angles_loss = []
    align_loss = []
    angles_measure = []
    point_distance2edge = []
    face_area_loss = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n_steps =  1500
    for step in range(n_steps):
        lr =0.00005
        train_op = ops['train']
        if step==0:
            feed_dict= {ops['learning_rate']: lr}
            to_run = [ops['triangles'], ops['indices'], ops['points'], ops['angles_loss'], ops['converted_parametrization'], ops['points_3D'], ops['neighbors'], ops['target_curvature']]
            triangles, indices, points_v, angles_loss_v, converted_parametrization, points_3D,  neighbors, target_curvature=session.run(to_run, feed_dict=feed_dict)
            indices = np.array([neighbors[i][indices[i]] for i in range(len(neighbors))])
            mesh_triangles = indices[triangles>0.5-1e-7]
            mesh_points = np.concatenate([points_v, np.zeros([len(points_v), 1])], axis = 1)
            mesh_triangles, count = np.unique(np.sort(mesh_triangles, axis = 1), return_counts=True, axis=0)
            mesh_triangles = mesh_triangles[count>2]
            mesh_old=trimesh.Trimesh(mesh_points, mesh_triangles, process=False)
            trimesh.repair.fix_winding(mesh_old)
            clean_faces = mesh_old.faces

            if not DEBUG:
                mesh_old.export(os.path.join(save_path,'opt_mesh_init_{}.ply'.format(mesh_number)))
                mesh_old3D=trimesh.Trimesh(converted_parametrization, clean_faces,process=False)
                print('trig max size', np.max(mesh_old3D.area_faces))
                print('trig min size', np.min(mesh_old3D.area_faces))
                print('trig mean size', np.mean(mesh_old3D.area_faces))

                mesh_old3D.export(os.path.join(save_path,'opt_mesh_init3D_{}.ply'.format(mesh_number)))
                np.save(os.path.join(save_path, 'target_curvature_init_{}.npy'.format(mesh_number)),target_curvature)

        feed_dict= {ops['learning_rate']: lr}

        to_run = [train_op,ops['step'] ,ops['loss'],ops['angles_loss'], ops['point_distance2edge'], ops['face_area_loss']]
        _,global_step, loss_v, angles_loss_v,point_distance2edge_v,face_area_loss_v=session.run(to_run, feed_dict=feed_dict)



        angles_loss.append(angles_loss_v)
        l_loss.append(loss_v)
        point_distance2edge.append(point_distance2edge_v)
        face_area_loss.append(face_area_loss_v)
        if (step ==0) or (step%100 == 99):
            print("step: {:4d}   Total loss:           {:0.3f}".format(step, np.mean(l_loss)))
            print("             Angles loss:          {:0.3f}   ".format( np.mean(angles_loss)  ))
            print("             Boundary repulsion:   {:0.3f}   ".format( np.mean(point_distance2edge)))
            print("             Face area loss:       {:0.3f}  " .format( np.mean(face_area_loss)))
            l_loss = []
            point_distance2edge = []
            angles_loss = []
            face_area_loss = []

        if step%1500 ==1499:
            feed_dict= {ops['learning_rate']: lr}
            to_run = [ops['triangles'], ops['indices'], ops['converted_parametrization'], ops['target_curvature'], ops['parametrization'], ops['neighbors'], ops['use_weights']]
            triangles, indices, points_v, target_curvature, points_2D, neighbors, weights_v=session.run(to_run, feed_dict=feed_dict)
            indices = np.array([neighbors[i][indices[i]] for i in range(len(neighbors))])
            mesh_triangles = indices[triangles>0.5-1e-7]
            mesh_triangles, count = np.unique(np.sort(mesh_triangles, axis = 1), return_counts=True, axis=0)
            mesh_triangles = mesh_triangles[count>2]

            mesh_points2D = np.concatenate([points_2D, np.zeros([len(points_2D), 1])], axis = 1)
            mesh2D=trimesh.Trimesh(mesh_points2D, mesh_triangles, process=False)
            trimesh.repair.fix_winding(mesh2D)
            clean_faces = mesh2D.faces
            mesh_points =points_v
            mesh=trimesh.Trimesh(mesh_points, clean_faces, process=False)
            print('trig max size', np.max(mesh.area_faces))
            if not DEBUG:
                np.save(os.path.join(save_path, 'target_curvature_{}.npy'.format(mesh_number)),target_curvature)
                mesh2D.export(os.path.join(save_path,'opt_mesh2D_{}.ply'.format(mesh_number)))

                mesh.export(os.path.join(save_path,'opt_mesh{}.ply'.format(mesh_number)))

                np.save(os.path.join(save_path, 'points2D_{}.npy'.format(mesh_number)),points_2D)
                np.save(os.path.join(save_path, 'points3D_{}.npy'.format(mesh_number)),mesh_points)
                np.save(os.path.join(save_path, 'weights_{}.npy'.format(mesh_number)),weights_v)





if __name__ == '__main__':
    shapes = ['100349']
    list_n_patches = [10]*len(shapes)
    save_path =  "data/triangle_size/{}"
    data_path = "data/{}"
    seen_shapes = -1
    for shape, n_patches in list(zip(shapes, list_n_patches)):
        if os.path.isfile(os.path.join(data_path, "param_{}/normals_0.npy").format(shape, shape)):
            seen_shapes+=1
            full_mesh = trimesh.load(os.path.join(data_path, "{}.ply").format(shape,shape))
            data_path = os.path.join(data_path,"param_{}").format(shape,shape)

            print('full mesh face size: ',np.quantile(full_mesh.area_faces, 0.1),np.quantile(full_mesh.area_faces, 0.9))
            for mesh_number in range(n_patches):
                print('shape {} patch number {} seen shapes {}/{}'.format(shape, mesh_number, seen_shapes, len(shapes)))

                cut_mesh = trimesh.load(os.path.join(data_path, 'cut_patch_{}.ply').format(mesh_number), process=False)
                parametrization = np.load(os.path.join(data_path, 'param_{}.npy').format(mesh_number))


                parametrization = utils.resize_parametrization(parametrization, cut_mesh)

                boundary = np.load(os.path.join(data_path, 'boundary_{}.npy').format(mesh_number))
                normals = np.load(os.path.join(data_path, 'normals_{}.npy').format(mesh_number))
                mean_curvature = np.load(os.path.join(data_path, 'mean_curvature_{}.npy').format(mesh_number))

                curvature = np.load(os.path.join(data_path, 'pc_{}.npy').format(mesh_number))
                mesh_2D_points = np.concatenate([parametrization, np.zeros([parametrization.shape[0], 1])], axis = 1)
                mesh_2D = trimesh.Trimesh(mesh_2D_points, cut_mesh.faces)
                n_points = len(parametrization)
                non_boundary = list(set(range(n_points)) - set(boundary))
                BATCH_SIZE = n_points
                N_BOUNDARY = len(boundary)
                boundary, boundary_normal = utils.make_boundary(boundary, parametrization)

                session, ops = init_graph(curvature[:, :3], parametrization,  cut_mesh.vertices, cut_mesh.faces, boundary,non_boundary, boundary_normal, normals, mean_curvature)
                optimize(session, ops, boundary[:, 0], mesh_2D, save_path.format(shape))
