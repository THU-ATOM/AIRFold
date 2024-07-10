import numpy as np
import torch
import math

def indexFromCoord(coord,coord_list):
    try:
        if coord in coord_list:
            index = coord_list.index(coord)
            return index
    except Exception as e:
        print(e, "coord is not in coord_list,it is error")

def compute_point_distance(point_1, point_2):

    vector = np.array(point_1) - np.array(point_2)
    point_dist = np.linalg.norm(vector)

    return point_dist

def compute_point_lrf(point_1, point_2, point_3):

    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    point_3 = np.array(point_3)
    V1 = point_3 - point_2
    V2 = point_1 - point_2
    ex = V1 / np.linalg.norm(V1)
    ey_normalized = V2 - np.dot(V2, ex) * ex
    ey_normalized /= np.linalg.norm(ey_normalized)
    eznorm = np.cross(ex, ey_normalized)
    lrf = [ex.tolist(), ey_normalized.tolist(), eznorm.tolist()]

    return lrf

# atom level
def dist_list(point,coord_list):
    dist_list = []

    for th, _ in enumerate(coord_list):
        x_2 = pow(coord_list[th][0] - point[0], 2)
        y_2 = pow(coord_list[th][1] - point[1], 2)
        z_2 = pow(coord_list[th][2] - point[2], 2)
        dist_X2 = math.sqrt(x_2 + y_2 + z_2)
        dist_list.append(dist_X2)

    return dist_list

def min_max(index,point,coord_list):

        ctd_dist_list = dist_list(point,coord_list)
        mean_index_dist = np.mean(ctd_dist_list)

        min_dist  = min(ctd_dist_list)
        max_dist  = max(ctd_dist_list)
        min_index = ctd_dist_list.index(min_dist)
        max_index = ctd_dist_list.index(max_dist)
        min_coord = coord_list[min_index]  # distance from ctd min(cst)
        max_coord = coord_list[max_index]  # distance from ctd max(fct)

        coord_list.insert(index,point)

        fct_dist_list = dist_list(max_coord, coord_list)
        max_fct_dist  = max(fct_dist_list)
        max_fct_index = fct_dist_list.index(max_fct_dist)
        fct_coord     = coord_list[max_fct_index]


        if point != fct_coord:  # 判断最远的远点是否为自身，如果时自身换成最近的一个点
            point_dist     = compute_point_distance(point, fct_coord)
            triangular_lrf = compute_point_lrf(max_coord, point, fct_coord)
            triangular_lrf.append(max_dist)
            triangular_lrf.append(max_fct_dist)
            triangular_lrf.append(point_dist)
        else:
            max_min_dist = compute_point_distance(max_coord, min_coord)
            triangular_lrf = compute_point_lrf(max_coord, point, min_coord)
            triangular_lrf.append(max_dist)
            triangular_lrf.append(max_min_dist)
            triangular_lrf.append(min_dist)

        return [[mean_index_dist, min_coord, max_coord, fct_coord], triangular_lrf]

def tri_location(path_name,out_path=None, save_npz=False):    #  输入

    per_atom_feature_list = []
    per_atom = []
    coord_list = []

    with open(path_name) as f:
        for line in f.readlines():

            if line.strip() == "TER" or line.strip() == "END":
                break
            if line.strip() == "":
                continue
            if line.split()[0] != 'ATOM':
                continue

            atom_name = line[12:16]

            # if "H" in atom_name and "N" not in atom_name \
            #         and "C" not in atom_name \
            #         and "O" not in atom_name:
            #     continue
            if "CA" in atom_name:
                coord_list.append([round(float(line[30:38]), 4),
                                   round(float(line[38:46]), 4),
                                   round(float(line[46:54]), 4)])

    xl = len(coord_list)

    for th,_ in enumerate(coord_list):

        point=coord_list.pop(th)
        meandist_coord = min_max(th,point,coord_list)
        per_atom_feature_list.append(meandist_coord)

        assert xl == len(coord_list)

    for th,feature in enumerate(per_atom_feature_list):

        [[mean_index_dist, min_coord, max_coord, fct_coord],
         [ex, ey_normalized, eznorm, max_dist, max_min_fct_dist, last_dist]] = feature
        min_index = indexFromCoord(min_coord,coord_list)
        max_index = indexFromCoord(max_coord,coord_list)
        fct_index = indexFromCoord(fct_coord,coord_list)

        min_mean_dist = per_atom_feature_list[min_index][0][0]
        max_mean_dist = per_atom_feature_list[max_index][0][0]
        fct_mean_dist = per_atom_feature_list[fct_index][0][0]

        ex1,ex2,ex3 = np.array(ex)
        ey1,ey2,ey3 = np.array(ey_normalized)
        ez1,ez2,ez3 = np.array(eznorm)
        max_dist = np.array(max_dist)
        max_min_fct_dist = np.array(max_min_fct_dist)

        per_atom.append([ex1,ex2,ex3,ey1,ey2,ey3,ez1,ez2,ez3, max_dist, max_min_fct_dist, last_dist,
                         mean_index_dist, min_mean_dist, max_mean_dist, fct_mean_dist])

    lg_usr_npz=np.vstack(np.array(per_atom))  # lrf,
    print(lg_usr_npz)
    if save_npz:
        np.savez_compressed(out_path +".lgusr.npz", lg_usr_npz=lg_usr_npz)
    else:
        return lg_usr_npz

  

# USR
# ex1,ex2,ex3,ey1,ey2,ey3,ez1,ez2,ez3, 局部坐标系
# max_dist, max_min_fct_dist, last_dist

def rotationMatrix_to_EulerAngles(R):
    """
    Convert the rotation matrix into three Euler angles.
    """

    cy = math.sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2])
    singular = cy < 1e-6

    if not singular:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], cy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], cy)
        z = 0

    x = x * 180.0 / 3.141592653589793
    y = y * 180.0 / 3.141592653589793
    z = z * 180.0 / 3.141592653589793

    result = [np.sin(x),np.cos(x),np.sin(y),np.cos(y),np.sin(z),np.cos(z)]

    return result


def tri_location_D(cb_map,cb_coords):  # input is dist map

    nres = len(cb_map)
    cb_mapT = torch.Tensor(cb_map)
    dm, idx = torch.topk(cb_mapT, nres, dim=-1)
    d_mean = torch.mean(cb_mapT,dim=0)

    res_dict = {}
    for i in range(nres):
        res_dict[i] = []
        res_dict[i].append(dm[i,0])   # for i : the max dist

        max_index=idx[i,0]   # for i : the max dist index
        mm_dist = dm[max_index, 0]  # for i-max : the max's dist
        mm_index = idx[max_index, 0]

        if str(mm_index.numpy()) == str(i) :   # if max-max point online

            mm_dist = dm[max_index,1]
            mm_index = idx[max_index, 1]
        res_dict[i].append(mm_dist)   # second edge for tri

        last_dist = cb_map[mm_index,i]  # the last edge for tri
        # tlast_dist=torch.Tensor(float(last_dist))
        res_dict[i].append(last_dist)

        d_i_max_last_mean = [d_mean[i], d_mean[max_index], d_mean[mm_index]]

        res_dict[i] += d_i_max_last_mean

        triangular_lrf=compute_point_lrf(cb_coords[max_index],cb_coords[i],cb_coords[mm_index])

        tri_lrf=np.vstack(triangular_lrf)
        sc_rot=rotationMatrix_to_EulerAngles(tri_lrf)
        res_dict[i]+=sc_rot

    tri = np.array([np.vstack(res_dict[i]) for i in res_dict.keys()]).reshape(nres,12)
    tri[:,0:6]=tri[:, 0:6]/100
    return tri