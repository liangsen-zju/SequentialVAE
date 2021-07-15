"""
This utils for geometry transform and carera projection
"""
import torch
import torch.nn as nn
import numpy as np

import render_util

def euler2rot(euler):
    """Convert euler to rotation matrix
    params:
        euler: torch.Tensor, (nbatch, 3)
    return:
        rotation matrix, torch.Tensor, (nbtach, 3, 3)
    """
    batch_size = euler.shape[0]
    theta = euler[:, 0].reshape(-1, 1, 1)
    phi = euler[:, 1].reshape(-1, 1, 1)
    psi = euler[:, 2].reshape(-1, 1, 1)
    ones = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler.device)

    rot_x = torch.cat((   # (nbatch, 3, 3)
        torch.cat((ones, zero, zero), 1),                 #(nbatch, 3, 1)
        torch.cat((zero, theta.cos(), theta.sin()), 1),   #(nbatch, 3, 1)
        torch.cat((zero, -theta.sin(), theta.cos()), 1),  #(nbatch, 3, 1)
    ), 2)
    
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, ones, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),   
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, ones), 1)
    ), 2)

    rot = torch.bmm(rot_x, torch.bmm(rot_y, rot_z))        # (nbatch, 3, 3)
    
    return rot


def geo_rotation_transfrom(geo, rot, trans=None):
    """ rotation geometry and trans
    params:
        geo: geometry, torch.Tensor, (nbatch, N, 3)
        rot: rotation matrix, torch.Tensor, (nbatch, 3, 3)
        trans: transform matrix, torch.Tensor, (nbtach, 3)
    return:
        geometry, torch.Tensor, (nbatch, N, 3)
    """
    # rotation
    geo = torch.bmm(rot, geo.permute(0, 2, 1))   # (nbatch, 3, N)

    # transform
    if trans is not None:
        geo += trans.view(-1, 3, 1)

    geo = geo.permute(0, 2, 1) 
    return geo


def geo_projection(geo, param_camera):
    """project geo to 2D plane
    params:
        geo, geometry, torch.Tensor, (nbatch, N, 3)
        param_camera: torch.Tensor, (nbatch, 4)
    return:
        geometry, torch.Tensor, (nbatch, N, 3)
    """

    fx, fy = param_camera[:, 0], param_camera[:, 1]     # (nbatch, )
    cx, cy = param_camera[:, 2], param_camera[:, 3]     # (nbatch, )
 
    X, Y, Z = geo[:, :, 0], geo[:, :, 1], geo[:, :, 2]  # (nbatch, N)

    fxX = fx[:, None] * X                               # (nbatch, N)    
    fyY = fy[:, None] * Y                               # (nbatch, N)

    proj_x = -fxX / Z + cx[:, None]                     # (nbatch, N)
    proj_y =  fyY / Z + cy[:, None]

    geo_pro = torch.cat([proj_x, proj_y, Z], 2)

    return geo_pro


def landmark_projection(geo, ldmk_info, ldmk_num=68):
    """Landmark Projection

    params:
        geo, geometry, torch.Tensor, (nbatch, N, 3)
        ldmk_info, torch.Tensor, (68,)
        ldmk_num, int, the number of landmarks, default=68
    
    return:
        projection landmark, (nbatch, 68, 2)
    """
    n_points = geo.shape[0]
    
    rot_tri_normal = compute_tri_normal(geo_rot, self.tris)
    rot_ver_normal = torch.index_select(rot_tri_normal, 1, self.vert_tris)


    is_visible = -torch.bmm(rot_ver_normal.reshape(-1, 1, 3), geo_rot.reshape(-1, 3, 1))
    is_visible = is_visible.reshape(-1, n_points)


    ldmk_idx = render_util.update_contour(ldmk_info, is_visible, ldmk_num)
    print(f"DEBUG: landmark projection, ldmk_num={ldmk_num}, update contour, ldmk_idx len={ldmk_idx.shape}, should be (Nbatch * 68)")

    ldmk_pro = torch.index_select(geo.reshape(-1, 3), 0, ldmk_idx)           # (len(ldmk_idx), 3)
    ldmk_pro = ldmk_pro[:, :2].reshape(-1, ldmk_num, 2)                      # (nbatch, ldmk_num, 2)   

    return ldmk_pro

    