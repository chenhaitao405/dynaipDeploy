import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray,Float32MultiArray

import articulate as art
import torch
import os
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from model.model_new import Poser
import utils.config as cfg
import random
from utils.data import fill_dip_nan, normalize_imu


def normalize_imu(acc, ori):
    r"""
    normalize imu w.r.t the root sensor
    """
    acc = acc.view(-1, 6, 3)
    ori = ori.view(-1, 6, 3, 3)
    acc = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(ori[:, 0])
    ori = torch.cat((ori[:, :1], ori[:, :1].transpose(2, 3).matmul(ori[:, 1:])), dim=1)
    data = torch.cat((ori.view(-1, 6, 9), acc), dim=-1)
    return data


# imu原始数据
acc = torch.stack(acc_list, dim=0) # tensor，shape:(6,3) 加速度
ori = torch.stack(ori_list, dim=0) # tensor，shape:(6,3,3) 旋转矩阵


Ry = torch.tensor([
    [0, 0, 1],  # 新x轴在原坐标系中的坐标 (z_old)
    [1, 0, 0],  # 新y轴在原坐标系中的坐标 (x_old)
    [0, 1, 0]   # 新z轴在原坐标系中的坐标 (y_old)
    ],dtype=ori.dtype,device=device)


# 对每个旋转矩阵应用变换：R_new = Ry @ R_original
ori = torch.einsum('ij,njk->nik', Ry, ori)

# 对旋转矩阵做投影后补偿z轴加速度，以抵消重力加速度
acc = ori.bmm(acc.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 1.])

acc *= 9.8  # 修改加速度量纲

single_imu = normalize_imu(acc, ori)  #1,6,12 对输入数据做归一化