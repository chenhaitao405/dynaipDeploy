import articulate as art
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import *


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=False)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        x = self.dropout(F.relu(self.linear1(x)))
        x,state = self.rnn(x.unsqueeze(1), state)
        x = self.linear2(x)
        x = x.squeeze(-2)
        return x,state


class RNNWithInit(RNN):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_init: int, n_rnn_layer: int
                 , bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        self.n_rnn_layer = n_rnn_layer
        self.n_hidden = n_hidden
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_init, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )

    def forward(self, x, state=None):

        h, c = state
        output, (h_out, c_out) = super().forward(x, (h, c))
        return output, (h_out, c_out)


class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super(SubPoser, self).__init__()
        self.extra_dim = extra_dim
        self.rnn1 = RNNWithInit(n_init=v_output, n_input=n_input - extra_dim,
                                n_hidden=n_hidden, n_output=v_output,
                                n_rnn_layer=num_layer, dropout=dropout)
        self.rnn2 = RNNWithInit(n_init=p_output, n_input=n_input + v_output,
                                n_hidden=n_hidden, n_output=p_output,
                                n_rnn_layer=num_layer, dropout=dropout)

    def forward(self, x, rnn1_state=None, rnn2_state=None):
        # 状态解析：每个SubPoser有两个RNN状态

#TODO:初步定位位置
        if self.extra_dim != 0:
            x_v = x[:, :-self.extra_dim]
            v, rnn1_state_out = self.rnn1((x_v), rnn1_state)
        else:
            v, rnn1_state_out = self.rnn1((x), rnn1_state)

        x_concat = torch.cat((x, v), dim=-1)
        p, rnn2_state_out = self.rnn2((x_concat), rnn2_state)
        # 返回该SubPoser的两个状态
        return v, p, rnn1_state_out, rnn2_state_out


class Poser_withoutInit(nn.Module):
    def __init__(self):
        super(Poser_withoutInit, self).__init__()
        n_hidden = 200
        num_layer = 2
        dropout = 0.2
        n_glb = 6

        self.posers = nn.ModuleList([SubPoser(n_input=36 + n_glb, v_output=6, p_output=24,
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb),
                                     SubPoser(n_input=48 + n_glb, v_output=12, p_output=12,
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb),
                                     SubPoser(n_input=24 + n_glb, v_output=6, p_output=30,
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=dropout,
                                              extra_dim=n_glb)])

        self.glb = RNN(n_input=72, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout)
        self.sensor_names = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
        self.v_names = ['Root', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
        self.p_names = ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'L3',
                        'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftUpperArm',
                        'RightUpperArm']
        self.generate_indices_list()
        print("Total Parameters:", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def find_indices(self, elements, lst):
        indices = []
        for element in elements:
            if element in lst:
                indices.append(lst.index(element))
        return indices

    def generate_indices_list(self):
        posers_config = [
            {'sensor': ['Root', 'LeftForeArm', 'RightForeArm'], 'velocity': ['LeftHand', 'RightHand'],
             'pose': ['LeftShoulder', 'LeftUpperArm', 'RightShoulder', 'RightUpperArm']},
            {'sensor': ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head'],
             'velocity': ['Root', 'LeftFoot', 'RightFoot', 'Head'],
             'pose': ['LeftUpperLeg', 'RightUpperLeg']},
            {'sensor': ['Root', 'Head'], 'velocity': ['Root', 'Head'],
             'pose': ['L5', 'L3', 'T12', 'T8', 'Neck']},
        ]
        self.indices = []
        for i in range(len(self.posers)):
            temp = {'sensor_indices': self.find_indices(posers_config[i]['sensor'], self.sensor_names),
                    'v_indices': self.find_indices(posers_config[i]['velocity'], self.v_names),
                    'p_indices': self.find_indices(posers_config[i]['pose'], self.p_names)}
            self.indices.append(temp)

    def forward(self, x, glb_states=None,poser_states=None):

        """
        glb_states[1,2,32]: 全局模块状态 (h_glb, c_glb)
        poser_states[2,12,200]: 全局模块状态 (h_glb, c_glb)
        """

        # 将输入状态从 [1,2,32] 拆分为两个独立状态
        h0 = glb_states[:, 0, :].unsqueeze(1)  # [1,32] -> [1,1,32]
        c0 = glb_states[:, 1, :].unsqueeze(1)  # [1,32] -> [1,1,32]

        # 使用拆分的状态执行模块
        s_glb, (hn, cn) = self.glb(
            x.flatten(1),
            (h0, c0)  # 保持元组输入（内部模块需兼容）
        )

        # 合并状态 - 使用最基础的操作
        hn_flat = hn.squeeze(-2)  # [1,1,32] -> [1,32]
        cn_flat = cn.squeeze(-2)  # [1,1,32] -> [1,32]
        new_glb_states = torch.stack([hn_flat, cn_flat], dim=1)  # [1,2,32]
        v_out, p_out = [], []

        #12个tensor单独输入

        # 处理每个SubPoser模块
        for i, poser in enumerate(self.posers):
            # 获取当前SubPoser的状态索引（1,2,3）
            sensor_indices = self.indices[i]['sensor_indices']
            sensor = torch.cat([x[:, idx] for idx in sensor_indices], dim=1).flatten(1)
            si = torch.cat((sensor, s_glb), dim=-1)
            #TODO:将sensor索引提到模型外处理

            # 创建临时变量避免复杂表达式
            state_index_0 = 4 * i
            state_index_1 = 4 * i + 1
            state_index_2 = 4 * i + 2
            state_index_3 = 4 * i + 3

            rnn1_state_h = poser_states[:, state_index_0:state_index_0 + 1, :]
            rnn1_state_c = poser_states[:, state_index_1:state_index_1 + 1, :]
            rnn2_state_h = poser_states[:, state_index_2:state_index_2 + 1, :]
            rnn2_state_c = poser_states[:, state_index_3:state_index_3 + 1, :]

            v, p, (rnn1_state_out_h,rnn1_state_out_c),  (rnn2_state_out_h,rnn2_state_out_c) = poser(si, (rnn1_state_h,rnn1_state_c), (rnn2_state_h,rnn2_state_c))
            v_out.append(v)
            p_out.append(p)
            #  更新隐状态（h、c）

        v_out = torch.cat(v_out, dim=1)
        p_out = torch.cat(p_out, dim=1)

        return  v_out,p_out,new_glb_states

    def _reduced_glb_6d_to_full_glb_mat_xsens(self, glb_reduced_pose, orientation):
        joint_set = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
        sensor_set = [0, 20, 16, 6, 13, 9]
        ignored = [10, 14, 17, 18, 21, 22]
        parent = [9, 13, 16, 16, 20, 20]
        root_rotation = orientation[:, 0].view(-1, 3, 3)
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, len(joint_set), 3, 3)
        # back to glb coordinate
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        orientation[:, 1:] = root_rotation.unsqueeze(1).matmul(orientation[:, 1:])
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 23, 1, 1)
        global_full_pose[:, joint_set] = glb_reduced_pose
        global_full_pose[:, sensor_set] = orientation
        global_full_pose[:, ignored] = global_full_pose[:, parent]
        return global_full_pose

    def _glb_mat_xsens_to_glb_mat_smpl(self, glb_full_pose_xsens):
        glb_full_pose_smpl = torch.eye(3).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
        for idx, i in enumerate(indices):
            glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]
        return glb_full_pose_smpl

    @torch.no_grad()
    def predict(self, x, states=None):
        self.eval()
        v_partition, p_partition, states_out = self.forward(x, states)
        pose, v = p_partition.cpu(), v_partition.cpu()
        pose = pose.view(-1, 11, 6)[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]]
        orientation = x[:, :, :9].view(-1, 6, 3, 3).cpu()
        glb_full_pose_xsens = self._reduced_glb_6d_to_full_glb_mat_xsens(pose, orientation)
        glb_full_pose_smpl = self._glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens)
        return_v = False
        if return_v:
            v = v.view(-1, 8, 3)[:, [2, 5, 0, 1, 3, 4]]
            v = v.bmm(orientation[:, 0].transpose(1, 2))
            v[:, 1:, 1] = v[:, 1:, 1] + v[:, :1, 1]
            return glb_full_pose_xsens, glb_full_pose_smpl, v, states_out
        else:
            return glb_full_pose_xsens, glb_full_pose_smpl, states_out