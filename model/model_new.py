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
        length = [_.shape[0] for _ in x]
        x = self.dropout(F.relu(self.linear1(pad_sequence(x))))
        packed_x = pack_padded_sequence(x, length, enforce_sorted=False)
        packed_out, state = self.rnn(packed_x, state)  # 返回状态
        x = self.linear2(pad_packed_sequence(packed_out)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)], state  # 返回输出和状态
   
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
        x, x_init = x
        if state is None:
            # 未提供状态时使用初始化网络
            nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
            h, c = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
            state = (h, c)
        # 使用传入的状态或初始化的状态
        return super(RNNWithInit, self).forward(x, state)

class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super(SubPoser, self).__init__()
        self.extra_dim = extra_dim
        self.rnn1 = RNNWithInit(n_init=v_output, n_input=n_input-extra_dim, 
                                n_hidden=n_hidden, n_output=v_output, 
                                n_rnn_layer=num_layer, dropout=dropout)
        self.rnn2 = RNNWithInit(n_init=p_output, n_input=n_input+v_output,
                                n_hidden=n_hidden, n_output=p_output, 
                                n_rnn_layer=num_layer, dropout=dropout)
                
    def forward(self, x, v_init, p_init, state1=None, state2=None):
        if self.extra_dim!=0:
            x_v = [_[:, :-self.extra_dim] for _ in x]
        else:
            x_v = x
            
        # 处理第一个RNN
        v, state1_out = self.rnn1((x_v, v_init), state1)
        
        # 准备第二个RNN的输入
        x_p = [torch.cat((xi, vi), dim=-1) for xi, vi in zip(x, v)]
        p, state2_out = self.rnn2((x_p, p_init), state2)
        
        return v, p, state1_out, state2_out  # 返回状态
    
    
class Poser(nn.Module):
    def __init__(self):
        super(Poser, self).__init__()
        n_hidden = 200
        num_layer = 2
        dropout = 0.2
        n_glb = 6
        
        self.posers = nn.ModuleList([SubPoser(n_input=36 + n_glb, v_output=6, p_output=24,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), 
                                    SubPoser(n_input=48 + n_glb, v_output=12, p_output=12,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), 
                                    SubPoser(n_input=24 + n_glb, v_output=6, p_output=30,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb)])        
        
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
            {'sensor': ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head'], 'velocity': ['Root', 'LeftFoot', 'RightFoot', 'Head'], 
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
        
    def forward(self, x, v_init, p_init, states=None):
        # 初始化状态容器
        if states is None:
            states = [None] * (len(self.posers) * 2 + 1)  # 3个SubPoser * 2 + 1个glb
            
        s_glb, glb_state = self.glb([_.flatten(1) for _ in x], states[-1])
        states[-1] = glb_state  # 更新全局状态
        
        v_out, p_out = [], []
        new_states = []
        
        for i in range(len(self.posers)):
            sensor = [_[:, self.indices[i]['sensor_indices']].flatten(1) for _ in x]
            si = [torch.cat((l, g), dim=-1) for l, g in zip(sensor, s_glb)]
            vi = v_init[:, self.indices[i]['v_indices']].flatten(1)
            pi = p_init[:, self.indices[i]['p_indices']].flatten(1)
            
            # 获取当前SubPoser的状态
            state_idx = i * 2
            state1 = states[state_idx]
            state2 = states[state_idx+1]
            
            v, p, state1_out, state2_out = self.posers[i](si, vi, pi, state1, state2)
            
            # 保存输出和状态
            v_out.append(v)
            p_out.append(p)
            new_states.extend([state1_out, state2_out])
        
        # 添加全局状态
        new_states.append(glb_state)
        
        v_out = [torch.cat(_, dim=-1) for _ in zip(*v_out)]
        p_out = [torch.cat(_, dim=-1) for _ in zip(*p_out)]    
        
        return v_out, p_out, new_states  # 返回所有状态


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
    def predict(self, x, v_init, p_init, states=None):
        self.eval()
        v_partition, p_partition, new_states = self.forward([x], v_init, p_init, states)
        pose, v = p_partition[0].cpu(), v_partition[0].cpu()
        pose = pose.view(-1, 11, 6)[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]] 
        orientation = x[:, :, :9].view(-1, 6, 3, 3).cpu()
        glb_full_pose_xsens = self._reduced_glb_6d_to_full_glb_mat_xsens(pose, orientation)
        glb_full_pose_smpl = self._glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens)
        return glb_full_pose_xsens, glb_full_pose_smpl, new_states  # 返回新状态