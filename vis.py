import articulate as art
import torch
import os
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from model.model import Poser
import utils.config as cfg
import random

body_model = art.ParametricModel(cfg.smpl_m, device='cpu')  
test_folder = os.path.join(cfg.work_dir, 'test')
test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
            for foldername, _, filenames in os.walk(test_folder)
            for filename in filenames if filename.endswith('.pt')]
device = torch.device("cpu")
 
net = Poser().to(device)
net.load_state_dict(torch.load(cfg.weight_s, map_location='cpu')) # DynaIP* in paper
net.eval()

f = os.path.join(test_folder,test_files[7])
print(f)
data = torch.load(f)
# prepare imu and initial states of the first frame
imu = data['imu']['imu'].to(device) # shape: batch,6,12
for i in range(1):
    # print(imu[(i+1)*40])
    print(f'frame{i}', imu[(i)][0,:9].reshape((3,3)))

# if 'dip' in f:
#     vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
#     local_gt_smpl = data['joint']['full smpl pose']
#     glb_gt_smpl, _ = body_model.forward_kinematics(local_gt_smpl, calc_mesh=False) 
# else:
#     vel_mask = torch.tensor([0, 6, 14, 10, 21, 17])
#     glb_gt_xsens = data['joint']['full xsens pose']

# v_init = data['joint']['velocity'][:1, vel_mask].float().to(device)
# pose = data['joint']['orientation']
# pose = pose.view(pose.shape[0], -1, 6)[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]]
# p_init = pose[:1].to(device)

# glb_full_pose_xsens, glb_full_pose_smpl = net.predict(imu, v_init, p_init)  # imu shape;(batch,6,12)

# # glb_full_pose_smpl = torch.load('inference_result_1751524817.pt')
# local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl).view(glb_full_pose_smpl.shape[0], 24, 3, 3)
# _, _, verts = body_model.forward_kinematics(local_full_pose_smpl, calc_mesh=True)  # verts： torch.Size([3389, 6890, 3])

# if 'dip' in f:
#     _, _, verts_gt = body_model.forward_kinematics(local_gt_smpl, calc_mesh=True)
# else:
#     glb_gt_smpl = net._glb_mat_xsens_to_glb_mat_smpl(glb_gt_xsens) # use smpl model to visulize so transform xsens to smpl
#     local_gt_smpl = body_model.inverse_kinematics_R(glb_gt_smpl).view(glb_gt_smpl.shape[0], 24, 3, 3)
#     _, _, verts_gt = body_model.forward_kinematics(local_gt_smpl, calc_mesh=True)

# verts_gt += torch.tensor([1.0, 0, 0], device=verts_gt.device)

# body_mesh = Meshes(
#             verts.numpy(),
#             body_model.face,
#             is_selectable=False,
#             gui_affine=False,
#             name="Predicted Body Mesh",
#         )

# gt_mesh = Meshes(
#             verts_gt.numpy(),
#             body_model.face,
#             is_selectable=False,
#             gui_affine=False,
#             name="Ground Truth Body Mesh",
#         )

# v = Viewer()
# v.scene.add(body_mesh)
# v.scene.add(gt_mesh)
# v.run()

