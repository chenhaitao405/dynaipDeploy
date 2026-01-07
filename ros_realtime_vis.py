import numpy as np
import torch
import smplx
import open3d as o3d
import time
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import os
import articulate as art


# Load SMPL model
model_type = "smpl"
model_path = os.path.expanduser('smpl_models/smpl_male.pkl')
model = smplx.create(model_path, model_type="smpl")
body_model = art.ParametricModel('smpl_models/smpl_male.pkl', device='cpu') 

# Parameters for SMPL model
betas = torch.zeros([1, model.num_betas], dtype=torch.float32)

# ROS initialization
rospy.init_node('smpl_visualizer', anonymous=True)
received_data = []

def callback(msg):
    global received_data
    received_data = msg.data

data_sub = rospy.Subscriber('/inference_data', Float32MultiArray, callback)

# Function to get data for current frame
def get_current_data():
    if len(received_data) != 72*3:
        return None, None  # 数据不完整
    
    vis_data = np.array(received_data).reshape(1, 24, 3,-1)
    body_pose = torch.from_numpy(vis_data).to(torch.float32)
    # global_data = np.zeros((1, 1, 3))  # 固定腰部
    global_data = np.array(received_data)[:9].reshape(1, 1, -1)  # 不固定腰部

    global_orient = torch.from_numpy(global_data).to(torch.float32)
    return body_pose, global_orient

# Create initial meshdet
# 等待第一帧数据到来
while len(received_data) != 216:
    time.sleep(0.1)
glb_full_pose_smpl, global_orient = get_current_data()

local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl).view(glb_full_pose_smpl.shape[0], 24, 3, 3)
_, _, verts = body_model.forward_kinematics(local_full_pose_smpl, calc_mesh=True)  # verts： torch.Size([3389, 6890, 3])

initial_vertices = verts[0].numpy()
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(initial_vertices)
mesh.triangles = o3d.utility.Vector3iVector(model.faces)
mesh.compute_vertex_normals()

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)

# Animation callback
def animation_callback(vis):
    glb_full_pose_smpl, global_orient = get_current_data()
    if glb_full_pose_smpl is None:
        time.sleep(0.1)  # 等待数据到来
        return True
    
    local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl).view(glb_full_pose_smpl.shape[0], 24, 3, 3)
    _, _, verts = body_model.forward_kinematics(local_full_pose_smpl, calc_mesh=True)  # verts： torch.Size([3389, 6890, 3])
    vertices = verts[0].detach().cpu().numpy()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    return True

# Register the animation callback
vis.register_animation_callback(animation_callback)

# Run the visualizer
vis.run()
vis.destroy_window()

