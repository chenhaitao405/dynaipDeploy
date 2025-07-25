import articulate as art
import torch
import os
import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from model.NCNN_model import NCNNPoserWithoutInit
import utils.config as cfg
from tqdm import tqdm
import time


def load_body_model():
    """加载身体模型"""
    return art.ParametricModel(cfg.smpl_m, device='cpu')


def get_test_files():
    """获取测试文件列表"""
    test_folder = os.path.join(cfg.work_dir, 'test')
    test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
                  for foldername, _, filenames in os.walk(test_folder)
                  for filename in filenames if filename.endswith('.pt')]
    return test_folder, test_files


def load_ncnn_model():
    """加载NCNN模型"""
    param_path = "./weights/model_withoutInit.ncnn.param"
    bin_path = "./weights/model_withoutInit.ncnn.bin"
    ncnn_model = NCNNPoserWithoutInit(param_path, bin_path)
    return ncnn_model


def load_data(file_path):
    """加载和预处理数据"""
    print(f"Loading data from: {file_path}")
    data = torch.load(file_path)

    # 准备IMU数据
    imu = data['imu']['imu'].cuda()

    # 根据数据类型设置参数
    if 'dip' in file_path:
        vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
        local_gt_smpl = data['joint']['full smpl pose']
        gt_data = {'type': 'dip', 'pose': local_gt_smpl}
    else:
        vel_mask = torch.tensor([0, 6, 14, 10, 21, 17])
        glb_gt_xsens = data['joint']['full xsens pose']
        gt_data = {'type': 'xsens', 'pose': glb_gt_xsens}

    # 分割IMU数据（用于逐帧处理）
    split_imu = [t.unsqueeze(0) for t in imu.unbind(0)]

    return split_imu, gt_data


def initialize_states():
    """初始化模型状态"""
    # 加载预保存的状态
    states = torch.load("states7.pth")
    glb_states = torch.stack(states[0], dim=0)

    # 处理poser状态
    all_tensors = []
    for tup in states[1:]:
        all_tensors.extend(tup)
    poser_states = torch.stack(all_tensors, dim=0)

    return glb_states, poser_states


def run_ncnn_inference(ncnn_model, split_imu, glb_states, poser_states):
    """运行NCNN模型推理"""
    output_list = []
    inference_time = 0

    # 转换状态为numpy数组
    glb_states_ncnn = glb_states.clone().cpu().numpy()
    poser_states_ncnn = poser_states.clone().cpu().numpy()

    print(f"\n开始NCNN推理，共 {len(split_imu) - 1} 帧...")

    # 使用tqdm显示进度条，跳过第一帧
    for single_imu in tqdm(split_imu[1:], desc="NCNN推理进度"):
        # 转换输入为numpy
        imu_numpy = single_imu.cpu().numpy()

        start_time = time.time()
        glb_full_pose_xsens, glb_full_pose_smpl_single, glb_states_ncnn_torch, poser_states_ncnn_torch = \
            ncnn_model.predict(imu_numpy, glb_states_ncnn, poser_states_ncnn)
        inference_time += time.time() - start_time

        # 更新状态为numpy格式以供下次使用
        glb_states_ncnn = glb_states_ncnn_torch.numpy()
        poser_states_ncnn = poser_states_ncnn_torch.numpy()

        output_list.append(glb_full_pose_smpl_single)

    # 合并输出结果
    glb_full_pose_smpl_combined = torch.cat(output_list, dim=0)

    print(f"\n推理完成!")
    print(f"总推理时间: {inference_time:.3f}秒")
    print(f"平均每帧时间: {inference_time / len(split_imu[1:]):.4f}秒")
    print(f"输出形状: {glb_full_pose_smpl_combined.shape}")

    return glb_full_pose_smpl_combined


def process_poses(body_model, glb_full_pose_smpl_combined):
    """处理姿态数据，生成网格顶点"""
    # 转换为局部姿态
    local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl_combined).view(
        glb_full_pose_smpl_combined.shape[0], 24, 3, 3).float()

    # 生成预测的网格顶点
    _, _, verts = body_model.forward_kinematics(local_full_pose_smpl, calc_mesh=True)

    return verts, local_full_pose_smpl


def process_ground_truth(body_model, gt_data):
    """处理真实标签数据"""
    if gt_data['type'] == 'dip':
        local_gt_smpl = gt_data['pose']
        _, _, verts_gt = body_model.forward_kinematics(local_gt_smpl, calc_mesh=True)
    else:
        # 对于xsens数据，需要额外的转换步骤
        # 这里简化处理，实际使用时可能需要适配
        print("注意：xsens数据的ground truth可能需要额外的坐标转换")
        verts_gt = None

    return verts_gt


def create_visualization_meshes(verts_ncnn, verts_gt, body_model):
    """创建可视化网格"""
    meshes = []

    # NCNN预测结果
    ncnn_mesh = Meshes(
        verts_ncnn.numpy(),
        body_model.face,
        is_selectable=False,
        gui_affine=False,
        name="NCNN Prediction",
        color=(0.2, 0.8, 0.2, 1.0)  # 绿色
    )
    meshes.append(ncnn_mesh)

    # 如果有真实标签，添加到可视化
    if verts_gt is not None:
        # 确保帧数相同
        min_frames = min(verts_ncnn.shape[0], verts_gt.shape[0])
        verts_gt = verts_gt[:min_frames]

        # 真实标签（右侧）
        verts_gt_shifted = verts_gt + torch.tensor([2.0, 0, 0], device=verts_gt.device)
        gt_mesh = Meshes(
            verts_gt_shifted.numpy(),
            body_model.face,
            is_selectable=False,
            gui_affine=False,
            name="Ground Truth",
            color=(0.2, 0.2, 0.8, 1.0)  # 蓝色
        )
        meshes.append(gt_mesh)

        # 差异热图
        vertex_diff = torch.norm(verts_ncnn[:min_frames] - verts_gt, dim=-1)
        diff_colors = plt_to_color(vertex_diff.numpy())

        verts_diff_shifted = verts_ncnn[:min_frames] + torch.tensor([1.0, -2.0, 0], device=verts_ncnn.device)
        diff_mesh = Meshes(
            verts_diff_shifted.numpy(),
            body_model.face,
            is_selectable=False,
            gui_affine=False,
            name="NCNN-GT Difference",
            vertex_colors=diff_colors
        )
        meshes.append(diff_mesh)

    return meshes


def plt_to_color(values, cmap='hot', vmin=None, vmax=None):
    """将数值映射到颜色，包含alpha通道"""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    # 归一化
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # 展平值并映射到颜色，包含alpha通道
    colors = mapper.to_rgba(values.flatten())  # RGBA格式，包含alpha通道
    colors = colors.reshape(values.shape + (4,))  # 重塑为(N, 4)形状

    return colors.astype(np.float32)


def visualize_results(meshes):
    """可视化结果"""
    v = Viewer()

    for mesh in meshes:
        v.scene.add(mesh)

    # 调整相机位置
    if len(meshes) > 1:
        v.scene.camera.position = np.array([1.0, 1.5, 4.0])
        v.scene.camera.target = np.array([1.0, 0.0, 0.0])
    else:
        v.scene.camera.position = np.array([0.0, 1.5, 3.0])
        v.scene.camera.target = np.array([0.0, 0.0, 0.0])

    v.run()


def main():
    """主函数"""
    print("=== NCNN模型推理和可视化 ===\n")

    # 1. 加载模型和数据
    print("1. 加载身体模型...")
    body_model = load_body_model()

    print("2. 获取测试文件...")
    test_folder, test_files = get_test_files()
    print(f"   找到 {len(test_files)} 个测试文件")

    print("3. 加载NCNN模型...")
    ncnn_model = load_ncnn_model()

    # 2. 选择测试文件
    if len(test_files) == 0:
        print("错误：未找到测试文件！")
        return

    print("\n可用的测试文件：")
    for i, file in enumerate(test_files):
        print(f"   [{i}] {file}")

    # 这里默认使用第一个文件，可以根据需要修改
    file_idx = 1
    file_path = os.path.join(test_folder, test_files[file_idx])
    print(f"\n4. 加载测试数据：{test_files[file_idx]}")
    split_imu, gt_data = load_data(file_path)

    # 3. 初始化状态
    print("5. 初始化模型状态...")
    glb_states, poser_states = initialize_states()

    # 4. 运行NCNN推理
    print("6. 运行NCNN推理...")
    ncnn_result = run_ncnn_inference(ncnn_model, split_imu, glb_states, poser_states)

    # 5. 处理姿态数据
    print("\n7. 处理姿态数据...")
    verts_ncnn, local_pose_ncnn = process_poses(body_model, ncnn_result)
    print(f"   生成顶点数据形状: {verts_ncnn.shape}")

    # 6. 处理真实标签（如果可用）
    print("8. 处理真实标签数据...")
    verts_gt = process_ground_truth(body_model, gt_data)

    # 7. 创建可视化网格
    print("9. 创建可视化网格...")
    meshes = create_visualization_meshes(verts_ncnn, verts_gt, body_model)

    # 8. 可视化结果
    print("\n10. 启动可视化窗口...")
    print("绿色: NCNN预测结果")
    if verts_gt is not None:
        print("蓝色: 真实标签")
        print("底部热图: NCNN与真实标签的差异")
    print("\n使用鼠标控制视角，按ESC退出")

    visualize_results(meshes)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()