#!/usr/bin/env python
"""
离线IMU数据处理脚本
基于原实时推理脚本修改，读取离线CSV数据进行人体姿态估计
"""
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float32MultiArray
import pandas as pd
import os
import glob

import articulate as art
import torch
from model.NCNN_model import NCNNPoserWithoutInit
import utils.config as cfg
from utils.data import fill_dip_nan, normalize_imu
import time


def quaternion_to_rotation_matrix(q_x, q_y, q_z, q_w):
    """
    将四元数转换为3x3旋转矩阵
    输入: q_x, q_y, q_z, q_w (标量在最后)
    输出: 3x3 旋转矩阵
    """
    # 归一化四元数
    norm = np.sqrt(q_x ** 2 + q_y ** 2 + q_z ** 2 + q_w ** 2)
    q_x, q_y, q_z, q_w = q_x / norm, q_y / norm, q_z / norm, q_w / norm

    # 计算旋转矩阵元素
    R = np.array([
        [1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x ** 2 + q_y ** 2)]
    ])
    return R


def load_imu_csv(csv_path):
    """
    加载单个IMU的CSV文件
    返回: DataFrame，包含时间戳、四元数和加速度
    """
    df = pd.read_csv(csv_path)
    return df


def load_all_imus_from_folder(folder_path):
    """
    从文件夹加载所有6个IMU的数据
    返回: 按时间对齐的IMU数据列表
    """
    imu_data_list = []
    for i in range(1, 7):
        csv_path = os.path.join(folder_path, f"imuData{i}.csv")
        if os.path.exists(csv_path):
            df = load_imu_csv(csv_path)
            imu_data_list.append(df)
        else:
            rospy.logerr(f"找不到文件: {csv_path}")
            return None
    return imu_data_list


def get_frame_data(imu_data_list, frame_idx):
    """
    获取指定帧的所有IMU数据，并转换为原始格式
    返回: (num_imus, 12) 的numpy数组，包含9维旋转矩阵 + 3维加速度
    """
    num_imus = 6
    data_per_imu = 12

    frame_data = np.zeros((num_imus, data_per_imu))

    for i, df in enumerate(imu_data_list):
        if frame_idx >= len(df):
            return None

        row = df.iloc[frame_idx]

        # 提取四元数并转换为旋转矩阵
        q_x, q_y, q_z, q_w = row['q_x'], row['q_y'], row['q_z'], row['q_w']
        R = quaternion_to_rotation_matrix(q_x, q_y, q_z, q_w)

        # 提取加速度
        acc = np.array([row['acc_x'], row['acc_y'], row['acc_z']])

        # 组合数据：旋转矩阵展平(9) + 加速度(3)
        frame_data[i, :9] = R.flatten()
        frame_data[i, 9:12] = acc

    return frame_data


def get_min_frame_count(imu_data_list):
    """获取所有IMU数据中的最小帧数"""
    return min(len(df) for df in imu_data_list)


class OfflineIMUDataProcessor:
    def __init__(self, data_root):
        rospy.init_node('offline_imu_processor', anonymous=True)

        # 发布推理结果
        self.pub = rospy.Publisher('inference_data', Float32MultiArray, queue_size=10)

        # 数据路径
        self.data_root = data_root
        self.tpose_folder = os.path.join(data_root, "T-POSE")
        self.action_folders = [
            os.path.join(data_root, "a1"),
            os.path.join(data_root, "a2")
        ]

        # IMU配置
        self.num_imus = 6
        self.data_per_imu = 12

        # 校准参数
        self.RMI = None
        self.RSB = None
        self.calib_imu = None
        self.calibrated = False

        self.temp_dir = 'temp'
        os.makedirs(self.temp_dir, exist_ok=True)

        rospy.loginfo("离线IMU数据处理器已初始化")

    def perform_tpose_calibration(self):
        """使用T-pose文件夹数据执行校准"""

        # RMS矩阵定义（与原代码相同）
        _RMS_ = torch.tensor([
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1.]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1.]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1.]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1.]],
            [[0, -1, 0], [0, 0, -1], [-1, 0, 0.]],
            [[0, -1, 0], [0, 0, -1], [-1, 0, 0.]],
        ])

        rospy.loginfo(f"从 {self.tpose_folder} 加载T-pose数据进行校准...")

        # 加载T-pose数据
        tpose_data_list = load_all_imus_from_folder(self.tpose_folder)
        if tpose_data_list is None:
            rospy.logerr("无法加载T-pose数据")
            return False

        # 使用第一帧进行校准（或可以取多帧平均）
        frame_idx = 0  # 可以修改为使用特定帧或平均值
        imu_data = get_frame_data(tpose_data_list, frame_idx)

        if imu_data is None:
            rospy.logerr("无法获取T-pose帧数据")
            return False

        self.calib_imu = imu_data.flatten()

        # 1. RMI校准
        RSI = torch.from_numpy(imu_data[:, :9].reshape(6, 3, -1)).float().transpose(1, 2)
        self.RMI = _RMS_.bmm(RSI)
        rospy.loginfo(f"RMI校准完成: \n{self.RMI}")

        # 2. T-pose校准 - 使用所有IMU
        RIS_list = []
        for i in range(self.num_imus):
            rotation_matrix = torch.from_numpy(imu_data[i][:9].reshape(3, 3)).float()
            RIS_list.append(rotation_matrix)

        RIS = torch.stack(RIS_list, dim=0)

        # 计算RSB
        self.RSB = self.RMI.bmm(RIS).transpose(1, 2).matmul(torch.eye(3))

        # 保存RSB
        torch.save(self.RSB, os.path.join(self.temp_dir, 'RSB.pt'))

        self.calibrated = True
        rospy.loginfo("T-pose校准完成！")
        rospy.loginfo(f"RSB shape: {self.RSB.shape}")

        return True

    def process_frame(self, imu_data, frame_idx):
        """
        处理单帧IMU数据
        imu_data: (num_imus, data_per_imu) 的numpy数组
        """
        global glb_states_ncnn, poser_states_ncnn, inference_count, total_inference_time
        global raw_imu, imu_datas, infer_result

        if not self.calibrated:
            rospy.logwarn("T-pose校准未完成，跳过数据处理")
            return

        try:
            raw_imu.append(imu_data.copy())

            # 提取加速度和方向数据
            acc_list, ori_list = [], []

            for i in range(self.num_imus):
                single_imu = imu_data[i]
                rotation_matrix = torch.from_numpy(single_imu[:9].reshape(3, 3)).float()
                acceleration = torch.from_numpy(single_imu[9:12]).float()

                acc_list.append(acceleration)
                ori_list.append(rotation_matrix)

            # 堆叠数据
            acc = torch.stack(acc_list, dim=0)  # (6, 3)
            RIS = torch.stack(ori_list, dim=0)  # (6, 3, 3)

            # 重力补偿
            gravity_compensation = torch.tensor([0., 0., 1.], dtype=acc.dtype)
            acc_calibrated = RIS.bmm(acc.unsqueeze(-1)).squeeze(-1) + gravity_compensation
            acc_calibrated *= 9.8

            # 应用T-pose校准
            RMB = self.RMI.bmm(RIS).bmm(self.RSB)

            # 转换到模型坐标系
            acc_model = self.RMI.matmul(acc_calibrated.unsqueeze(-1)).squeeze(-1)

            # 归一化IMU数据
            single_imu_normalized = normalize_imu(acc_model, RMB)

            imu_datas.append(single_imu_normalized)

            # NCNN神经网络预测
            imu_numpy = single_imu_normalized.cpu().numpy()

            start_time = time.time()

            glb_full_pose_xsens, glb_full_pose_smpl_single, glb_states_ncnn_torch, poser_states_ncnn_torch = \
                ncnn_model.predict(imu_numpy, glb_states_ncnn, poser_states_ncnn)

            inference_time = time.time() - start_time
            total_inference_time += inference_time
            inference_count += 1

            # 更新状态
            glb_states_ncnn = glb_states_ncnn_torch.numpy()
            poser_states_ncnn = poser_states_ncnn_torch.numpy()

            # 每100帧打印一次平均推理时间
            if inference_count % 100 == 0:
                avg_time = total_inference_time / inference_count
                rospy.loginfo(f"已处理 {inference_count} 帧, 平均推理时间: {avg_time:.4f}秒/帧")

            # 逆运动学计算
            if isinstance(glb_full_pose_smpl_single, np.ndarray):
                glb_full_pose_smpl_tensor = torch.from_numpy(glb_full_pose_smpl_single)
            else:
                glb_full_pose_smpl_tensor = glb_full_pose_smpl_single.cpu()

            local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl_tensor).view(
                glb_full_pose_smpl_tensor.shape[0], 24, 3, 3)
            infer_result.append(local_full_pose_smpl)

            # 转换并发布
            inference_data = glb_full_pose_smpl_tensor.view(-1).detach().cpu().numpy().astype(np.float32)

            msg = Float32MultiArray()
            msg.data = inference_data.tolist()
            self.pub.publish(msg)

        except Exception as e:
            rospy.logerr(f"处理IMU数据时出错 (帧 {frame_idx}): {str(e)}")
            import traceback
            rospy.logerr(f"详细错误信息: {traceback.format_exc()}")

    def process_action_folder(self, folder_path, action_name):
        """处理单个动作文件夹的所有数据"""
        global glb_states_ncnn, poser_states_ncnn, inference_count, total_inference_time
        global raw_imu, imu_datas, infer_result

        rospy.loginfo(f"\n{'=' * 50}")
        rospy.loginfo(f"开始处理: {action_name}")
        rospy.loginfo(f"文件夹路径: {folder_path}")

        # 重置状态
        glb_states_ncnn, poser_states_ncnn = initialize_states()
        raw_imu = []
        imu_datas = []
        infer_result = []
        inference_count = 0
        total_inference_time = 0.0

        # 加载数据
        imu_data_list = load_all_imus_from_folder(folder_path)
        if imu_data_list is None:
            rospy.logerr(f"无法加载 {action_name} 的数据")
            return

        # 获取帧数
        total_frames = get_min_frame_count(imu_data_list)
        rospy.loginfo(f"总帧数: {total_frames}")

        # 设置处理频率
        rate = rospy.Rate(60)  # 60Hz，可根据实际数据采样率调整

        # 逐帧处理
        for frame_idx in range(total_frames):
            if rospy.is_shutdown():
                break

            frame_data = get_frame_data(imu_data_list, frame_idx)
            if frame_data is not None:
                self.process_frame(frame_data, frame_idx)

            rate.sleep()

        # 保存该动作的结果
        self.save_action_results(action_name)

        rospy.loginfo(f"{action_name} 处理完成")

    def save_action_results(self, action_name):
        """保存单个动作的推理结果"""
        global infer_result, raw_imu, imu_datas, inference_count, total_inference_time

        if not infer_result:
            rospy.logwarn(f"{action_name}: 无推理结果需要保存")
            return

        try:
            # 创建输出目录
            output_dir = os.path.join('datasets', action_name.replace(' ', '_'))
            os.makedirs(output_dir, exist_ok=True)

            rospy.loginfo(f"正在保存 {action_name} 的推理结果...")

            infer_result_torch = torch.stack(infer_result, dim=0)
            imu_data_torch = torch.stack(imu_datas, dim=0)

            rospy.loginfo(f"推理结果形状: {infer_result_torch.shape}")

            # 保存数据
            raw_imu_array = np.array(raw_imu)
            raw_imu_array.astype(np.float32).tofile(os.path.join(output_dir, "raw_imu_data.bin"))
            infer_result_torch.numpy().astype(np.float32).tofile(os.path.join(output_dir, "infer_result.bin"))
            imu_data_torch.numpy().astype(np.float32).tofile(os.path.join(output_dir, "process_imu_data.bin"))

            # 打印统计信息
            if inference_count > 0:
                avg_time = total_inference_time / inference_count
                rospy.loginfo(f"总推理帧数: {inference_count}")
                rospy.loginfo(f"总推理时间: {total_inference_time:.3f}秒")
                rospy.loginfo(f"平均推理时间: {avg_time:.4f}秒/帧")

            rospy.loginfo(f"结果已保存到: {output_dir}")

        except Exception as e:
            rospy.logerr(f"保存 {action_name} 结果时出错: {str(e)}")

    def save_calibration_data(self):
        """保存校准数据"""
        if self.calibrated:
            os.makedirs('datasets', exist_ok=True)
            self.calib_imu.astype(np.float32).tofile("datasets/calib_imu.bin")
            self.RMI.numpy().astype(np.float32).tofile("datasets/RMI.bin")
            self.RSB.numpy().astype(np.float32).tofile("datasets/RSB.bin")
            rospy.loginfo("校准数据已保存")

    def run(self):
        """运行离线数据处理"""
        # 1. 执行T-pose校准
        if not self.perform_tpose_calibration():
            rospy.logerr("T-pose校准失败，退出")
            return

        # 保存校准数据
        self.save_calibration_data()

        # 2. 依次处理每个动作文件夹
        for i, folder in enumerate(self.action_folders):
            if os.path.exists(folder):
                action_name = f"动作数据{i + 1}"
                self.process_action_folder(folder, action_name)
            else:
                rospy.logwarn(f"文件夹不存在: {folder}")

        rospy.loginfo("\n所有动作数据处理完成！")


def initialize_states():
    """初始化模型状态"""
    states = torch.load("states7.pth")
    glb_states = torch.stack(states[0], dim=0)

    all_tensors = []
    for tup in states[1:]:
        all_tensors.extend(tup)
    poser_states = torch.stack(all_tensors, dim=0)

    return glb_states.cpu().numpy(), poser_states.cpu().numpy()


def load_ncnn_model():
    """加载NCNN模型"""
    param_path = "./weights/model_withoutInit.ncnn.param"
    bin_path = "./weights/model_withoutInit.ncnn.bin"
    ncnn_model = NCNNPoserWithoutInit(param_path, bin_path)
    return ncnn_model


if __name__ == '__main__':
    # 数据根目录
    DATA_ROOT = "./datasets/1205/t2"

    # 加载NCNN模型
    rospy.loginfo("正在加载NCNN模型...")
    ncnn_model = load_ncnn_model()
    rospy.loginfo("NCNN模型加载完成")

    # 初始化状态
    rospy.loginfo("正在初始化模型状态...")
    glb_states_ncnn, poser_states_ncnn = initialize_states()
    rospy.loginfo("模型状态初始化完成")

    # 加载身体模型
    body_model = art.ParametricModel(cfg.smpl_m, device='cpu')

    # 全局变量
    raw_imu = []
    imu_datas = []
    infer_result = []
    inference_count = 0
    total_inference_time = 0.0

    # 运行离线处理器
    try:
        processor = OfflineIMUDataProcessor(DATA_ROOT)
        processor.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("用户中断处理")