#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float32MultiArray

import articulate as art
import torch
import os
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from model.NCNN_model import NCNNPoserWithoutInit
import utils.config as cfg
import random
from utils.data import fill_dip_nan, normalize_imu
import time


class IMUDataProcessor:
    def __init__(self):
        rospy.init_node('imu_data_processor', anonymous=True)

        # 订阅 /imu_rot 话题
        rospy.Subscriber("/imu_rot", Float64MultiArray, self.imu_callback)
        self.pub = rospy.Publisher('inference_data', Float32MultiArray, queue_size=10)

        # 根据数据格式初始化变量
        self.num_imus = 6
        self.data_per_imu = 12  # 9维旋转矩阵 + 3维加速度

        # T-pose校准相关变量
        self.RMI = None  # IMU到模型坐标系的旋转矩阵
        self.RSB = None  # 传感器到身体的旋转矩阵
        self.calib_imu = None
        self.calibrated = False
        self.temp_dir = 'temp'
        os.makedirs(self.temp_dir, exist_ok=True)

        # 进行T-pose校准
        self.perform_tpose_calibration()

        rospy.loginfo("IMU 数据处理器已启动，等待数据...")
        rospy.on_shutdown(self.save_results)

    def perform_tpose_calibration(self):
        """执行T-pose校准（合并RMI和T-pose校准）"""

        _RMS_ = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1.]],  # z朝前，x朝左，y朝下
                              [[-1, 0, 0], [0, -1, 0], [0, 0, 1.]],
                              [[-1, 0, 0], [0, -1, 0], [0, 0, 1.]],
                              [[-1, 0, 0], [0, -1, 0], [0, 0, 1.]],
                              [[0, 1, 0], [0, 0, 1], [1, 0, 0.]],  # z朝上，x朝前，y朝左
                              [[0, 1, 0], [0, 0, 1], [1, 0, 0.]]])
        rospy.loginfo("开始T-pose校准流程...")

        # 提示用户准备
        input(
            '请站直并保持T-pose姿势，确保IMU 5（左手）方向正确：x = 前方, y = 左侧, z = 上方，然后按Enter键。校准将在3秒后开始...')
        rospy.sleep(3)

        rospy.loginfo("正在进行校准...")

        # 获取IMU数据
        imu_data = None

        def imu_callback(msg):
            nonlocal imu_data
            imu_data = msg

        imu_sub = rospy.Subscriber("/imu_rot", Float64MultiArray, imu_callback)

        # 等待数据
        rate = rospy.Rate(10)
        while imu_data is None and not rospy.is_shutdown():
            rate.sleep()

        imu_sub.unregister()

        if imu_data is not None:
            # 处理IMU数据
            raw_data = np.array(imu_data.data)[1:]

            imu_array = raw_data.reshape(self.num_imus, self.data_per_imu)
            self.calib_imu = raw_data
            # 1. RMI校准
            RSI = torch.from_numpy(imu_array[:,:9].reshape(6,3, -1)).float().transpose(1,2)
            self.RMI = _RMS_.bmm(RSI)
            rospy.loginfo(f"RMI校准完成: \n{self.RMI}")

            # 2. T-pose校准 - 使用所有IMU
            RIS_list = []
            for i in range(self.num_imus):
                rotation_matrix = torch.from_numpy(imu_array[i][:9].reshape(3, 3)).float()
                RIS_list.append(rotation_matrix)

            RIS = torch.stack(RIS_list, dim=0)

            # 计算RSB
            self.RSB = self.RMI.bmm(RIS).transpose(1, 2).matmul(torch.eye(3))

            # 保存RSB（如果需要）
            torch.save(self.RSB, os.path.join(self.temp_dir, 'RSB.pt'))

            self.calibrated = True
            rospy.loginfo("T-pose校准完成！")
            rospy.loginfo(f"RSB shape: {self.RSB.shape}")
        else:
            rospy.logerr("无法获取IMU数据进行校准")

    def imu_callback(self, msg):
        global glb_states_ncnn, poser_states_ncnn, inference_count, total_inference_time
        """处理接收到的 IMU 数据"""

        # 检查是否已完成校准
        if not self.calibrated:
            # 初始化计数器（如果尚未初始化）
            if not hasattr(self, 'uncalibrated_frame_count'):
                self.uncalibrated_frame_count = 0

            # 每200帧提示一次
            self.uncalibrated_frame_count += 1
            if self.uncalibrated_frame_count >= 200:
                rospy.logwarn("T-pose校准未完成，跳过数据处理")
                self.uncalibrated_frame_count = 0  # 重置计数器
            return

        try:
            # 1. 获取原始数据数组
            raw_data = np.array(msg.data)[1:]
            total_elements = len(raw_data)

            # 2. 检查数据长度是否合理
            expected_elements = self.num_imus * self.data_per_imu
            if total_elements != expected_elements:
                rospy.logwarn(f"数据长度异常: 收到 {total_elements} 元素, 期望 {expected_elements}")
                return

            # 3. 将一维数组转换为三维结构 (num_imus, data_per_imu)
            imu_data = raw_data.reshape(self.num_imus, self.data_per_imu)
            raw_imu.append(imu_data)

            # 4. 提取并打印每个 IMU 的数据
            rospy.loginfo("\n" + "=" * 50 + "\n收到新的 IMU 数据:")
            acc_list, ori_list = [], []

            for i in range(self.num_imus):
                imu_id = i + 1
                start_idx = i * self.data_per_imu
                end_idx = start_idx + self.data_per_imu

                # 提取单个 IMU 的数据
                single_imu = imu_data[i]

                # 分离旋转矩阵和加速度数据 - 保持在CPU
                rotation_matrix = torch.from_numpy(single_imu[:9].reshape(3, 3)).float()
                acceleration = torch.from_numpy(single_imu[9:12]).float()

                acc_list.append(acceleration)
                ori_list.append(rotation_matrix)

            # 5. 堆叠数据 - 保持在CPU
            acc = torch.stack(acc_list, dim=0)  # (6, 3)
            RIS = torch.stack(ori_list, dim=0)  # (6, 3, 3)

            # 重力补偿
            # 使用校准后的旋转矩阵投影加速度
            gravity_compensaton = torch.tensor([0., 0., 1.],
                                               dtype=acc.dtype)
            acc = -acc
            acc_calibrated = RIS.bmm(acc.unsqueeze(-1)).squeeze(-1) + gravity_compensaton

            acc_calibrated *= 9.8

            print(acc_calibrated[4].reshape(-1, 3))  # 打印第一个IMU的加速度

            # 8. 应用T-pose校准 - 保持在CPU
            # 计算校准后的旋转矩阵：RMB = RMI @ RIS @ RSB
            RMB = self.RMI.bmm(RIS).bmm(self.RSB)

            # 转换到模型坐标系
            acc_model = self.RMI.matmul(acc_calibrated.unsqueeze(-1)).squeeze(-1)

            # 11. 归一化IMU数据
            single_imu = normalize_imu(acc_model, RMB)  # (1, 6, 12)

            # 12. 设置打印选项并打印调试信息
            torch.set_printoptions(sci_mode=False, precision=4)

            imu_datas.append(single_imu)

            # 13. NCNN神经网络预测
            # 转换IMU数据为numpy格式
            imu_numpy = single_imu.cpu().numpy()

            # 记录推理时间
            start_time = time.time()

            # 使用NCNN模型进行推理
            glb_full_pose_xsens, glb_full_pose_smpl_single, glb_states_ncnn_torch, poser_states_ncnn_torch = \
                ncnn_model.predict(imu_numpy, glb_states_ncnn, poser_states_ncnn)

            # 更新推理时间统计
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            inference_count += 1

            # 更新状态为numpy格式以供下次使用
            glb_states_ncnn = glb_states_ncnn_torch.numpy()
            poser_states_ncnn = poser_states_ncnn_torch.numpy()

            # 每100帧打印一次平均推理时间
            if inference_count % 100 == 0:
                avg_time = total_inference_time / inference_count
                rospy.loginfo(f"平均推理时间: {avg_time:.4f}秒/帧")

            # 14. 逆运动学计算 - 在CPU上进行
            # 检查输出类型并相应处理
            if isinstance(glb_full_pose_smpl_single, np.ndarray):
                glb_full_pose_smpl_tensor = torch.from_numpy(glb_full_pose_smpl_single)
            else:
                # 如果已经是tensor，确保在CPU上
                glb_full_pose_smpl_tensor = glb_full_pose_smpl_single.cpu()

            local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl_tensor).view(
                glb_full_pose_smpl_tensor.shape[0], 24, 3, 3)
            infer_result.append(local_full_pose_smpl)

            body_rotations = glb_full_pose_smpl_tensor  # 形状: (23, 3, 3)

            # 15. 转换为轴角表示 (每个关节3个参数)
            axis_angle = art.math.rotation_matrix_to_axis_angle(body_rotations)

            # 16. 提取72维数据并发布
            inference_data = glb_full_pose_smpl_tensor.view(-1).detach().cpu().numpy().astype(np.float32)

            # 创建Float32MultiArray消息
            msg = Float32MultiArray()
            msg.data = inference_data.tolist()

            # 发布消息
            self.pub.publish(msg)

        except Exception as e:
            rospy.logerr(f"处理IMU数据时出错: {str(e)}")
            # 添加更详细的错误信息
            import traceback
            rospy.logerr(f"详细错误信息: {traceback.format_exc()}")

    def save_results(self):
        """保存结果到文件"""
        global infer_result, raw_imu, inference_count, total_inference_time

        if infer_result:
            try:
                rospy.loginfo("正在保存推理结果...")
                infer_result_torch = torch.stack(infer_result, dim=0)
                print(infer_result_torch.shape[0])
                imu_data_torch = torch.stack(imu_datas, dim=0)
                timestamp = rospy.Time.now().to_sec()
                filename = f"inference_result_{timestamp:.0f}.pt"
                filename_imu = f"imu_data_{timestamp:.0f}.pt"

                # 保存校准数据
                if self.calibrated:
                    self.calib_imu.astype(np.float32).tofile(("datasets/calib_imu.bin"))
                    self.RMI.numpy().astype(np.float32).tofile(("datasets/RMI.bin"))
                    self.RSB.numpy().astype(np.float32).tofile(("datasets/RSB.bin"))
                    rospy.loginfo("校准数据已保存")

                raw_imu = np.array(raw_imu)
                raw_imu.astype(np.float32).tofile(("datasets/raw_imu_data.bin"))
                infer_result_torch.numpy().astype(np.float32).tofile(("datasets/infer_result.bin"))
                imu_data_torch.numpy().astype(np.float32).tofile(("datasets/process_imu_data.bin"))

                # 打印推理统计信息
                if inference_count > 0:
                    avg_time = total_inference_time / inference_count
                    rospy.loginfo(f"总推理帧数: {inference_count}")
                    rospy.loginfo(f"总推理时间: {total_inference_time:.3f}秒")
                    rospy.loginfo(f"平均推理时间: {avg_time:.4f}秒/帧")

            except Exception as e:
                rospy.logerr(f"保存结果时出错: {str(e)}")
        else:
            rospy.logwarn("无推理结果需要保存")


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

    # 转换为numpy数组供NCNN使用
    return glb_states.cpu().numpy(), poser_states.cpu().numpy()


def load_ncnn_model():
    """加载NCNN模型"""
    param_path = "./weights/model_withoutInit.ncnn.param"
    bin_path = "./weights/model_withoutInit.ncnn.bin"
    ncnn_model = NCNNPoserWithoutInit(param_path, bin_path)
    return ncnn_model


if __name__ == '__main__':

    # 加载NCNN模型
    rospy.loginfo("正在加载NCNN模型...")
    ncnn_model = load_ncnn_model()
    rospy.loginfo("NCNN模型加载完成")

    # 初始化状态
    rospy.loginfo("正在初始化模型状态...")
    glb_states_ncnn, poser_states_ncnn = initialize_states()
    rospy.loginfo("模型状态初始化完成")

    # 加载身体模型 - 在CPU上
    body_model = art.ParametricModel(cfg.smpl_m, device='cpu')

    # 存贮所有推理结果，用于后续可视化
    raw_imu = []
    imu_datas = []
    infer_result = []
    calib_imu = []
    # 推理统计
    inference_count = 0
    total_inference_time = 0.0

    # 接入ros接口数据，在回调函数中进行推理
    try:
        processor = IMUDataProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass