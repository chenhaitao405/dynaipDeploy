#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Float32MultiArray

import articulate as art
import torch
import os
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
from model.model_new import Poser
import utils.config as cfg
import random
from utils.data import fill_dip_nan, normalize_imu


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
        self.calibrated = False
        self.temp_dir = 'temp'
        os.makedirs(self.temp_dir, exist_ok=True)

        # 进行T-pose校准
        self.perform_tpose_calibration()

        rospy.loginfo("IMU 数据处理器已启动，等待数据...")
        rospy.on_shutdown(self.save_results)

    def perform_tpose_calibration(self):
        """执行T-pose校准"""
        rospy.loginfo("开始T-pose校准流程...")

        # 1. RMI校准（IMU参考坐标系校准）
        use_cached = input(
            '使用缓存的RMI校准数据? [y]/n    (如果选择no，请将IMU 5（左手）放直：x = 前方, y = 左侧, z = 上方，左手坐标系): ')

        if use_cached.lower() == 'n':
            rospy.loginfo("等待IMU数据进行RMI校准...")

            # 等待并获取一帧IMU数据
            imu_data = None

            def temp_callback(msg):
                nonlocal imu_data
                imu_data = msg

            temp_sub = rospy.Subscriber("/imu_rot", Float64MultiArray, temp_callback)

            # 等待数据
            rate = rospy.Rate(10)
            while imu_data is None and not rospy.is_shutdown():
                rate.sleep()

            temp_sub.unregister()

            if imu_data is not None:
                # 处理数据获取第5个IMU的旋转矩阵
                raw_data = np.array(imu_data.data)[1:]
                imu_array = raw_data.reshape(self.num_imus, self.data_per_imu)

                # 获取IMU 5（索引4）的旋转矩阵
                rotation_matrix = torch.from_numpy(imu_array[4][:9].reshape(3, 3)).float()

                RSI = rotation_matrix.t()
                self.RMI = torch.tensor([[0, -1, 0], [0, 0, -1], [-1, 0, 0.]]).mm(RSI)

                # 保存RMI
                torch.save(self.RMI, os.path.join(self.temp_dir, 'RMI.pt'))
                rospy.loginfo(f"RMI校准完成并保存: \n{self.RMI}")
            else:
                rospy.logerr("无法获取IMU数据进行RMI校准")
                return
        else:
            # 加载缓存的RMI
            rmi_path = os.path.join(self.temp_dir, 'RMI.pt')
            if os.path.exists(rmi_path):
                self.RMI = torch.load(rmi_path)
                rospy.loginfo(f"加载缓存的RMI: \n{self.RMI}")
            else:
                rospy.logerr("未找到缓存的RMI文件，请重新校准")
                return

        # 2. T-pose校准
        input('请站直并保持T-pose姿势，然后按Enter键。校准将在3秒后开始...')
        rospy.sleep(3)

        rospy.loginfo("正在进行T-pose校准...")

        # 获取T-pose时的IMU数据
        tpose_data = None

        def tpose_callback(msg):
            nonlocal tpose_data
            tpose_data = msg

        tpose_sub = rospy.Subscriber("/imu_rot", Float64MultiArray, tpose_callback)

        # 等待数据
        rate = rospy.Rate(10)
        while tpose_data is None and not rospy.is_shutdown():
            rate.sleep()

        tpose_sub.unregister()

        if tpose_data is not None:
            # 处理T-pose数据
            raw_data = np.array(tpose_data.data)[1:]
            imu_array = raw_data.reshape(self.num_imus, self.data_per_imu)

            # 获取所有IMU的旋转矩阵
            RIS_list = []
            for i in range(self.num_imus):
                rotation_matrix = torch.from_numpy(imu_array[i][:9].reshape(3, 3)).float()

                RIS_list.append(rotation_matrix)

            RIS = torch.stack(RIS_list, dim=0)

            # 计算RSB
            self.RSB = self.RMI.matmul(RIS).transpose(1, 2).matmul(torch.eye(3))

            # 保存RSB
            torch.save(self.RSB, os.path.join(self.temp_dir, 'RSB.pt'))

            self.calibrated = True
            rospy.loginfo("T-pose校准完成！")
            rospy.loginfo(f"RSB shape: {self.RSB.shape}")
        else:
            rospy.logerr("无法获取T-pose数据")

    def imu_callback(self, msg):
        global states
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

                # 分离旋转矩阵和加速度数据
                rotation_matrix = torch.from_numpy(single_imu[:9].reshape(3, 3)).float()
                acceleration = torch.from_numpy(single_imu[9:12]).float()

                acc_list.append(acceleration)
                ori_list.append(rotation_matrix)

            # 5. 堆叠数据并统一转移到CUDA设备
            acc = torch.stack(acc_list, dim=0).to(device)  # (6, 3)
            RIS = torch.stack(ori_list, dim=0).to(device)  # (6, 3, 3)


            #重力补偿
            # 使用校准后的旋转矩阵投影加速度
            gravity_compensaton = torch.tensor([0., 0., 1.],
                                                dtype=acc.dtype,
                                                device=device)
            acc_calibrated = RIS.bmm(acc.unsqueeze(-1)).squeeze(-1) + gravity_compensaton

            acc_calibrated *= 9.8

            print(acc_calibrated[4].reshape(-1, 3))  # 打印第一个IMU的加速度
            # 7. 对每个旋转矩阵应用

            # 8. 应用T-pose校准
            # 将校准矩阵移到正确的设备
            RMI_device = self.RMI.to(device)
            RSB_device = self.RSB.to(device)

            # 计算校准后的旋转矩阵：RMB = RMI @ RIS @ RSB
            RMB = torch.einsum('ij,njk,nkl->nil', RMI_device, RIS, RSB_device)



            # 转换到模型坐标系
            acc_model = acc_calibrated.mm(RMI_device.t())



            # 11. 归一化IMU数据
            single_imu = normalize_imu(acc_model, RMB)  # (1, 6, 12)

            # 12. 设置打印选项并打印调试信息
            torch.set_printoptions(sci_mode=False, precision=4)


            imu_datas.append(single_imu)

            # 13. 神经网络预测
            glb_full_pose_xsens, glb_full_pose_smpl_single, states = net.predict(
                single_imu,
                v_init,
                p_init,
                states  # 传入上一帧状态
            )

            # 14. 逆运动学计算
            local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl_single).view(
                glb_full_pose_smpl_single.shape[0], 24, 3, 3)
            infer_result.append(local_full_pose_smpl)

            body_rotations = glb_full_pose_smpl_single  # 形状: (23, 3, 3)

            # 15. 转换为轴角表示 (每个关节3个参数)
            axis_angle = art.math.rotation_matrix_to_axis_angle(body_rotations)

            # 16. 提取72维数据并发布
            inference_data = glb_full_pose_smpl_single.view(-1).detach().cpu().numpy().astype(np.float32)

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
        global infer_result, raw_imu

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
                    calibration_data = {
                        'RMI': self.RMI,
                        'RSB': self.RSB
                    }
                    torch.save(calibration_data, os.path.join(self.temp_dir, 'calibration_data.pt'))
                    rospy.loginfo("校准数据已保存")

                raw_imu = np.array(raw_imu)
                raw_imu.astype(np.float32).tofile(("datasets/raw_imu_data.bin"))
                infer_result_torch.numpy().astype(np.float32).tofile(("datasets/infer_result.bin"))
                imu_data_torch.numpy().astype(np.float32).tofile(("datasets/process_imu_data.bin"))

            except Exception as e:
                rospy.logerr(f"保存结果时出错: {str(e)}")
        else:
            rospy.logwarn("无推理结果需要保存")


if __name__ == '__main__':

    # 模型加载
    device = torch.device("cuda:0")
    net = Poser().to(device)
    net.load_state_dict(torch.load(cfg.weight_s, map_location='cuda:0'))  # DynaIP* in paper
    net.eval()

    # p_init与v_init初始化参数加载
    test_folder = os.path.join(cfg.work_dir, 'test')
    test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
                  for foldername, _, filenames in os.walk(test_folder)
                  for filename in filenames if filename.endswith('.pt')]

    f = os.path.join(test_folder, test_files[0])
    data = torch.load(f)
    vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
    v_init = data['joint']['velocity'][:1, vel_mask].float().to(device)
    pose = data['joint']['orientation']
    pose = pose.view(pose.shape[0], -1, 6)[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]]
    p_init = pose[:1].to(device)
    body_model = art.ParametricModel(cfg.smpl_m, device='cuda:0')

    # 存贮所有推理结果，用于后续可视化
    raw_imu = []
    imu_datas = []
    infer_result = []
    states = None

    # 接入ros接口数据，在回调函数中进行推理
    try:
        processor = IMUDataProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass