#!/usr/bin/env python
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


class IMUDataProcessor:
    def __init__(self):
        rospy.init_node('imu_data_processor', anonymous=True)

        # 订阅 /imu_rot 话题
        rospy.Subscriber("/imu_rot", Float64MultiArray, self.imu_callback)
        self.pub = rospy.Publisher('inference_data', Float32MultiArray, queue_size=10)

        # 根据数据格式初始化变量
        self.num_imus = 6
        self.data_per_imu = 12  # 9维旋转矩阵 + 3维加速度

        rospy.loginfo("IMU 数据处理器已启动，等待数据...")

        rospy.on_shutdown(self.save_results)

    def imu_callback(self, msg):
        global states
        """处理接收到的 IMU 数据"""
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
            ori = torch.stack(ori_list, dim=0).to(device)  # (6, 3, 3)
            

            # 使用imu2数据代替imu3（如果需要）
            # ori[3,:,:] = ori[0,:,:]

            # 6. 投影后补偿z轴加速度
            # *** 修正：确保重力补偿向量在同一设备上 ***
            gravity_compensation = torch.tensor([0., 0., 1.],
                                                dtype=acc.dtype,
                                                device=device)
            acc = ori.bmm(acc.unsqueeze(-1)).squeeze(-1) + gravity_compensation

            # 7. 对输入imu旋转矩阵做预处理
            # *** 修正：确保Ry矩阵在同一设备上 ***
            Ry = torch.tensor([
                [0, 1, 0],  # B的x轴 = A的y轴
                [0, 0, 1],  # B的y轴 = A的z轴
                [1, 0, 0]  # B的z轴 = A的x轴
            ], dtype=ori.dtype, device=device)

            # 8. 对每个旋转矩阵应用变换：R_new = Ry @ R_original
            ori = torch.einsum('ij,njk->nik', Ry, ori)
            acc = torch.einsum('ij,nj->ni', Ry, acc)  # 适用于2维张量

            # 9. 加速度缩放
            acc *= 9.8

            # 10. 归一化IMU数据
            single_imu = normalize_imu(acc, ori)  # (1, 6, 12)


            # 11. 设置打印选项并打印调试信息
            torch.set_printoptions(sci_mode=False, precision=4)
            print(single_imu[:, 5, 9:].reshape(-1, 3))  # 打印第一个IMU的旋转矩阵

            imu_datas.append(single_imu)

            # 12. 神经网络预测
            glb_full_pose_xsens, glb_full_pose_smpl_single, states = net.predict(
                single_imu,
                v_init,
                p_init,
                states  # 传入上一帧状态
            )

            # 13. 逆运动学计算
            local_full_pose_smpl = body_model.inverse_kinematics_R(glb_full_pose_smpl_single).view(
                glb_full_pose_smpl_single.shape[0], 24, 3, 3)
            infer_result.append(local_full_pose_smpl)

            body_rotations = glb_full_pose_smpl_single  # 形状: (23, 3, 3)

            # 14. 转换为轴角表示 (每个关节3个参数)
            axis_angle = art.math.rotation_matrix_to_axis_angle(body_rotations)

            # 15. 提取72维数据并发布
            # *** 修正：确保转换到CPU再转numpy ***
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

        except Exception as e:
            rospy.logerr(f"处理IMU数据时出错: {str(e)}")


    def save_results(self):
        """保存结果到文件"""
        global infer_result,raw_imu
        
        if infer_result:
            try:
                rospy.loginfo("正在保存推理结果...")
                infer_result_torch = torch.stack(infer_result, dim=0)
                print(infer_result_torch.shape[0])
                imu_data_torch = torch.stack(imu_datas, dim=0)
                timestamp = rospy.Time.now().to_sec()
                filename = f"inference_result_{timestamp:.0f}.pt"
                filename_imu = f"imu_data_{timestamp:.0f}.pt"

                # torch.save(infer_result_torch, filename)
                # torch.save(imu_data_torch, filename_imu)

                # rospy.loginfo(f"结果已保存到 {filename_imu}")

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
    net.load_state_dict(torch.load(cfg.weight_s, map_location='cuda:0')) # DynaIP* in paper
    net.eval()

    # p_init与v_init初始化参数加载
    test_folder = os.path.join(cfg.work_dir, 'test')
    test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
            for foldername, _, filenames in os.walk(test_folder)
            for filename in filenames if filename.endswith('.pt')]
    

    f = os.path.join(test_folder,test_files[0])
    data = torch.load(f)
    vel_mask = torch.tensor([0, 15, 20, 21, 7, 8])
    v_init = data['joint']['velocity'][:1, vel_mask].float().to(device)
    pose = data['joint']['orientation']
    pose = pose.view(pose.shape[0], -1, 6)[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]]
    p_init = pose[:1].to(device)
    body_model = art.ParametricModel(cfg.smpl_m, device='cuda:0')

    #存贮所有推理结果，用于后续可视化
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

