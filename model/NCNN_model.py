import numpy as np
import ncnn
import torch


class NCNNPoserWithoutInit:
    def __init__(self, param_path, bin_path):
        """
        初始化NCNN模型

        Args:
            param_path (str): NCNN模型参数文件路径 (.param)
            bin_path (str): NCNN模型权重文件路径 (.bin)
        """
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # 可选：使用Vulkan加速
        self.net.opt.num_threads = 4  # 设置线程数

        # 加载模型
        ret = self.net.load_param(param_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load param file: {param_path}")

        ret = self.net.load_model(bin_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load model file: {bin_path}")

        # 定义传感器和关节名称
        self.sensor_names = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
        self.v_names = ['Root', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
        self.p_names = ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'L3',
                        'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftUpperArm',
                        'RightUpperArm']

        print(f"NCNN模型加载成功: {param_path}")

    def r6d_to_rotation_matrix(self, r6d):
        """
        将6D旋转表示转换为旋转矩阵

        Args:
            r6d: shape (..., 6) 的6D旋转表示

        Returns:
            旋转矩阵 shape (..., 3, 3)
        """
        # 重塑为 (..., 2, 3)
        original_shape = r6d.shape[:-1]
        r6d = r6d.reshape(-1, 6)

        x_raw = r6d[:, :3]  # 第一个向量
        y_raw = r6d[:, 3:]  # 第二个向量

        # 标准化第一个向量
        x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)

        # 计算第三个向量（叉积）
        z = np.cross(x, y_raw, axis=1)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)

        # 重新计算第二个向量
        y = np.cross(z, x, axis=1)

        # 组合成旋转矩阵
        rotation_matrix = np.stack([x, y, z], axis=2)

        # 恢复原始形状
        return rotation_matrix.reshape(*original_shape, 3, 3)

    def _reduced_glb_6d_to_full_glb_mat_xsens(self, glb_reduced_pose, orientation):
        """
        将减少的6D全局姿态转换为完整的全局矩阵（Xsens格式）

        Args:
            glb_reduced_pose: shape (batch, 11*6) 的6D姿态
            orientation: shape (batch, 6, 3, 3) 的方向矩阵

        Returns:
            全局完整姿态矩阵
        """
        joint_set = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
        sensor_set = [0, 20, 16, 6, 13, 9]
        ignored = [10, 14, 17, 18, 21, 22]
        parent = [9, 13, 16, 16, 20, 20]

        batch_size = glb_reduced_pose.shape[0]

        # 获取根旋转
        root_rotation = orientation[:, 0]  # shape: (batch, 3, 3)

        # 将6D转换为旋转矩阵
        glb_reduced_pose = self.r6d_to_rotation_matrix(glb_reduced_pose).reshape(batch_size, len(joint_set), 3, 3)

        # 转换到全局坐标系
        glb_reduced_pose = np.matmul(root_rotation[:, np.newaxis], glb_reduced_pose)
        orientation[:, 1:] = np.matmul(root_rotation[:, np.newaxis], orientation[:, 1:])

        # 创建完整的全局姿态矩阵
        global_full_pose = np.eye(3)[np.newaxis, np.newaxis].repeat(batch_size, axis=0).repeat(23, axis=1)
        global_full_pose[:, joint_set] = glb_reduced_pose
        global_full_pose[:, sensor_set] = orientation
        global_full_pose[:, ignored] = global_full_pose[:, parent]

        return global_full_pose

    def _glb_mat_xsens_to_glb_mat_smpl(self, glb_full_pose_xsens):
        """
        将Xsens格式的全局矩阵转换为SMPL格式

        Args:
            glb_full_pose_xsens: Xsens格式的全局姿态矩阵

        Returns:
            SMPL格式的全局姿态矩阵
        """
        batch_size = glb_full_pose_xsens.shape[0]
        glb_full_pose_smpl = np.eye(3)[np.newaxis, np.newaxis].repeat(batch_size, axis=0).repeat(24, axis=1)

        indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]

        for idx, i in enumerate(indices):
            glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]

        return glb_full_pose_smpl

    def forward(self, x, glb_states=None, poser_states=None):
        """
        使用NCNN模型进行前向推理

        Args:
            x: 输入数据，shape为 (1, 6, 12)
            glb_states: 全局状态，shape为 (2, 1, 1, 36)
            poser_states: 姿态状态，shape为 (12, 2, 1, 200)

        Returns:
            tuple: (v_out, p_out, updated_glb_states, updated_poser_states)
        """
        # 准备输入数据 - 根据测试脚本的格式调整
        in0 = x.squeeze(0).astype(np.float32)  # (1,6,12) -> (6,12)
        in1 = glb_states.squeeze(1).astype(np.float32)  # (2,1,1,36) -> (2,1,36)
        in2 = poser_states.squeeze(2).astype(
            np.float32)  # (12,2,1,200) -> (12,2,200) - Fixed: squeeze dimension 2, not 1

        # 创建提取器
        with self.net.create_extractor() as ex:
            # 设置输入
            ex.input("in0", ncnn.Mat(in0).clone())
            ex.input("in1", ncnn.Mat(in1).clone())
            ex.input("in2", ncnn.Mat(in2).clone())

            # 提取输出
            _, out0 = ex.extract("out0")  # v_out
            _, out1 = ex.extract("out1")  # p_out
            _, out2 = ex.extract("out2")  # updated_glb_states
            _, out3 = ex.extract("out3")  # updated_poser_states

            # 转换为numpy数组并调整形状
            v_out = np.array(out0)[np.newaxis, :]  # 添加batch维度
            p_out = np.array(out1)
            updated_glb_states = np.array(out2)[:, np.newaxis, :]  # 恢复中间维度
            updated_poser_states = np.array(out3)[:, :, np.newaxis, :]  # 恢复中间维度

        return v_out, p_out, updated_glb_states, updated_poser_states

    def predict(self, x, glb_states=None, poser_states=None):
        """
        使用NCNN模型进行推理预测

        Args:
            x: 输入数据，shape为 (1, 6, 12) - 单帧IMU数据
            glb_states: 全局状态，shape为 (2, 1, 1, 36)（可选）
            poser_states: 姿态状态，shape为 (12, 2, 1, 200)（可选）

        Returns:
            tuple: (glb_full_pose_xsens, glb_full_pose_smpl, glb_states, poser_states)
        """
        # 如果没有提供状态，创建零状态
        if glb_states is None:
            glb_states = np.zeros((2, 1, 1, 36), dtype=np.float32)
        if poser_states is None:
            poser_states = np.zeros((12, 2, 1, 200), dtype=np.float32)

        # 确保输入是numpy数组
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(glb_states, torch.Tensor):
            glb_states = glb_states.numpy()
        if isinstance(poser_states, torch.Tensor):
            poser_states = poser_states.numpy()

        # 执行前向推理
        v_partition, p_partition, updated_glb_states, updated_poser_states = self.forward(
            x, glb_states, poser_states
        )

        # 后处理 - 重新排列姿态数据
        pose = p_partition.reshape(-1, 11, 6)[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]]

        # 处理方向数据
        orientation = x[:, :, :9].reshape(-1, 6, 3, 3)

        # 转换为完整的姿态矩阵
        glb_full_pose_xsens = self._reduced_glb_6d_to_full_glb_mat_xsens(pose, orientation)
        glb_full_pose_smpl = self._glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens)

        # 转换回torch张量以保持与原始接口一致
        glb_full_pose_xsens = torch.from_numpy(glb_full_pose_xsens)
        glb_full_pose_smpl = torch.from_numpy(glb_full_pose_smpl)
        updated_glb_states = torch.from_numpy(updated_glb_states)
        updated_poser_states = torch.from_numpy(updated_poser_states)

        return glb_full_pose_xsens, glb_full_pose_smpl, updated_glb_states, updated_poser_states

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'net'):
            del self.net


# 测试函数
def test_ncnn_inference():
    """测试NCNN推理功能"""
    # 替换为你的模型路径
    param_path = "../weights/model_withoutInit.ncnn.param"
    bin_path = "../weights/model_withoutInit.ncnn.bin"

    try:
        # 初始化预测器
        predictor = NCNNPoserWithoutInit(param_path, bin_path)

        # 准备测试数据
        torch.manual_seed(0)
        x = torch.rand(1, 6, 12, dtype=torch.float)
        glb_states = torch.rand(2, 1, 1, 36, dtype=torch.float)
        poser_states = torch.rand(12, 2, 1, 200, dtype=torch.float)

        # 进行推理
        glb_pose_xsens, glb_pose_smpl, updated_glb_states, updated_poser_states = predictor.predict(
            x, glb_states, poser_states
        )

        print("NCNN推理成功!")
        print(f"输入形状: x={x.shape}, glb_states={glb_states.shape}, poser_states={poser_states.shape}")
        print(f"Xsens姿态形状: {glb_pose_xsens.shape}")
        print(f"SMPL姿态形状: {glb_pose_smpl.shape}")
        print(f"更新后glb状态形状: {updated_glb_states.shape}")
        print(f"更新后poser状态形状: {updated_poser_states.shape}")

        return glb_pose_xsens, glb_pose_smpl, updated_glb_states, updated_poser_states

    except Exception as e:
        print(f"NCNN推理失败: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 测试推理
    result = test_ncnn_inference()

    if result is not None:
        print("测试完成，所有输出正常!")
    else:
        print("测试失败，请检查模型路径和输入数据!")