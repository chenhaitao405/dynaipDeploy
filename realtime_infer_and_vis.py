#!/usr/bin/env python3
import rospy
import numpy as np
import math
import threading
from collections import deque
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class IMUVisualizer:
    def __init__(self):
        rospy.init_node('imu_visualizer', anonymous=True)
        
        # 数据缓冲区设置
        self.buffer_size = 100  # 存储最近100个数据点
        self.timestamps = deque(maxlen=self.buffer_size)
        self.imu_data = [
            {'rotation_matrix': [deque(maxlen=self.buffer_size) for _ in range(9)],  # 旋转矩阵 (9个元素)
             'acceleration': [deque(maxlen=self.buffer_size) for _ in range(3)]}  # 加速度 (x,y,z)
            for _ in range(6)  # 6个IMU
        ]
        
        # 线程锁确保数据安全
        self.lock = threading.Lock()
        
        # 创建matplotlib图形
        self.fig = plt.figure(figsize=(15, 15))
        self.axes = self._create_subplots()
        self.quivers = self._initialize_3d_quivers()
        self.acc_vectors = self._initialize_acc_vectors()
        
        # 设置布局
        plt.tight_layout()
        
        # 订阅ROS话题 - 替换为你的实际话题名称
        self.sub = rospy.Subscriber("/your/imu/topic", Float32MultiArray, self.callback)
        
        # 动画更新间隔 (ms)
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        
    def _create_subplots(self):
        """创建3x2的3D子图布局，每个IMU一个图"""
        axes = []
        for i in range(6):
            # 创建3D坐标系
            ax = self.fig.add_subplot(3, 2, i+1, projection='3d')
            ax.set_title(f'IMU {i+1}')
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 添加全局坐标轴参考
            ax.quiver(0, 0, 0, 1, 0, 0, color='gray', alpha=0.3, linestyle='dashed', length=0.5)
            ax.quiver(0, 0, 0, 0, 1, 0, color='gray', alpha=0.3, linestyle='dashed', length=0.5)
            ax.quiver(0, 0, 0, 0, 0, 1, color='gray', alpha=0.3, linestyle='dashed', length=0.5)
            
            axes.append(ax)
        return axes

    def _initialize_3d_quivers(self):
        """初始化3D方向图的箭头对象 (坐标轴)"""
        quivers = []
        for i, ax in enumerate(self.axes):
            # 创建三个坐标轴箭头 (红:X, 绿:Y, 蓝:Z)
            qx = ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.5, normalize=True)
            qy = ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.5, normalize=True)
            qz = ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.5, normalize=True)
            quivers.append((qx, qy, qz))
        return quivers

    def _initialize_acc_vectors(self):
        """初始化加速度向量箭头"""
        acc_vectors = []
        for i, ax in enumerate(self.axes):
            # 创建加速度向量箭头 (紫色)
            acc_vector = ax.quiver(0, 0, 0, 0, 0, 0, color='purple', linewidth=2, arrow_length_ratio=0.2)
            acc_vectors.append(acc_vector)
        return acc_vectors

    def callback(self, msg):
        """ROS话题回调函数，解析73维数据"""
        with self.lock:
            # 确保数据维度正确
            if len(msg.data) != 73:
                rospy.logwarn(f"Invalid data length: {len(msg.data)} (expected 73)")
                return
            
            # 提取时间戳 (索引0)
            self.timestamps.append(msg.data[0])
            
            # 解析6个IMU数据 (每个IMU占12个数据)
            for imu_idx in range(6):
                start_idx = 1 + imu_idx * 12
                
                # 提取旋转矩阵 (9个元素)
                rotation_matrix = msg.data[start_idx:start_idx+9]
                for i in range(9):
                    self.imu_data[imu_idx]['rotation_matrix'][i].append(rotation_matrix[i])
                
                # 提取并存储加速度 (后3个元素)
                acc_start = start_idx + 9
                acceleration = msg.data[acc_start:acc_start+3]
                for i in range(3):
                    self.imu_data[imu_idx]['acceleration'][i].append(acceleration[i])

    def update_plot(self, frame):
        """更新图形显示"""
        with self.lock:
            # 如果没有数据，直接返回
            if len(self.timestamps) == 0:
                return []
            
            # 更新所有图形元素
            all_artists = []
            
            # 更新每个IMU的3D图
            for imu_idx in range(6):
                ax = self.axes[imu_idx]
                
                # 获取最新的旋转矩阵
                rot_matrix = np.array([
                    self.imu_data[imu_idx]['rotation_matrix'][0][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][1][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][2][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][3][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][4][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][5][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][6][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][7][-1],
                    self.imu_data[imu_idx]['rotation_matrix'][8][-1]
                ]).reshape(3, 3)
                
                # 获取最新的加速度
                acc_x = self.imu_data[imu_idx]['acceleration'][0][-1]
                acc_y = self.imu_data[imu_idx]['acceleration'][1][-1]
                acc_z = self.imu_data[imu_idx]['acceleration'][2][-1]
                
                # 计算加速度向量长度用于缩放
                acc_magnitude = np.linalg.norm([acc_x, acc_y, acc_z])
                scale = min(1.0, 1.0 / (acc_magnitude + 0.001))  # 防止除以0
                
                # 删除旧箭头
                for artist in self.quivers[imu_idx]:
                    artist.remove()
                self.acc_vectors[imu_idx].remove()
                
                # 创建新箭头
                origin = [0, 0, 0]
                
                # 坐标轴箭头
                qx = ax.quiver(*origin, *rot_matrix[:,0], color='r', length=0.5, normalize=True)
                qy = ax.quiver(*origin, *rot_matrix[:,1], color='g', length=0.5, normalize=True)
                qz = ax.quiver(*origin, *rot_matrix[:,2], color='b', length=0.5, normalize=True)
                self.quivers[imu_idx] = (qx, qy, qz)
                
                # 加速度向量箭头 (紫色)，缩放以适应坐标系
                acc_vector = ax.quiver(*origin, 
                                       acc_x * scale, 
                                       acc_y * scale, 
                                       acc_z * scale, 
                                       color='purple', linewidth=2, arrow_length_ratio=0.2)
                self.acc_vectors[imu_idx] = acc_vector
                
                # 添加加速度标签
                acc_label = f"Acc: ({acc_x:.2f}, {acc_y:.2f}, {acc_z:.2f}) m/s²"
                if hasattr(self, f'acc_label_{imu_idx}'):
                    getattr(self, f'acc_label_{imu_idx}').remove()
                label = ax.text2D(0.05, 0.95, acc_label, transform=ax.transAxes, color='purple')
                setattr(self, f'acc_label_{imu_idx}', label)
                
                all_artists.extend([qx, qy, qz, acc_vector, label])
            
            return all_artists

    def run(self):
        """运行可视化"""
        try:
            plt.show()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    visualizer = IMUVisualizer()
    visualizer.run()