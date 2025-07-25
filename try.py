#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
import articulate as art
import torch
import os
from model.model import Poser
import utils.config as cfg

def imu_publisher():
    # 初始化ROS节点
    rospy.init_node('imu_publisher', anonymous=True)
    
    # 创建发布者，话题名为"imu_rot"，消息类型为Float64MultiArray
    pub = rospy.Publisher('imu_rot', Float64MultiArray, queue_size=10)
    
    # 设置发布频率为60Hz
    rate = rospy.Rate(60)
    
    # 初始化模型和加载数据（根据您提供的代码）
    body_model = art.ParametricModel(cfg.smpl_m, device='cpu')  
    test_folder = os.path.join(cfg.work_dir, 'test')
    test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
                for foldername, _, filenames in os.walk(test_folder)
                for filename in filenames if filename.endswith('.pt')]
    device = torch.device("cpu")

    net = Poser().to(device)
    net.load_state_dict(torch.load(cfg.weight_s, map_location='cpu'))
    net.eval()

    # 加载数据文件
    f = os.path.join(test_folder, test_files[13])
    print(f)
    data = torch.load(f)
    imu = data['imu']['imu'].to(device)  # shape: batch,6,12
    
    # 获取数据的总帧数
    num_frames = imu.shape[0]
    current_frame = 0
    
    print("开始发布IMU数据...")
    
    while not rospy.is_shutdown() and current_frame < num_frames:
        # 获取当前时间戳
        timestamp = rospy.get_time()
        
        # 获取当前帧的IMU数据 (6x12=72维)
        imu_data = imu[current_frame].flatten().tolist()
        
        # 创建消息并填充数据 (时间戳 + IMU数据 = 73维)
        msg = Float64MultiArray()
        msg.data = [timestamp] + imu_data
        
        # 发布消息
        pub.publish(msg)
        
        # 打印日志信息
        rospy.loginfo(f"发布第 {current_frame} 帧数据，时间戳: {timestamp}")
        
        # 移动到下一帧
        current_frame += 1
        
        # 按照60Hz的频率等待
        rate.sleep()

if __name__ == '__main__':
    try:
        imu_publisher()
    except rospy.ROSInterruptException:
        pass