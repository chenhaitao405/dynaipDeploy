import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import os
import utils.config as cfg

# 加载数据
test_folder = os.path.join(cfg.work_dir, 'test')
test_files = [os.path.relpath(os.path.join(foldername, filename), test_folder)
              for foldername, _, filenames in os.walk(test_folder)
              for filename in filenames if filename.endswith('.pt')]

f = os.path.join(test_folder, test_files[0])
print("Loading:", f)
data = torch.load(f)
imu = data['imu']['imu']  # shape: (batch, 6, 12)
imu_acc = imu[:, :, 9:].numpy()  # shape: (batch, 6, 3)
num_frames = imu_acc.shape[0]


# f = 'imu_data_1751770825.pt'
# data = torch.load(f).squeeze()
# imu_acc = data[:, :, 9:].numpy()  # shape: (batch, 6, 3)
# num_frames = imu_acc.shape[0]

# 设置绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
text_box = None  # 用于显示文本
spacing = 2.0
origins = np.array([[i * spacing, 0, 0] for i in range(6)])
colors = ['r', 'g', 'b', 'c', 'm', 'y']
play_speed = 5
paused = [False]

# 初始化函数
def init():
    ax.clear()
    ax.set_title("IMU Acceleration Vectors (Dynamic)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 6 * spacing])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.view_init(elev=20, azim=-35)
    return []

# 更新函数  
def update(frame):
    if paused[0]:
        return []

    ax.clear()
    acc_data = imu_acc[frame % num_frames]

    # 绘制矢量和原点
    for i, (vec, origin) in enumerate(zip(acc_data, origins)):
        ax.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=colors[i],
            arrow_length_ratio=0.2,
            linewidth=2,
            length=1.0,
            label=f'IMU {i+1}'
        )
        ax.scatter(origin[0], origin[1], origin[2],
                   color=colors[i], s=50, marker='o')

    # 添加右上角文字（加速度数值）
    acc_text = "\n".join([
        f"IMU {i+1}: [{vec[0]:+.2f}, {vec[1]:+.2f}, {vec[2]:+.2f}]"
        for i, vec in enumerate(acc_data)
    ])
    ax.text2D(0.75, 0.95, acc_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f"IMU Accelerations - Frame {frame + 1}/{num_frames}")
    ax.set_xlim([-1, 6 * spacing])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    # ax.view_init(elev=20, azim=-35)
    return []

# 键盘交互
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
        print("Paused" if paused[0] else "Resumed")
    elif event.key == 'up':
        ani.event_source.interval = max(10, ani.event_source.interval / 1.5)
        print(f"Increased speed: interval = {ani.event_source.interval:.1f} ms")
    elif event.key == 'down':
        ani.event_source.interval *= 1.5
        print(f"Decreased speed: interval = {ani.event_source.interval:.1f} ms")

fig.canvas.mpl_connect('key_press_event', on_key)

# 启动动画
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                    blit=False, interval=int(300 / play_speed), repeat=True)

plt.tight_layout()
plt.show()
