import numpy as np

with open('datasets/infer_result.bin', 'rb') as file:
    infer_result = file.read()

infer_result = np.frombuffer(infer_result, dtype=np.float32)


with open('datasets/process_imu_data.bin', 'rb') as file:
    process_imu_data = file.read()

process_imu_data = np.frombuffer(process_imu_data, dtype=np.float32).reshape(-1,6,12)

with open('datasets/raw_imu_data.bin', 'rb') as file:
    raw_imu_data = file.read()
raw_imu_data = np.frombuffer(raw_imu_data, dtype=np.float32).reshape(-1,6,12)


print(1)