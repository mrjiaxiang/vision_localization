import rosbag
import numpy as np
import matplotlib.pyplot as plt

bag = rosbag.Bag('/home/melody/data/img_imu/img_imu.bag')

# 计算采样时间间隔
dt = 1.0 / 200.0

# 存储时间戳和IMU数据的列表
timestamps = []
acceleration_data = []
angular_velocity_data = []

# 读取bag文件中的IMU消息
for topic, msg, t in bag.read_messages(topics=['/imu']):
    timestamp = msg.header.stamp.to_sec()
    acceleration = msg.linear_acceleration
    angular_velocity = msg.angular_velocity

    timestamps.append(timestamp)
    acceleration_data.append(acceleration)
    angular_velocity_data.append(angular_velocity)

bag.close()

# 提取加速度数据的X、Y、Z分量
acceleration_x_data = [data.x for data in acceleration_data]
acceleration_y_data = [data.y for data in acceleration_data]
acceleration_z_data = [data.z for data in acceleration_data]

# 提取角速度数据的X、Y、Z分量
angular_velocity_x_data = [data.x for data in angular_velocity_data]
angular_velocity_y_data = [data.y for data in angular_velocity_data]
angular_velocity_z_data = [data.z for data in angular_velocity_data]

freqs = np.fft.fftfreq(len(acceleration_x_data), dt)


# 执行FFT
# fft_data = np.fft.fft(imu_data)
# fft_freqs = np.fft.fftfreq(N, dt)

# # 取FFT结果的幅值谱
# fft_amplitudes = np.abs(fft_data)

# # 只保留正频率部分
# positive_freq_indices = fft_freqs >= 0
# fft_freqs = fft_freqs[positive_freq_indices]
# fft_amplitudes = fft_amplitudes[positive_freq_indices]

# # 绘制频谱图
# plt.plot(fft_freqs, fft_amplitudes)

# acc
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
acceleration_x_fft_data = np.fft.fft(acceleration_x_data)
fft_amplitudes_x = np.abs(acceleration_x_fft_data)
# 只保留正频率部分
positive_freq_indices = freqs >= 0
fft_freqs = freqs[positive_freq_indices]
fft_amplitudes_x = fft_amplitudes_x[positive_freq_indices]

plt.plot(fft_freqs, fft_amplitudes_x)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude X')

plt.subplot(3, 1, 2)
acceleration_y_fft_data = np.fft.fft(acceleration_y_data)
plt.plot(freqs, np.abs(acceleration_y_fft_data))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Y')

plt.subplot(3, 1, 3)
acceleration_z_fft_data = np.fft.fft(acceleration_z_data)
plt.plot(freqs, np.abs(acceleration_z_fft_data))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Z')
plt.legend()

plt.tight_layout()
plt.show()

# gyro
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
angular_velocity_x_fft_data = np.fft.fft(angular_velocity_x_data)
plt.plot(freqs, np.abs(angular_velocity_x_fft_data))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude X')

plt.subplot(3, 1, 2)
angular_velocity_y_fft_data = np.fft.fft(angular_velocity_y_data)
plt.plot(freqs, np.abs(angular_velocity_y_fft_data))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Y')

plt.subplot(3, 1, 3)
angular_velocity_z_fft_data = np.fft.fft(angular_velocity_z_data)
plt.plot(freqs, np.abs(angular_velocity_z_fft_data))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude Z')
plt.legend()

plt.tight_layout()
plt.show()
