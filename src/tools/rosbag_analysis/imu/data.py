import rosbag
import matplotlib.pyplot as plt

bag = rosbag.Bag('/home/melody/data/img_imu/img_imu.bag')

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

# 绘制加速度数据
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(timestamps, acceleration_x_data, label='X')
plt.xlabel('time_stamp')
plt.ylabel('acceleration X')

plt.subplot(3, 1, 2)
plt.plot(timestamps, acceleration_y_data, label='Y')
plt.xlabel('time_stamp')
plt.ylabel('acceleration Y')

plt.subplot(3, 1, 3)
plt.plot(timestamps, acceleration_z_data, label='Z')
plt.xlabel('time_stamp')
plt.ylabel('acceleration Z')
plt.title('acceleration plot')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制角速度数据
plt.subplot(3, 1, 1)
plt.plot(timestamps, angular_velocity_x_data, label='X')
plt.xlabel('time_stamp')
plt.ylabel('angular X')

plt.subplot(3, 1, 2)
plt.plot(timestamps, angular_velocity_y_data, label='Y')
plt.xlabel('time_stamp')
plt.ylabel('angular Y')

plt.subplot(3, 1, 3)
plt.plot(timestamps, angular_velocity_z_data, label='Z')
plt.xlabel('time_stamp')
plt.ylabel('angular Z')
plt.title('angular plot')
plt.legend()

plt.tight_layout()
plt.show()
