import rosbag
import matplotlib.pyplot as plt

bag = rosbag.Bag('img_imu.bag')

timestamps = []

# 读取bag文件中的IMU消息
for topic, msg, t in bag.read_messages(topics=['/imu']):
    timestamp = msg.header.stamp.to_sec()
    timestamps.append(timestamp)

bag.close()

plt.plot(timestamps)
plt.xlabel('data_point')
plt.ylabel('time_stamp')
plt.title('IMU TimeStamp')
plt.show()
