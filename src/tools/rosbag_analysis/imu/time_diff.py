import rosbag
import matplotlib.pyplot as plt
import rospy

bag = rosbag.Bag('/home/melody/data/V1_03_difficult.bag')

timestamps = []

# 读取bag文件中的IMU消息
for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw']):
    timestamp = msg.header.stamp.to_sec()
    timestamps.append(timestamp)

bag.close()

time_diffs = [timestamps[i] - timestamps[i-1]
              for i in range(1, len(timestamps))]  # 计算时间差值

data_points = range(1, len(time_diffs)+1)  # 数据点索引

plt.scatter(data_points, time_diffs)
plt.xlabel('data_point')
plt.ylabel('time_diff')
plt.title('IMG Time Difference')
plt.show()
