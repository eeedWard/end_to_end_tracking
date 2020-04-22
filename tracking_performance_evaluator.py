# rosbag record /driver/ptu_state /estimation/predicted_vip_traj_in_pantilt /cinematography/pt_trajectory /computer_vision/target_projected
# use ptu state and re-create the measure vip state in pan - tilt.

import sys
sys.path.insert(0,'/opt/ros/melodic/lib/python2.7/dist-packages/')
import rosbag
import matplotlib.pyplot as plt
import numpy as np

bag_path = 'bags/ws2.bag'
bag = rosbag.Bag(bag_path, mode='r')


pan_ptu_list = []
tilt_ptu_list = []
fov = 0.0
for topic, msg, t in bag.read_messages(topics='/driver/ptu_state'):
    pan_ptu_list.append([msg.header.stamp.to_sec(), msg.pan_position])
    tilt_ptu_list.append([msg.header.stamp.to_sec(), msg.tilt_position])
    fov = msg.field_of_view_angle
pan_ptu = np.array(pan_ptu_list)
tilt_ptu = np.array(tilt_ptu_list)

pan_ptu[:,0] -= bag.get_start_time()
tilt_ptu[:,0] -= bag.get_start_time()

assert fov != 0.0

x_centre_list = []
y_centre_list = []
for topic, msg, t in bag.read_messages(topics='/computer_vision/target_projected'):
    if len(msg.detections)<=0:
        continue
    x_tuple = (msg.detections[msg.target_index].cvBox_tracker.xmin, msg.detections[msg.target_index].cvBox_tracker.xmax)
    y_tuple = (msg.detections[msg.target_index].cvBox_tracker.ymin, msg.detections[msg.target_index].cvBox_tracker.ymax)
    x_centre = (x_tuple[0] + (x_tuple[1]-x_tuple[0])/2.0) * fov
    y_centre = (y_tuple[0] + (y_tuple[1]-y_tuple[0])/2.0) * fov * 9.0/16.0
    x_centre_list.append([msg.header.stamp.to_sec(), x_centre])
    y_centre_list.append([msg.header.stamp.to_sec(), y_centre])

x_centre_arr = np.array(x_centre_list)
y_centre_arr = np.array(y_centre_list)
x_centre_arr[:,0] -= bag.get_start_time()
y_centre_arr[:,0] -= bag.get_start_time()



# restric arrays to tracking period
# choose manually start and end time
tracking_start_time = 0.0
tracking_lost_time = 2.1

# Look for closest timestamp
for i in range(pan_ptu.shape[0]):
    if pan_ptu[i, 0] >= tracking_start_time:
        start_index_ptu = i
        break
for i in range(pan_ptu.shape[0]):
    if pan_ptu[i, 0] >= tracking_lost_time:
        end_index_ptu = i
        break
pan_ptu = pan_ptu[start_index_ptu:end_index_ptu, :]
tilt_ptu = tilt_ptu[start_index_ptu:end_index_ptu, :]


# restrict x and y arrays. Look for closest timestamp
for i in range(x_centre_arr.shape[0]):
    if x_centre_arr[i, 0] >= tracking_start_time:
        start_index_xy = i
        break
for i in range(x_centre_arr.shape[0]):
    if x_centre_arr[i, 0] >= tracking_lost_time:
        end_index_xy = i
        break
x_centre_arr = x_centre_arr[start_index_xy:end_index_xy, :]
y_centre_arr = y_centre_arr[start_index_xy:end_index_xy, :]

print("Episode length: {} seconds".format(tracking_lost_time-tracking_start_time))



# evaluate second derivative: [(X_t+1 - X_t) / deltaT1 - (X_t - X_t-1) / (deltaT2)]/ (deltaT1/2 + deltaT2/2)
pan_second_der_list = []
tilt_second_der_list = []
TOL = 0.01
for i in range(1, pan_ptu.shape[0]-1):
    delta_t_up = pan_ptu[i+1,0] - pan_ptu[i,0]
    delta_t_down = pan_ptu[i,0] - pan_ptu[i-1,0]
    if delta_t_up <= TOL or delta_t_down <= TOL:
        continue
    pan_first_der_up = (pan_ptu[i+1,1] - pan_ptu[i,1]) / delta_t_up
    pan_first_der_down = (pan_ptu[i,1] - pan_ptu[i-1,1]) / delta_t_down
    pan_second_der = (pan_first_der_up - pan_first_der_down) / (delta_t_up/2 + delta_t_down/2)
    pan_second_der_list.append(pan_second_der**2)

    tilt_first_der_up = (tilt_ptu[i+1,1] - tilt_ptu[i,1]) / delta_t_up
    tilt_first_der_down = (tilt_ptu[i,1] - tilt_ptu[i-1,1]) / delta_t_down
    tilt_second_der = (tilt_first_der_up - tilt_first_der_down) / (delta_t_up/2 + delta_t_down/2)
    tilt_second_der_list.append(tilt_second_der**2)

smoothness_pan = sum(pan_second_der_list)/len(pan_second_der_list)
smoothness_tilt = sum(tilt_second_der_list)/len(tilt_second_der_list)
smoothness = 1.0/np.sqrt((smoothness_pan + smoothness_tilt) / 2.0)

print("smoothness score: {}".format(smoothness))


# evaluate tracking accuracy: distance target - centre
img_centre_x = 0.5 * fov
img_centre_y = 0.5 * fov * 9.0/16.0

squared_distance_list = []
for i in range(x_centre_arr.shape[0]):
    squared_distance = (x_centre_arr[i,1] - img_centre_x)**2 + (y_centre_arr[i,1] - img_centre_y)**2
    squared_distance_list.append(squared_distance)

tracking_accuracy = 1.0/np.sqrt(sum(squared_distance_list)/len(squared_distance_list))
print("tracking accuracy score: {}".format(tracking_accuracy))