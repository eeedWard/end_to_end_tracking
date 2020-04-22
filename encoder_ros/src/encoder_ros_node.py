#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import mmap
from torchvision import transforms
import torch
import random
from msgs.msg import cvDetections, GPUFrameHandle, targetSelection, PTUState
from carla_msgs.msg import CarlaControl
import sys
sys.path.append("catkin_ws/src")
sys.path.append("catkin_ws/src/end_to_end")
sys.path.append("catkin_ws/src/end_to_end/encoder")
sys.path.append("catkin_ws/src/end_to_end/predictor")
from end_to_end.encoder.encoder import Encoder

class EncoderRos:
    # how this works:
    # stop ptu virtual driver: rosnode kill /driver/ptu_virtual_driver (otherwise we both publish to /driver/setvelpos)
    # Do few times /carla/control step once, until lists are full: rostopic pub -r 0.5 /carla/control ...
    # you are in pause with all lists full (prev imgs and so on)
    # select a detection on current frame: click on bounding box
    # put carla simulation back in continuous mode: the policy encoder will take over
    # After tracking, you need to restart encoder to clean the target box

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.control_mode = CarlaControl.PLAY
        self.steps_count = 0
        self.target_box = None
        self.detection_msg = None
        self.last_ptu_state_msg = None

        image_shared_topic = rospy.get_param('~image_shared_topic')
        in_detections_topic = rospy.get_param('~in_detections_topic')
        in_sim_control_topic = rospy.get_param('~in_sim_control_topic')
        in_target_command_topic = rospy.get_param('~in_target_command_topic')
        ptu_state_topic = rospy.get_param('~ptu_state_topic')

        self.model = Encoder(prev_img_number=5, inference_mode=True)

        self.memoryMappedBuffersDict = {}

        rospy.Subscriber(image_shared_topic, GPUFrameHandle, self.callback_image, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(in_detections_topic, cvDetections, self.detections_cb, queue_size=1)
        rospy.Subscriber(in_sim_control_topic, CarlaControl, self.sim_control_cb, queue_size=1)
        rospy.Subscriber(in_target_command_topic, targetSelection, self.select_target_command_callback, queue_size=1)
        rospy.Subscriber(ptu_state_topic, PTUState, self.ptu_state_cb, queue_size=1)

        self.pub_ptu_state = rospy.Publisher(ptu_state_topic, PTUState, queue_size=1)

        self.prev_imgs_list = []
        self.x_list = []
        self.y_list = []
        self.prev_tilt_list = []
        self.prev_pan_list = []

    def callback_image(self, image_msg):
        if self.last_ptu_state_msg is None:
            rospy.logwarn_throttle(1, "last_ptu_state_msg is none, return")
            return
        if self.control_mode == CarlaControl.PLAY and self.target_box is None:
            rospy.loginfo_throttle(3.0, "Mode PLAY, doing nothing")
            self.steps_count = 0
            self.target_box = None
            self.prev_imgs_list = []
            self.x_list = []
            self.y_list = []
            self.prev_tilt_list = []
            self.prev_pan_list = []
            return

        # grab image
        filePath = "/dev/shm/" + image_msg.share_path
        if filePath in self.memoryMappedBuffersDict:
            buf = self.memoryMappedBuffersDict[filePath]
        else:
            fd = os.open(filePath, os.O_RDONLY)
            buf = mmap.mmap(fd, 540 * 960 * 3, mmap.MAP_SHARED, mmap.PROT_READ)
            self.memoryMappedBuffersDict[filePath] = buf
        img = np.frombuffer(buf, np.uint8).reshape((540, 960, 3))

        img = cv2.resize(img, (208, 160), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.prev_imgs_list.append(img_rgb)


        # same transformation as utils_dataset pt_to_xy
        x = image_msg.ptu_state_of_this_frame.pan_position / image_msg.ptu_state_of_this_frame.field_of_view_angle
        y = image_msg.ptu_state_of_this_frame.tilt_position / (image_msg.ptu_state_of_this_frame.field_of_view_angle * 9.0/16.0)

        self.x_list.append(x)
        self.y_list.append(y)
        self.prev_pan_list.append(image_msg.ptu_state_of_this_frame.pan_position)
        self.prev_tilt_list.append(image_msg.ptu_state_of_this_frame.tilt_position)

        assert len(self.prev_imgs_list) == len(self.x_list) == len(self.y_list) == len(self.prev_tilt_list) == len(self.prev_pan_list) <= 5

        rospy.loginfo("Prev imgs stored {}/5".format(len(self.prev_imgs_list)))

        if len(self.prev_imgs_list) < 5:
            return

        while self.target_box is None:
            rospy.logwarn_throttle(0.5, "Doing nothing, waiting for target box")
            if self.control_mode == CarlaControl.PLAY or self.control_mode == CarlaControl.STEP_ONCE:
                # get out of this locking mode, either reset if play or contiue updating frames if we step
                self.pop_all_lists()
                return

        prev_imgs_stacked = np.zeros((160, 208, 3 * len(self.prev_imgs_list)), dtype=np.uint8)
        for i in range(len(self.prev_imgs_list)):
            prev_imgs_stacked[:,:, 3*i : 3*i+3] = self.prev_imgs_list[i]

        if self.target_box is None:
            rospy.logwarn("You need to choose a target to track, returning")
            self.pop_all_lists()
            return

        dx_list = []
        dy_list = []
        for i in range(len(self.x_list) - 1):
            dx_list.append(self.x_list[i+1]- self.x_list[i])
            dy_list.append(self.y_list[i+1] - self.y_list[i])
        dx_list.append(0.0)
        dy_list.append(0.0)

        to_tensor = transforms.ToTensor()
        sample = {'prev_imgs': to_tensor(prev_imgs_stacked).unsqueeze(0).to(self.device),
                  'commands_dx': torch.tensor(dx_list, dtype=torch.float).unsqueeze(0).to(self.device),
                  'commands_dy': torch.tensor(dy_list, dtype=torch.float).unsqueeze(0).to(self.device),
                  'pan': torch.tensor(self.prev_pan_list, dtype=torch.float).div(450.0).unsqueeze(0).to(self.device),
                  'tilt': torch.tensor(self.prev_tilt_list, dtype=torch.float).div(450.0).unsqueeze(0).to(self.device),
                  'target_box': to_tensor(self.target_box).unsqueeze(0).to(self.device)
                  }

        # rospy.loginfo('prev_imgs: {}'.format(sample['prev_imgs'].size()))
        # rospy.loginfo('commands_dx: {}'.format(sample['commands_dx'].size()))
        # rospy.loginfo('commands_dy: {}'.format(sample['commands_dy'].size()))
        # rospy.loginfo('pan: {}'.format(sample['pan'].size()))
        # rospy.loginfo('tilt: {}'.format(sample['tilt'].size()))
        # rospy.loginfo('target_box: {}'.format(sample['target_box'].size()))
        #
        rospy.loginfo_throttle(0.33, "tracking")

        # do your shit publish command
        with torch.no_grad():
            cmd_out = self.model.inference(sample)

        out_pan = (self.x_list[-1] + cmd_out[0,0].item()) * image_msg.ptu_state_of_this_frame.field_of_view_angle
        out_tilt = (self.y_list[-1] + cmd_out[0,1].item()) * image_msg.ptu_state_of_this_frame.field_of_view_angle * 9.0/16.0
        self.publish_command((out_pan, out_tilt))

        self.pop_all_lists()

    def select_target_command_callback(self, user_input_msg):
        if user_input_msg.command != "target_reinit":
            return
        if len(self.detection_msg.detections) == 0:
            rospy.logwarn("There were no detections in the last frame, wait for next one and click again")
            return

        clicked_point_x = (user_input_msg.data[0] + user_input_msg.data[2]) / 2.0
        clicked_point_y = (user_input_msg.data[1] + user_input_msg.data[3]) / 2.0
        clicked_point_xy = (clicked_point_x, clicked_point_y)

        # find clicked box
        min_distance = 1.5 # just an initial large value
        for i in range(len(self.detection_msg.detections)):
            detection = self.detection_msg.detections[i]

            box_centre_x = (detection.cvBox.xmin + detection.cvBox.xmax) / 2.0
            box_centre_y = (detection.cvBox.ymin + detection.cvBox.ymax) / 2.0

            distance = l2norm(clicked_point_xy, (box_centre_x, box_centre_y))

            if distance < min_distance:
                min_distance = distance
                target_index = i

        # extract box
        target_detection = self.detection_msg.detections[target_index]

        filePath = "/dev/shm/" + self.detection_msg.gpu_frame.share_path
        if filePath in self.memoryMappedBuffersDict:
            buf = self.memoryMappedBuffersDict[filePath]
        else:
            fd = os.open(filePath, os.O_RDONLY)
            buf = mmap.mmap(fd, 540*960*3, mmap.MAP_SHARED, mmap.PROT_READ)
            self.memoryMappedBuffersDict[filePath] = buf
        img = np.frombuffer(buf, np.uint8).reshape((540,960,3))

        im_h, im_w = img.shape[:2]
        assert im_w > im_h

        target_bbox = target_detection.cvBox

        x = int(target_bbox.xmin * im_w  + 0.5)
        y = int(target_bbox.ymin * im_h + 0.5)
        w = int((target_bbox.xmax - target_bbox.xmin) * im_w  + 0.5)
        h = int((target_bbox.ymax - target_bbox.ymin) * im_h + 0.5)

        target_crop = img[y:y+h, x:x+w, :]
        target_crop = cv2.resize(target_crop, (104, 80), interpolation=cv2.INTER_AREA)
        cv2.imwrite("repos/target_box_{}.png".format(random.random()), target_crop)
        self.target_box = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)

    def sim_control_cb(self, sim_control_msg):
        self.control_mode = sim_control_msg.command

    def detections_cb(self, detection_msg):
        self.detection_msg = detection_msg

    def ptu_state_cb(self, ptu_state_msg):
        self.last_ptu_state_msg = ptu_state_msg

    def pop_all_lists(self):
        self.prev_imgs_list.pop(0)
        self.x_list.pop(0)
        self.y_list.pop(0)
        self.prev_tilt_list.pop(0)
        self.prev_pan_list.pop(0)

    def publish_command(self, out_pan_tilt):
        state_msg_sim = self.last_ptu_state_msg
        state_msg_sim.header.stamp = rospy.get_rostime()
        state_msg_sim.pan_position = out_pan_tilt[0]
        state_msg_sim.tilt_position = out_pan_tilt[1]
        self.pub_ptu_state.publish(state_msg_sim)

def l2norm(a, b):
    accum = 0.0
    for i in range(len(a)):
        temp = a[i] - b[i]
        accum += temp **2
    return np.sqrt(accum)

if __name__ == '__main__':
    rospy.init_node('encoder_ros')
    encoder_ros = EncoderRos()
    rospy.spin()
