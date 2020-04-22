import pylib.cvbridge as cvbridge
import math
import random
import numpy as np
import os
from PIL import Image

def time_pan_tilt_fov_from_camera_info(bag, topic_str, fix_pan_range=False):
    # returns out_arr = [time stamp, value] (shape N,2), out_time_att (shape N): time at which msg were received
    time_list = []
    pan_list = []
    tilt_list = []
    fov_list = []
    # D = [pan tilt fov width heigth]
    for topic, msg, t in bag.read_messages(topics = topic_str):
        time_list.append(msg.header.stamp.to_sec())
        pan_list.append(msg.D[0])
        tilt_list.append(msg.D[1])
        fov_list.append(msg.D[2])

    if fix_pan_range:
        for i in range(len(tilt_list)):
            if tilt_list[i] < -88:
                print("Remove this bag, tilting too much down")
                exit()
        for i in range(len(pan_list) -1):
            if pan_list[i+1] - pan_list[i] > 70.0:
                pan_list[i+1] -= 360.0
            elif pan_list[i+1] - pan_list[i] < - 70.0:
                pan_list[i+1] += 360.0

    # check if pan and tilt are smooth
    for i in range(len(pan_list)-1):
        if abs(tilt_list[i+1] - tilt_list[i]) > 5.0:
            print("TILT going from {} to {}, NOT SMOOTH!".format(tilt_list[i], tilt_list[i+1]))
            exit()
        if abs(pan_list[i+1] - pan_list[i]) > 5.0:
            print("PAN going from {} to {} at index {}, NOT SMOOTH!".format(pan_list[i], pan_list[i+1], i))
            exit()

    time_pan_tilt_fov_dict = {
        "time": time_list,
        "pan": pan_list,
        "tilt": tilt_list,
        "fov": fov_list,
    }
    return time_pan_tilt_fov_dict

def time_img_from_sensor_img(bag, topic_str):
    time_list = []
    img_list = []
    for topic, msg, t in bag.read_messages(topics = topic_str):
        time_list.append(msg.header.stamp.to_sec())
        img_list.append(cvbridge.imgmsg_to_cv2_bgr8(msg))

    time_img_dict = {
        "time": time_list,
        "img": img_list,
    }
    return time_img_dict

def time_box_from_projector(bag, topic_str):
    time_list = []
    box_list = []
    for topic, msg, t in bag.read_messages(topics = topic_str):
        time_list.append(msg.gpu_frame.sim_header.stamp.to_sec())
        target_idx = msg.target_index
        box = (msg.detections[target_idx].cvBox_tracker.xmin,
               msg.detections[target_idx].cvBox_tracker.ymin,
               msg.detections[target_idx].cvBox_tracker.xmax,
               msg.detections[target_idx].cvBox_tracker.ymax)
        box_list.append(box)

    time_box_dict = {
        "time": time_list,
        "box": box_list,
    }
    return time_box_dict

def match_extract_box_img(time_box_dict, time_img_dict):
    # match projector box with corresponding img, if no match, get the closest img
    # then crop img patch
    time_box_boximg_dict = {'time' : time_img_dict['time'],
                            'box' : [None] * len(time_img_dict['time']),
                            'box_img':[]}

    if len(time_box_dict["time"]) == 0:
        return time_box_boximg_dict

    min_time_img = time_img_dict["time"][0]
    min_time_box = time_box_dict["time"][0]

    # pad the list, these boxes will be removed anyway later
    if min_time_img < min_time_box:
        min_time_box_idx = time_img_dict["time"].index(min_time_box)
        for i in range(min_time_box_idx):
            time_box_boximg_dict['box'][i] = time_box_dict['box'][0]

    for i in range(len(time_img_dict['time'])):
        if time_img_dict['time'][i] < min_time_box:
            continue
        # look for time, if not found, look for the previous one
        index_found = False
        time_to_look_for = time_img_dict['time'][i]
        index = i
        while not index_found:
            try:
                box_index = time_box_dict['time'].index(time_to_look_for)
                index_found = True
            except ValueError:
                index -= 1
                assert index != i #check that value is copied, not pointed to
                time_to_look_for = time_img_dict['time'][index]
        time_box_boximg_dict['box'][i] = time_box_dict['box'][box_index]
    assert time_box_boximg_dict['box'].count(None) == 0

    # now we have a list of box (xmin, ymin, xmax, ymax) corresponding to the images. Let's crop the images!
    for idx in range(len(time_img_dict['img'])):
        # assert img size is (h, w, c)
        assert time_img_dict['img'][idx].shape[1] >= time_img_dict['img'][idx].shape[0]
        x_min = int(round(time_box_boximg_dict['box'][idx][0] * time_img_dict['img'][idx].shape[1]))
        y_min = int(round(time_box_boximg_dict['box'][idx][1] * time_img_dict['img'][idx].shape[0]))
        x_max = int(round(time_box_boximg_dict['box'][idx][2] * time_img_dict['img'][idx].shape[1]))
        y_max = int(round(time_box_boximg_dict['box'][idx][3] * time_img_dict['img'][idx].shape[0]))
        img_cropped =  time_img_dict['img'][idx][y_min:y_max, x_min:x_max, :]
        time_box_boximg_dict['box_img'].append(img_cropped)

    return time_box_boximg_dict

def pt_to_xy(pan, tilt, fov):
    x = pan / fov
    y = tilt / (fov * 9.0/16.0)
    return [x,y]


def decrease_fps(time_img_x_y_dict, bag_fps, desired_fps):
    # generate empty dict
    time_img_x_y_dict_reduced = {}
    for key in time_img_x_y_dict:
        time_img_x_y_dict_reduced[key] = []

    count = 0
    for element_idx in range(len(time_img_x_y_dict["time"])):
        if count % int(round(bag_fps/desired_fps)) != 0:
            count += 1
            continue

        for key in time_img_x_y_dict:
            if len(time_img_x_y_dict[key]) > 0: # accounts for when box_img is empty (we do not read cv proj)
                time_img_x_y_dict_reduced[key].append(time_img_x_y_dict[key][element_idx])

        count += 1

    return time_img_x_y_dict_reduced

def time_fov_from_camera_info(bag, topic_str):
    time_list = []
    fov_list = []
    img_w = 960.0
    for topic, msg, t in bag.read_messages(topics = topic_str):
        time_list.append(msg.header.stamp.to_sec())
        fov = math.atan(img_w / (2 * msg.P[0])) * 360.0 / math.pi
        fov_list.append(fov)

    time_fov_dict = {
        "time": time_list,
        "fov": fov_list,
    }

    # careful here: the time of the camera info msg is meaningless! not related to the other time stamps
    return time_fov_dict

def center_crop(tensor, out_size_hw):
    assert len(out_size_hw) == 2
    h_out = out_size_hw[0]
    w_out = out_size_hw[1]
    if len(tensor.size()) == 3:
        # tensor is C x H x W
        h = tensor.size()[1]
        w = tensor.size()[2]
        i_0 = (h-h_out)//2
        i_1 = i_0 + (h-h_out)%2
        j_0 = (w-w_out)//2
        j_1 = j_0 + (w-w_out)%2
        tensor_out = tensor[:, i_0:-i_1, j_0:-j_1]
        assert tensor_out.size()[1] == h_out and tensor_out.size()[2] == w_out
    elif len(tensor.size()) == 4:
        # tensor is N x C x H x W
        h = tensor.size()[2]
        w = tensor.size()[3]
        i_0 = (h-h_out)//2
        i_1 = i_0 + (h-h_out)%2
        j_0 = (w-w_out)//2
        j_1 = j_0 + (w-w_out)%2
        tensor_out = tensor[:, :, i_0:-i_1, j_0:-j_1]
        assert tensor_out.size()[2] == h_out and tensor_out.size()[3] == w_out
    else: return print("ERROR, WRONG TENSOR INPUT SIZE")

    return tensor_out

def crop_tensor(tensor, tl, out_size_hw):
    if len(tensor.size()) == 3:
        assert tl[0] + out_size_hw[0] <= tensor.size()[1] and tl[1] + out_size_hw[1] <= tensor.size()[2]
        tensor_out = tensor[:, tl[0]:tl[0] + out_size_hw[0], tl[1]:tl[1] + out_size_hw[1]]

    elif len(tensor.size()) == 4:
        assert tl[0] + out_size_hw[0] <= tensor.size()[2] and tl[1] + out_size_hw[1] <= tensor.size()[3]
        tensor_out = tensor[:, :, tl[0]:tl[0] + out_size_hw[0], tl[1]:tl[1] + out_size_hw[1]]

    else: return print("ERROR, WRONG TENSOR INPUT SIZE")

    return tensor_out

def img_to_ss(stacked_img):
    # img is in format (h,w,c)
    assert stacked_img.shape[0] < stacked_img.shape[1]
    #
    classes = {
        0: [0, 0, 0],         # None ok
        1: [70, 70, 70],      # Buildings ok
        2: [190, 153, 153],   # Fences ok
        3: [250, 170, 160],   # Other was [72, 0, 90]
        4: [220, 20, 60],     # Pedestrians ok
        5: [153, 153, 153],   # Poles ok
        6: [157, 234, 50],    # RoadLines ok
        7: [128, 64, 128],    # Roads ok
        8: [244, 35, 232],    # Sidewalks ok
        9: [107, 142, 35],    # Vegetation ok
        10: [0, 0, 142],      # Vehicles, was [0, 0, 255]
        11: [102, 102, 156],  # Walls ok
        12: [220, 220, 0]     # TrafficSigns ok
    }

    semseg = 15 * np.ones((stacked_img.shape[0], stacked_img.shape[1], stacked_img.shape[2]//3))

    for img_idx in range(stacked_img.shape[2]//3):
        img = stacked_img[:, :, img_idx*3 : 3* img_idx + 3]

        for key, value in classes.items():
            # for these keys we can just check the first value, probably faster
            if key != 0 and key != 4 and key != 10 and key != 12:
                indexes = np.asarray(img[:, :, 0] == value[0]).nonzero()
                semseg[indexes] = key
            else:
                indexes = np.asarray(img == value).all(axis=2).nonzero()
                semseg[indexes] = key

    assert len(semseg[semseg==15]) == 0

    return semseg


def ss_to_img(ss_img, out_chw=False):

    # img is in format (h,c)
    assert len(ss_img.shape) == 2 and ss_img.shape[0] <= ss_img.shape[1]

    classes = {
        0: [0, 0, 0],         # None ok
        1: [70, 70, 70],      # Buildings ok
        2: [190, 153, 153],   # Fences ok
        3: [250, 170, 160],   # Other was [72, 0, 90]
        4: [220, 20, 60],     # Pedestrians ok
        5: [153, 153, 153],   # Poles ok
        6: [157, 234, 50],    # RoadLines ok
        7: [128, 64, 128],    # Roads ok
        8: [244, 35, 232],    # Sidewalks ok
        9: [107, 142, 35],    # Vegetation ok
        10: [0, 0, 142],      # Vehicles, was [0, 0, 255]
        11: [102, 102, 156],  # Walls ok
        12: [220, 220, 0]     # TrafficSigns ok
    }

    img_rgb = 258 * np.ones((ss_img.shape[0], ss_img.shape[1],3))
    for key, value in classes.items():
        indexes = np.asarray(ss_img == key).nonzero()
        img_rgb[indexes] = value
    if out_chw:
        img_rgb = np.moveaxis(img_rgb, 2, 0)

    assert len(img_rgb[img_rgb==258]) == 0

    return img_rgb

def count_semseg_classes_dataset(data_dir):
    classes_count_per_pixel = np.zeros(13) # count/img_pixels_n
    classes_count_dataset = np.zeros(13) #final percentage
    n_of_imgs = 0

    classes = {
        0: [0, 0, 0],         # None ok
        1: [70, 70, 70],      # Buildings ok
        2: [190, 153, 153],   # Fences ok
        3: [250, 170, 160],   # Other was [72, 0, 90]
        4: [220, 20, 60],     # Pedestrians ok
        5: [153, 153, 153],   # Poles ok
        6: [157, 234, 50],    # RoadLines ok
        7: [128, 64, 128],    # Roads ok
        8: [244, 35, 232],    # Sidewalks ok
        9: [107, 142, 35],    # Vegetation ok
        10: [0, 0, 142],      # Vehicles, was [0, 0, 255]
        11: [102, 102, 156],  # Walls ok
        12: [220, 220, 0]     # TrafficSigns ok
    }

    # [0.01758005 0.0846384  0.0011241  0.00686975 0.00059465 0.01023665
    #  0.03058421 0.55855576 0.19537849 0.0188431  0.04627842 0.02788447
    #  0.00143195]
    # this is the frequency evaluated!

    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if "ss" in name and "png" in name:
                img_name = os.path.join(root, name)
                img = np.array(Image.open(img_name))
                pixels_n = float(img.shape[0] * img.shape[1])
                for key, value in classes.items():
                    # for these keys we can just check the first value, probably faster
                    if key != 0 and key != 4 and key != 10 and key != 12:
                        indexes = np.asarray(img[:, :, 0] == value[0]).nonzero()
                        assert len(indexes[0]) == len(indexes[1])
                    else:
                        indexes = np.asarray(img == value).all(axis=2).nonzero()
                        assert len(indexes[0]) == len(indexes[1])

                    classes_count_per_pixel[key] = float(len(indexes[0]) / pixels_n)

                    # cumulative moving average formula
                    classes_count_dataset[key] += (classes_count_per_pixel[key] - classes_count_dataset[key]) / (n_of_imgs + 1)
                n_of_imgs += 1
                assert np.allclose(np.sum(classes_count_per_pixel), 1.0)
                assert np.allclose(np.sum(classes_count_dataset), 1.0)



    return classes_count_dataset

def is_large_box(box, img_size_hw, min_patch_size):

    box_w = (box[2] - box[0]) * img_size_hw[1]
    box_h = (box[3] - box[1]) * img_size_hw[0]

    # # check what the min boxes allowed look like (just for debugging)
    # if box_w**2 + box_h**2 + 2 >= min_patch_size**2 >= box_w**2 + box_h**2 - 2:
    #     return True

    if box_w**2 + box_h**2 >= min_patch_size**2:
        return True
    else:
        return False

def is_central_box(box, img_size_hw, centre_max_relative_distance_h=0.5, edge_thresh=None):
    assert 0 <= centre_max_relative_distance_h <= 1

    # # check what the furthest boxes allowed look like (just for debugging)
    # thresh = 0.25
    # if  box[0] - 0.001<= thresh <= box[0]+0.001:
    #     return True
    # else:
    #     return False

    if edge_thresh is not None:
        assert edge_thresh >= 0.0
        if box[0] < edge_thresh or box[1] < edge_thresh or box[2] > 1-edge_thresh or box[3] > 1-edge_thresh:
            return False
        else:
            return True

    #centre_max_relative_distance is the max dist of the box centre relative to image height

    box_w = (box[2] - box[0]) * img_size_hw[1]
    box_h = (box[3] - box[1]) * img_size_hw[0]
    box_centre_w = box[0] * img_size_hw[1] + box_w / 2.0
    box_centre_h = box[1] * img_size_hw[0] + box_h / 2.0

    img_centre_w = img_size_hw[1]/2.0
    img_centre_h = img_size_hw[0]/2.0
    max_radius = (centre_max_relative_distance_h * img_size_hw[0])

    if (box_centre_h - img_centre_h)**2 + (box_centre_w - img_centre_w)**2 <= max_radius**2:
        return  True
    else:
        return False


class RandomCrop:
    # expects img in format (H, W, C)
    def __init__(self, out_size_hw, rand_x=random.random(), rand_y=random.random(), delta=0.0):
        assert 0.0 <= rand_x <= 1.0 and 0.0 <= rand_y <= 1.0 and 0.0 <= delta <= 1.0
        self.out_size_hw = out_size_hw
        self.rand_x = rand_x
        self.rand_y = rand_y
        self.delta = delta
    def __call__(self, sample):
        # check that sample is sent in format (H, W, C)
        assert sample.shape[0] <= sample.shape[1]
        # limit out size
        if self.out_size_hw[0] > sample.shape[0]:
            self.out_size_hw[0] = sample.shape[0]
        if self.out_size_hw[1] > sample.shape[1]:
            self.out_size_hw[1] = sample.shape[1]

        # delta is used to create a crop with the same patch centre location but a larger size
        # delta = 0 corresponds to no change, delta=1 correspond to creating the largest available patch crop
        available_gap_y = min(self.rand_y, 1-self.rand_y) * (sample.shape[0] - self.out_size_hw[0])
        available_gap_x = min(self.rand_x, 1-self.rand_x) * (sample.shape[1] - self.out_size_hw[1])
        delta_y = int(round(self.delta * available_gap_y))
        delta_x = int(round(self.delta * available_gap_x))

        y_1 = int(round(self.rand_y * (sample.shape[0] - self.out_size_hw[0]))) - delta_y
        y_2 = y_1 + self.out_size_hw[0] + 2 * delta_y
        x_1 = int(round(self.rand_x * (sample.shape[1] - self.out_size_hw[1]))) - delta_x
        x_2 = x_1 + self.out_size_hw[1] + 2 * delta_x
        cropped_img = sample[y_1:y_2, x_1:x_2, :]
        return cropped_img


class CropPatch:
    # expects img in format (H, W, C)
    def __init__(self, out_size_hw, top=random.random(), left=random.random(), delta=0.0):
        assert 0.0 <= top <= 1.0 and 0.0 <= top <= 1.0 and 0.0 <= delta <= 1.0
        self.out_size_hw = out_size_hw
        self.left = left
        self.top = top
        self.delta = delta
        self.original_img_h = 120
        self.original_img_w = 158
        self.min_patch_size = 30
    def __call__(self, sample):
        # check that sample is sent in format (H, W, C)
        assert sample.shape[0] <= sample.shape[1]
        # limit out size
        if self.out_size_hw[0] > sample.shape[0]:
            self.out_size_hw[0] = sample.shape[0]
        if self.out_size_hw[1] > sample.shape[1]:
            self.out_size_hw[1] = sample.shape[1]

        # delta is used to create a crop with the same patch centre location but a larger size
        # delta = 0 corresponds to no change,
        # delta=1 correspond to cropping the largest available patch up to twice original size
        available_img_gap_y = min(self.top, 1-self.top) * sample.shape[0]
        available_img_gap_x = min(self.left, 1-self.left) * sample.shape[1]
        available_gap_y = min(available_img_gap_y, self.out_size_hw[0] // 2)
        available_gap_x = min(available_img_gap_x, self.out_size_hw[1] // 2)
        delta_y = int(round(self.delta * available_gap_y))
        delta_x = int(round(self.delta * available_gap_x))

        y_1 = int(round(self.top * (sample.shape[0]))) - delta_y
        y_2 = y_1 + self.out_size_hw[0] + 2 * delta_y
        x_1 = int(round(self.left * (sample.shape[1]))) - delta_x
        x_2 = x_1 + self.out_size_hw[1] + 2 * delta_x
        cropped_img = sample[y_1:y_2, x_1:x_2, :]
        return cropped_img