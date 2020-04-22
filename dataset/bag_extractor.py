import cv2
import os
import sys
sys.path.insert(0,os.getcwd())
import rosbag
import numpy as np
import dataset.utils_dataset as utils_dataset

class BagExtractor:
    def __init__(self, bag_name, series_number, desired_fps, bag_fps=30, read_cv_box=False):
        # record dataset using this command:
        # rosbag record /carla/camera/rgb/camera01/image_color /carla/camera/rgb/camera01/camera_info /camera/960x540_BGR3/image_shared /driver/ptu_state /computer_vision/target_projected /carla/camera/semantic_segmentation/camera_semseg/image_segmentation /carla/camera/semantic_segmentation/camera_semseg/camera_info
        self.series_number = series_number #string
        self.dataset_dir = "Mthesis/database/my_database"
        self.dataset_folder = "/datasets/predictor_ds/"
        # self.dataset_folder = "encoder_ds/"
        self.notes = "No notes were specified"
        self.desired_fps = desired_fps
        self.bag_fps = bag_fps
        self.bag = rosbag.Bag(self.dataset_dir + "/bags/" + bag_name)
        self.read_cv_box = read_cv_box

        self.time_xy_ptfov_dict = None
        self.time_xy_ptfov_dict_ss = None
        self.time_img_dict = None
        self.time_img_dict_ss = None

        self.time_img_dxdy_ptfov_box_boximg_dict = None
        self.time_img_dxdy_ptfov_ss_dict = None
        self.time_box_dict = None

    def set_bag_name(self, bag_name):
        self.bag = rosbag.Bag(self.dataset_dir + "/bags/" + bag_name)

    def set_notes(self, notes):
        self.notes = notes

    def set_series_number(self, series_number):
        self.series_number = series_number

    def red_bag(self):
        # ptu state of each frame
        topic = '/carla/camera/rgb/camera01/camera_info'
        time_pan_tilt_fov_dict = utils_dataset.time_pan_tilt_fov_from_camera_info(self.bag, topic)
        topic = '/carla/camera/semantic_segmentation/camera_semseg/camera_info'
        time_pan_tilt_fov_dict_ss = utils_dataset.time_pan_tilt_fov_from_camera_info(self.bag, topic)
        
        x_list, y_list = utils_dataset.pt_to_xy(time_pan_tilt_fov_dict["pan"], time_pan_tilt_fov_dict["tilt"], np.median(time_pan_tilt_fov_dict["fov"]))
        self.time_xy_ptfov_dict = {
            "time": time_pan_tilt_fov_dict["time"],
            "x": x_list,
            "y": y_list,
            "pan": time_pan_tilt_fov_dict["pan"],
            "tilt": time_pan_tilt_fov_dict["tilt"],
            "fov": time_pan_tilt_fov_dict["fov"],
        }

        x_list, y_list = utils_dataset.pt_to_xy(time_pan_tilt_fov_dict_ss["pan"], time_pan_tilt_fov_dict_ss["tilt"], np.median(time_pan_tilt_fov_dict_ss["fov"]))
        self.time_xy_ptfov_dict_ss = {
            "time": time_pan_tilt_fov_dict_ss["time"],
            "x": x_list,
            "y": y_list,
            "pan": time_pan_tilt_fov_dict_ss["pan"],
            "tilt": time_pan_tilt_fov_dict_ss["tilt"],
            "fov": time_pan_tilt_fov_dict_ss["fov"],
        }
        
        topic = '/carla/camera/rgb/camera01/image_color'
        self.time_img_dict = utils_dataset.time_img_from_sensor_img(self.bag, topic)
        topic = '/carla/camera/semantic_segmentation/camera_semseg/image_segmentation'
        self.time_img_dict_ss = utils_dataset.time_img_from_sensor_img(self.bag, topic)

        topic = '/computer_vision/target_projected'
        self.time_box_dict = utils_dataset.time_box_from_projector(self.bag, topic)

        self.bag.close()
    def do_some_checks(self):
        # check if two successive images are the same or have the same time stamps (maybe simulations can't keep up with FPS and publishes same image)
        count = 0
        img_prev = self.time_img_dict["img"][10] #random first entry
        for img in self.time_img_dict["img"]:
            if not np.any(cv2.subtract(img_prev, img)):
                count += 1
            img_prev = img
        if count > 0:
            print("Two equal consecutive images were observed ", count, " times")

        count = 0
        img_prev = self.time_img_dict_ss["img"][10] #random first entry
        for img in self.time_img_dict_ss["img"]:
            if not np.any(cv2.subtract(img_prev, img)):
                count += 1
            img_prev = img
        if count > 0:
            print("Two equal consecutive images for semseg were observed ", count, " times")

        # check if fov changes in data recorded, check if it is the same as in ptu_state
        initial_fov = self.time_xy_ptfov_dict["fov"][0]
        for fov in self.time_xy_ptfov_dict["fov"]:
            if fov != initial_fov or fov != self.time_xy_ptfov_dict_ss["fov"][0]:
                print("FOV in camera_info changed from ", initial_fov, " to ", fov)
        
        # check if fov changes in data recorded, check if it is the same as in ptu_state
        initial_fov = self.time_xy_ptfov_dict_ss["fov"][0]
        for fov in self.time_xy_ptfov_dict_ss["fov"]:
            if fov != initial_fov:
                print("FOV in camera_info changed from ", initial_fov, " to ", fov)

    def generate_dict(self):
        time_box_boximg_dict = utils_dataset.match_extract_box_img(self.time_box_dict, self.time_img_dict)

        #find bounds for time number so that all the time numbers in between are present in both image and x_y arrays
        min_time = self.time_img_dict["time"][0]
        max_time = self.time_img_dict["time"][-1]
        min_time_ss = self.time_img_dict_ss["time"][0]
        max_time_ss = self.time_img_dict_ss["time"][-1]

        if self.read_cv_box:
            min_time_box = self.time_box_dict["time"][0]
            max_time_box = self.time_box_dict["time"][-1]
            min_time = max(min_time, min_time_ss, min_time_box)
            max_time = min(max_time, max_time_ss, max_time_box)
            min_time_boximg = time_box_boximg_dict["time"].index(min_time)
            max_time_boximg = time_box_boximg_dict["time"].index(max_time)
        else:
            min_time = max(min_time, min_time_ss)
            max_time = min(max_time, max_time_ss)
            min_time_boximg = 0 # generates empty lists
            max_time_boximg = 0


        min_time_index = self.time_img_dict["time"].index(min_time)
        max_time_index = self.time_img_dict["time"].index(max_time)
        min_time_index_ss = self.time_img_dict_ss["time"].index(min_time)
        max_time_index_ss = self.time_img_dict_ss["time"].index(max_time)

        time_img_xy_ptfov_box_boximg_dict = {
            "time": [x - min_time for x in self.time_img_dict["time"][min_time_index:max_time_index]], # make time start from 0
            "img": self.time_img_dict["img"][min_time_index:max_time_index],
            "x": self.time_xy_ptfov_dict["x"][min_time_index:max_time_index],
            "y": self.time_xy_ptfov_dict["y"][min_time_index:max_time_index],
            "pan": self.time_xy_ptfov_dict["pan"][min_time_index:max_time_index],
            "tilt": self.time_xy_ptfov_dict["tilt"][min_time_index:max_time_index],
            "fov": self.time_xy_ptfov_dict["fov"][min_time_index:max_time_index],
            "box": time_box_boximg_dict["box"][min_time_boximg:max_time_boximg],
            "box_img": time_box_boximg_dict["box_img"][min_time_boximg:max_time_boximg]
        }
        time_img_xy_ptfov_ss_dict = {
            "time": [x - min_time for x in self.time_img_dict_ss["time"][min_time_index_ss:max_time_index_ss]], # make time start from 0
            "img": self.time_img_dict_ss["img"][min_time_index_ss:max_time_index_ss],
            "x": self.time_xy_ptfov_dict_ss["x"][min_time_index_ss:max_time_index_ss],
            "y": self.time_xy_ptfov_dict_ss["y"][min_time_index_ss:max_time_index_ss],
            "pan": self.time_xy_ptfov_dict_ss["pan"][min_time_index_ss:max_time_index_ss],
            "tilt": self.time_xy_ptfov_dict_ss["tilt"][min_time_index_ss:max_time_index_ss],
            "fov": self.time_xy_ptfov_dict_ss["fov"][min_time_index_ss:max_time_index_ss],
        }
        assert len(time_img_xy_ptfov_box_boximg_dict["time"]) == len(time_img_xy_ptfov_box_boximg_dict["img"]) == len(time_img_xy_ptfov_box_boximg_dict["x"]) ==\
               len(time_img_xy_ptfov_box_boximg_dict["y"]) == len(time_img_xy_ptfov_box_boximg_dict["pan"]) == len(time_img_xy_ptfov_box_boximg_dict["tilt"]) ==\
               len(time_img_xy_ptfov_box_boximg_dict["fov"]) == \
               len(time_img_xy_ptfov_ss_dict["time"]) == len(time_img_xy_ptfov_ss_dict["img"]) == len(time_img_xy_ptfov_ss_dict["x"]) == \
               len(time_img_xy_ptfov_ss_dict["y"]) == len(time_img_xy_ptfov_ss_dict["pan"]) == len(time_img_xy_ptfov_ss_dict["tilt"]) == \
               len(time_img_xy_ptfov_ss_dict["fov"])

        if self.read_cv_box:
            assert len(time_img_xy_ptfov_box_boximg_dict["box_img"]) == len(time_img_xy_ptfov_box_boximg_dict["box"]) == len(time_img_xy_ptfov_box_boximg_dict["time"])
        
        time_img_xy_ptfov_box_boximg_dict = utils_dataset.decrease_fps(time_img_xy_ptfov_box_boximg_dict, self.bag_fps, self.desired_fps)
        time_img_xy_ptfov_ss_dict = utils_dataset.decrease_fps(time_img_xy_ptfov_ss_dict, self.bag_fps, self.desired_fps)
        assert len(time_img_xy_ptfov_box_boximg_dict["time"]) == len(time_img_xy_ptfov_box_boximg_dict["img"]) == len(time_img_xy_ptfov_box_boximg_dict["x"]) == \
               len(time_img_xy_ptfov_box_boximg_dict["y"]) == len(time_img_xy_ptfov_box_boximg_dict["pan"]) == len(time_img_xy_ptfov_box_boximg_dict["tilt"]) == \
               len(time_img_xy_ptfov_box_boximg_dict["fov"]) == \
               len(time_img_xy_ptfov_ss_dict["time"]) == len(time_img_xy_ptfov_ss_dict["img"]) == len(time_img_xy_ptfov_ss_dict["x"]) == \
               len(time_img_xy_ptfov_ss_dict["y"]) == len(time_img_xy_ptfov_ss_dict["pan"]) == len(time_img_xy_ptfov_ss_dict["tilt"]) == \
               len(time_img_xy_ptfov_ss_dict["fov"])
        
        ## time, x, y, pan, tilt, fov --> time, dx, dy, pan, tilt, fov
        dx_list = []
        dy_list = []
        for i in range(len(time_img_xy_ptfov_box_boximg_dict["x"]) - 1):
            dx_list.append(time_img_xy_ptfov_box_boximg_dict["x"][i + 1] - time_img_xy_ptfov_box_boximg_dict["x"][i])
            dy_list.append(time_img_xy_ptfov_box_boximg_dict["y"][i + 1] - time_img_xy_ptfov_box_boximg_dict["y"][i])
        self.time_img_dxdy_ptfov_box_boximg_dict = {
            "time": time_img_xy_ptfov_box_boximg_dict["time"],
            "img": time_img_xy_ptfov_box_boximg_dict["img"],
            "dx": dx_list,
            "dy": dy_list,
            "pan": time_img_xy_ptfov_box_boximg_dict["pan"],
            "tilt": time_img_xy_ptfov_box_boximg_dict["tilt"],
            "fov": time_img_xy_ptfov_box_boximg_dict["fov"],
            "box": time_img_xy_ptfov_box_boximg_dict["box"],
            "box_img": time_img_xy_ptfov_box_boximg_dict["box_img"]
        }

        ## time, x, y, pan, tilt, fov --> time, dx, dy, pan, tilt, fov
        dx_list = []
        dy_list = []
        for i in range(len(time_img_xy_ptfov_ss_dict["x"]) - 1):
            dx_list.append(time_img_xy_ptfov_ss_dict["x"][i + 1] - time_img_xy_ptfov_ss_dict["x"][i])
            dy_list.append(time_img_xy_ptfov_ss_dict["y"][i + 1] - time_img_xy_ptfov_ss_dict["y"][i])
        self.time_img_dxdy_ptfov_ss_dict = {
            "time": time_img_xy_ptfov_ss_dict["time"],
            "img": time_img_xy_ptfov_ss_dict["img"],
            "dx": dx_list,
            "dy": dy_list,
            "pan": time_img_xy_ptfov_ss_dict["pan"],
            "tilt": time_img_xy_ptfov_ss_dict["tilt"],
            "fov": time_img_xy_ptfov_ss_dict["fov"],
        }

        #dx and dy should have one less element since the laste img of the sequence does not have a corresponding command (it is just a label)
        assert len(self.time_img_dxdy_ptfov_box_boximg_dict["time"]) == len(self.time_img_dxdy_ptfov_box_boximg_dict["img"]) == len(self.time_img_dxdy_ptfov_box_boximg_dict["dx"]) + 1 == \
               len(self.time_img_dxdy_ptfov_box_boximg_dict["dy"]) + 1 == len(self.time_img_dxdy_ptfov_box_boximg_dict["pan"]) == len(self.time_img_dxdy_ptfov_box_boximg_dict["tilt"]) == \
               len(self.time_img_dxdy_ptfov_box_boximg_dict["fov"]) == \
               len(self.time_img_dxdy_ptfov_ss_dict["time"]) == len(self.time_img_dxdy_ptfov_ss_dict["img"]) == len(self.time_img_dxdy_ptfov_ss_dict["dx"]) + 1 == \
               len(self.time_img_dxdy_ptfov_ss_dict["dy"]) + 1 == len(self.time_img_dxdy_ptfov_ss_dict["pan"]) == len(self.time_img_dxdy_ptfov_ss_dict["tilt"]) == \
               len(self.time_img_dxdy_ptfov_ss_dict["fov"])

        # this assert is not necessarily true, but for curiosity leave it
        assert self.time_img_dxdy_ptfov_ss_dict["time"] == self.time_img_dxdy_ptfov_box_boximg_dict["time"]


    def save_dict(self):
        series_folder = self.dataset_dir + self.dataset_folder + "seq_" + self.series_number
        img_folder =  series_folder + "/imgs"
        if os.path.exists(series_folder):
            sys.exit("A folder for this series number already exists. Program terminates to avoid overriding data")
        try:
            os.makedirs(img_folder)
        except OSError:
            print ("Creation of the directory %s failed" % img_folder)

        out_img_size_wh = (263, 200)
        img_shape_hw = self.time_img_dxdy_ptfov_box_boximg_dict["img"][0].shape
        resize_ratio_wh = (out_img_size_wh[0]/img_shape_hw[1], out_img_size_wh[1]/img_shape_hw[0])

        img_name_list = []
        for i in range(len(self.time_img_dxdy_ptfov_box_boximg_dict["img"])):
            img_name = img_folder + "/" + self.series_number + "_%05d.png" % i
            img_name_list.append(img_name)
            cv2.imwrite(img_name, cv2.resize(self.time_img_dxdy_ptfov_box_boximg_dict["img"][i], out_img_size_wh, interpolation=cv2.INTER_AREA))
        for i in range(len(self.time_img_dxdy_ptfov_box_boximg_dict["box_img"])):
            patch_size_hw = self.time_img_dxdy_ptfov_box_boximg_dict["box_img"][i].shape
            out_size_patch_wh = (int(round(resize_ratio_wh[0] * patch_size_hw[1])), int(round(resize_ratio_wh[1] * patch_size_hw[0])))
            cv2.imwrite(img_folder + "/" + self.series_number + "_target_box_%05d.png" % i, cv2.resize(self.time_img_dxdy_ptfov_box_boximg_dict["box_img"][i], out_size_patch_wh, interpolation=cv2.INTER_AREA))

        with open(series_folder + "/dx_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (dx, img_name) for dx, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["dx"], img_name_list[:-1]))
        with open(series_folder + "/dy_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (dy, img_name) for dy, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["dy"], img_name_list[:-1]))
        with open(series_folder + "/time_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (time, img_name) for time, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["time"], img_name_list))
        with open(series_folder + "/pan_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (pan, img_name) for pan, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["pan"], img_name_list))
        with open(series_folder + "/tilt_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (tilt, img_name) for tilt, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["tilt"], img_name_list))
        with open(series_folder + "/fov_list.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (fov, img_name) for fov, img_name in zip(self.time_img_dxdy_ptfov_box_boximg_dict["fov"], img_name_list))

        with open(series_folder + "/box_xmin_list.txt", 'w') as filehandle:
            filehandle.writelines("%s\n" % box[0] for box in self.time_img_dxdy_ptfov_box_boximg_dict["box"])
        with open(series_folder + "/box_ymin_list.txt", 'w') as filehandle:
            filehandle.writelines("%s\n" % box[1] for box in self.time_img_dxdy_ptfov_box_boximg_dict["box"])
        with open(series_folder + "/box_xmax_list.txt", 'w') as filehandle:
            filehandle.writelines("%s\n" % box[2] for box in self.time_img_dxdy_ptfov_box_boximg_dict["box"])
        with open(series_folder + "/box_ymax_list.txt", 'w') as filehandle:
            filehandle.writelines("%s\n" % box[3] for box in self.time_img_dxdy_ptfov_box_boximg_dict["box"])


        img_name_list = []
        for i in range(len(self.time_img_dxdy_ptfov_ss_dict["img"])):
            img_name = img_folder + "/" + self.series_number + "_ss_%05d.png" % i
            img_name_list.append(img_name)
            cv2.imwrite(img_name, cv2.resize(self.time_img_dxdy_ptfov_ss_dict["img"][i], out_img_size_wh, interpolation=cv2.INTER_NEAREST))

        with open(series_folder + "/dx_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (dx, img_name) for dx, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["dx"], img_name_list[:-1]))
        with open(series_folder + "/dy_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (dy, img_name) for dy, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["dy"], img_name_list[:-1]))
        with open(series_folder + "/time_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (time, img_name) for time, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["time"], img_name_list))
        with open(series_folder + "/pan_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (pan, img_name) for pan, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["pan"], img_name_list))
        with open(series_folder + "/tilt_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (tilt, img_name) for tilt, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["tilt"], img_name_list))
        with open(series_folder + "/fov_list_ss.txt", 'w') as filehandle:
            filehandle.writelines("%s, %s\n" % (fov, img_name) for fov, img_name in zip(self.time_img_dxdy_ptfov_ss_dict["fov"], img_name_list))

        with open(series_folder + "/notes.txt", 'w') as filehandle:
            lines = [self.notes, "this series contains %05d frames with relative data" % len(self.time_img_dxdy_ptfov_ss_dict["time"])]
            filehandle.writelines("%s\n" % element for element in lines)

    def extract_dict_from_bag(self):
        self.red_bag()
        self.do_some_checks()
        self.generate_dict()
        self.save_dict()

if __name__ == "__main__":
    bag_extractor = BagExtractor(bag_name="pt_only_ss_335.bag", series_number=None, desired_fps=5, bag_fps=30)
    for root, dirs, files in os.walk(bag_extractor.dataset_dir + "/bags/"):
        for filename in sorted(files):
            if "pt_only_ss_" in filename:
                if int(filename[-7:-4]) >= 0:
                    print("Analysing: ", filename)
                    bag_extractor.set_bag_name(filename)
                    first_series_n = 1
                    series_number = "{:03d}".format(int(filename[-7:-4])+first_series_n - 1 )
                    bag_extractor.set_series_number(series_number)
                    bag_extractor.set_notes("5fps, pt_only, date: ~ 07/03, all data recorded in town03")
                    bag_extractor.extract_dict_from_bag()
