import torch
import cv2
import os
import sys
sys.path.insert(0,os.getcwd())
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils_e2e import tensor_to_cv2
from dataset.utils_dataset import is_large_box, is_central_box, CropPatch, RandomCrop
import linecache

# note: extracted commands list also contains the latest command (the one we do inference for)
#       however, it is ignored it in the encoder.py

class EncoderDataset(Dataset):
    def __init__(self, data_dir, prev_img_number, generate_ground_truth=False):

        self.data_dir = data_dir
        self.prev_img_number = prev_img_number
        self.transform = ScaleResizeToTensor()
        self.generate_ground_truth = generate_ground_truth

        self.original_img_h = 200
        self.original_img_w = 263
        self.min_patch_size = 28 #25 for warm start, 28 for cold

        self.boxes_list, self.img_box_path_list, self.cmd_list, \
        self.pt_list, self.gt_cmd_list, self.heatmap_path_list = self.load_dataset()

    def load_dataset(self):

        boxes_list = []
        img_box_path_list = []
        cmd_list = []
        pt_list = []
        heatmap_path_list = []
        gt_cmd_list = []

        # boxes_list:
        #     boxes_in_sequence_list = [box, box, box ...] #one list for each seq_000, seq_001 ... one element for each data point (img sequence fed to net)
        #
        # with corresponding:
        # img_box_path_list:
        #     img_box_path_in_sequence_list
        #
        # similarly heatmap_path_list, gt_cmd_list

        # cmd_list:
        #     cmd_in_seq_list #one list for each seq_000, seq_001 ...
        #         prev_cmd_list = (dx,dy), (dx,dy)...] #len = prev_img_n
        #
        # similarly pt_list

        for root, dirs, files in os.walk(self.data_dir):
            for dir in sorted(dirs): # seq001, seq002..

                #each of these contain lists of boxes/paths
                boxes_in_sequence_list = [] #[ box1[] box1[] ]
                img_box_path_in_sequence_list = []  #[ path1[] path2[] ] path1 might be == to path2 if two good boxes in same img
                img_index_seq_list = [] # [ idx1, 2, 4, 4, 7] #used to then read corresponding cmd and pt

                boxes_file_path = os.path.join(root, dir) + "/detector_boxes.txt"
                with open(boxes_file_path) as boxes_file:
                    img_idx = 0
                    line = boxes_file.readline()
                    while line:
                        if img_idx < self.prev_img_number - 1:
                            # skip first indexes since they are part of the previous images (we only store the last img paths)
                            line = boxes_file.readline()
                            img_idx += 1
                            continue
                        boxes_saved_on_last_idx = 0
                        boxes, img_box_path = line.split("--")
                        # if there are boxes
                        if len(boxes) > 0:
                            boxes_coords = boxes.split(", ")[:-1]
                            box = []
                            for coord in boxes_coords:
                                box.append(float(coord))
                                if len(box) == 4 :
                                    # NOTE use same threshold values (for large/central) as the ones used when generating heatmaps!
                                    is_large_box_ = is_large_box(box, (self.original_img_h, self.original_img_w), self.min_patch_size)
                                    is_central_box_ = is_central_box(box, (self.original_img_h, self.original_img_w), edge_thresh=0.25) #for warm start use 0.15, cold start:0.25
                                    if is_large_box_ and is_central_box_:
                                        boxes_in_sequence_list.append(box)
                                        img_box_path_in_sequence_list.append(self.data_dir + img_box_path[-28:])
                                        img_index_seq_list.append(img_idx) #this is the line index corresponding to the images stored in img_box_path_in_sequence_list
                                        boxes_saved_on_last_idx += 1
                                    box = []

                        line = boxes_file.readline()
                        img_idx += 1

                #remove last element if we saved it, it is just the ground truth predictor img. We have no dx and dy for it!!
                # we need boxes_saved_on_last_idx because we might have saved multiple boxes for the last img, remove all of them
                if boxes_saved_on_last_idx > 0:
                    boxes_in_sequence_list = boxes_in_sequence_list[:-boxes_saved_on_last_idx]
                    img_box_path_in_sequence_list = img_box_path_in_sequence_list[:-boxes_saved_on_last_idx]
                    img_index_seq_list = img_index_seq_list[:-boxes_saved_on_last_idx]

                cmd_in_seq_list = []
                heatmap_path_in_seq_list = []
                pt_in_seq_list = []
                gt_cmd_in_seq_list = []

                dx_file_path = os.path.join(root, dir) + "/dx_list.txt"
                dy_file_path = os.path.join(root, dir) + "/dy_list.txt"
                gt_cmd_x_file_path = os.path.join(root, dir) + "/cmd_x_ground_truth.txt"
                gt_cmd_y_file_path = os.path.join(root, dir) + "/cmd_y_ground_truth.txt"
                pan_file_path = os.path.join(root, dir) + "/pan_list.txt"
                tilt_file_path = os.path.join(root, dir) + "/tilt_list.txt"
                for i in range(len(img_index_seq_list)):

                    prev_cmds_list = []
                    prev_pt_list = []

                    idx = img_index_seq_list[i]

                    # for each index we store the data relative to the previous images too
                    for prev_img_index in reversed(range(self.prev_img_number)): #4,3,2,1,0
                        line = linecache.getline(dx_file_path, idx+1 - prev_img_index)
                        dx, img_path_dx = line.split(", ")
                        line = linecache.getline(dy_file_path, idx+1 - prev_img_index)
                        dy, img_path_dy = line.split(", ")
                        prev_cmds_list.append((float(dx), float(dy)))

                        line = linecache.getline(pan_file_path, idx+1 - prev_img_index)
                        pan, img_path_pan = line.split(", ")
                        line = linecache.getline(tilt_file_path, idx+1 - prev_img_index)
                        tilt, img_path_tilt = line.split(", ")
                        prev_pt_list.append((float(pan), float(tilt)))

                    assert len(prev_cmds_list) == len(prev_pt_list) ==  self.prev_img_number
                    # this just checks the last img path
                    assert img_path_dx[-27:] == img_path_dy[-27:] == img_path_pan[-27:]== \
                           img_path_tilt[-27:] == img_box_path_in_sequence_list[i][-27:]

                    cmd_in_seq_list.append(prev_cmds_list)
                    pt_in_seq_list.append(prev_pt_list)

                    # DEPRECATED: we now do warm start using the bbox location
                    # if not self.generate_ground_truth:
                    #     # here we must read the i index, not idx!
                    #     # these data were created with heat map generator
                    #     # each val in gt_cmd_list correspond to a one of the boxes previously saved
                    #     line = linecache.getline(gt_cmd_x_file_path, i+1)
                    #     x_cmd, img_path = line.split(",")
                    #     assert img_path[:-23] + img_path[-11:] == img_box_path_in_sequence_list[i]
                    #     line = linecache.getline(gt_cmd_y_file_path, i+1)
                    #     y_cmd, img_path = line.split(",")
                    #     assert img_path[:-23] + img_path[-11:] == img_box_path_in_sequence_list[i]
                    #     gt_cmd_in_seq_list.append((float(x_cmd), float(y_cmd)))
                    #     heatmap_path_in_seq_list.append(img_path[:-11]+"_heat_map" + img_path[-11:-1])

                boxes_list.append(boxes_in_sequence_list)
                img_box_path_list.append(img_box_path_in_sequence_list)
                cmd_list.append(cmd_in_seq_list)
                pt_list.append(pt_in_seq_list)
                gt_cmd_list.append(gt_cmd_in_seq_list)
                heatmap_path_list.append(heatmap_path_in_seq_list)

            break #stop after first outer iter

        for seq_idx in range(len(boxes_list)):
            # assert that for each sequence we store the same number of elements
            assert len(boxes_list[seq_idx]) == len(img_box_path_list[seq_idx]) == len(cmd_list[seq_idx]) == len(pt_list[seq_idx])

        return boxes_list, img_box_path_list, cmd_list, pt_list, gt_cmd_list, heatmap_path_list


    def __len__(self):
        ds_length = 0
        for seq_paths in self.img_box_path_list:
            ds_length += len(seq_paths)
        return ds_length

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        #find in which sequence the element is
        element_passed = 0
        seq_index = 0
        idx_in_seq = idx
        for seq_paths in self.img_box_path_list:
            if idx >= len(seq_paths) + element_passed:
                element_passed += len(seq_paths)
                seq_index += 1
                idx_in_seq = idx - element_passed
            else:
                break
        assert idx_in_seq >= 0 and idx >= 0

        #read prev imgs
        prev_imgs_stacked = np.zeros((200, 263, 3 * self.prev_img_number), dtype=np.uint8)
        last_img_path = self.img_box_path_list[seq_index][idx_in_seq][:-1]
        last_img_number = int(last_img_path[-9:-4])

        #read commands etc..
        dx_list = []
        dy_list = []
        pan_list = []
        tilt_list = []

        for i in range(self.prev_img_number):
            img_path = last_img_path[:-9] + "%05d.png" %(last_img_number - self.prev_img_number + 1 + i)
            prev_imgs_stacked[:,:, 3*i : 3*i+3] = np.array(Image.open(img_path))

            assert last_img_number - self.prev_img_number + 1 + i >=  0
            dx_list.append(self.cmd_list[seq_index][idx_in_seq][i][0])
            dy_list.append(self.cmd_list[seq_index][idx_in_seq][i][1])
            pan_list.append(self.pt_list[seq_index][idx_in_seq][i][0])
            tilt_list.append(self.pt_list[seq_index][idx_in_seq][i][1])

        assert last_img_path == img_path #check that last img loaded is the one with the corresponding box here below
        box = self.boxes_list[seq_index][idx_in_seq] #corresponding to last img of the sequence

        sample = {'prev_imgs': prev_imgs_stacked, 'commands_dx': dx_list, 'commands_dy': dy_list, 'pan': pan_list, 'tilt': tilt_list}

        assert sample['prev_imgs'].shape[0] == self.original_img_h, sample['prev_imgs'].shape[1] == self.original_img_w

        if not self.generate_ground_truth:
            # box coords refer to uncropped last prev img. The img is actually randomly cropped, this is fixed in the transform
            sample['box'] = box
            sample['heat_map'] = torch.zeros(3, 160, 208)

            # DEPRECATED
            # to_tensor = transforms.ToTensor()
            # sample['cmd_x_gound_truth'] = self.gt_cmd_list[seq_index][idx_in_seq][0]
            # sample['cmd_y_gound_truth'] = self.gt_cmd_list[seq_index][idx_in_seq][1]
            # heat_map = Image.open(self.heatmap_path_list[seq_index][idx_in_seq])
            # sample['heat_map'] = to_tensor(heat_map)

        sample = self.transform(sample, box)
        # this is used for heatmap generation, to give the hm a name linking it to the source img + box
        sample['img_code_name'] = last_img_path[:-9] + "%05fbox_%05d.png" %(box[0], last_img_number)
        return sample

def extract_box_centre(box):
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    box_centre_x = box[0] + box_w / 2.0
    box_centre_y = box[1] + box_h / 2.0
    return box_centre_x, box_centre_y

class ScaleResizeToTensor(object):
    #! make sure that the same img transforms are applied to

    # resize to self.img_size
    # convert to tensor, (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # divide by 255
    # --> data is in range [-1, 1] with ~0 mean if we substract the mean, else it is in range [0,1]

    def __init__(self):
        self.random_cropped_size_hw = (160, 208)

    def __call__(self, sample, box_coords):
        assert 0.0 <= box_coords[0] <= box_coords[2] <= 1.0 and 0.0 <= box_coords[1] <= box_coords[3] <= 1.0

        im_h, im_w = sample['prev_imgs'][:,:,-3:].shape[:2]

        x1_crop = int(round(box_coords[0] * im_w))
        y1_crop = int(round(box_coords[1] * im_h))
        x2_crop = int(round(box_coords[2] * im_w))
        y2_crop = int(round(box_coords[3] * im_h))

        h_crop = y2_crop - y1_crop
        w_crop = x2_crop - x1_crop

        crop_transform = CropPatch(out_size_hw=(h_crop, w_crop), top=box_coords[1], left=box_coords[0])

        x_min_crop, x_max_crop, y_min_crop, y_max_crop = self.crop_limits(box_coords)

        # todo leave at 0.5 when using the heat map and ground truth
        rand_x = 0.5
        rand_y = 0.5
        rand_x = random.uniform(x_min_crop, x_max_crop)
        rand_y = random.uniform(y_min_crop, y_max_crop)

        random_crop = RandomCrop(out_size_hw=self.random_cropped_size_hw, rand_x=rand_x, rand_y=rand_y)
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([random_crop, to_tensor])

        # box coords refer to uncropped last prev img. The img is actually randomly cropped,
        # thus the resulting gt commands would be a bit wrong. Here we fix that
        original_h = sample['prev_imgs'].shape[0]
        original_w = sample['prev_imgs'].shape[1]
        box_centre_x, box_centre_y = extract_box_centre(sample['box'])
        box_centre_y = (box_centre_y * original_h - rand_y * (original_h - self.random_cropped_size_hw[0]))  / self.random_cropped_size_hw[0]
        cmd_y_gound_truth = - box_centre_y + 0.5
        box_centre_x = (box_centre_x * original_w - rand_x * (original_w - self.random_cropped_size_hw[1]))  / self.random_cropped_size_hw[1]
        cmd_x_gound_truth = - box_centre_x + 0.5

        #note: target_box is cropped BEFORE the prev_img is randomly cropped.
        # the resulting crop is fine, as long as car is not near the edges (should be safe with is_central_box) or cropped out
        target_box = crop_transform(sample['prev_imgs'][:,:,-3:])

        target_box_net = cv2.resize(target_box, (104, 80))
        target_box_loss = cv2.resize(target_box, (80, 64))

        transformed_sample = {'prev_imgs': transform(sample['prev_imgs']),
                              'commands_dx': torch.tensor(sample['commands_dx'], dtype=torch.float),
                              'commands_dy': torch.tensor(sample['commands_dy'], dtype=torch.float),
                              'pan': torch.tensor(sample['pan'], dtype=torch.float).div(450.0),
                              'tilt': torch.tensor(sample['tilt'], dtype=torch.float).div(450.0),
                              'cmd_x_gound_truth': cmd_x_gound_truth,
                              'cmd_y_gound_truth': cmd_y_gound_truth,
                              'heat_map': sample['heat_map'],
                              'target_box': to_tensor(target_box_net),
                              'target_box_loss': to_tensor(target_box_loss),
                              'target_box_size_hw': torch.tensor([h_crop, w_crop])}
        # all tensors are of type Float
        return transformed_sample

    def crop_limits(self, box):
        # find approx crop limits so that the box ends up at the centre of the frame!

        box_centre_x, box_centre_y = extract_box_centre(box)
        x_dist_from_centre = box_centre_x - 0.5 #range -0.5 to 0.5
        y_dist_from_centre = box_centre_y - 0.5

        x_min_crop = 0.0
        x_max_crop = 1.0
        y_min_crop = 0.0
        y_max_crop = 1.0

        factor = 3.0 #how much to restrict cropping, the large the more central the resulting box
        do_nothing_threshold = 0.0

        if abs(x_dist_from_centre) >= do_nothing_threshold:
            if x_dist_from_centre >= 0.0:
                x_min_crop = min(factor * x_dist_from_centre, 1.0)
            elif x_dist_from_centre <= 0.0:
                x_max_crop = max(1.0 + factor * x_dist_from_centre, 0.0)

        if abs(y_dist_from_centre) >= do_nothing_threshold:
            if y_dist_from_centre >= 0.0:
                y_min_crop = min(factor * y_dist_from_centre, 1.0)
            elif y_dist_from_centre <= 0.0:
                y_max_crop = max(1.0 + factor * y_dist_from_centre, 0.0)

        return x_min_crop, x_max_crop, y_min_crop, y_max_crop


if __name__ == "__main__":
    # this is just a test for the code

    data_dir = "Mthesis/database/my_database/datasets/encoder_ds"
    dataset = EncoderDataset(data_dir, prev_img_number=5, generate_ground_truth=False)
    print("database length: ", len(dataset))
    for i in range(25):
        rand = np.random.randint(0, len(dataset))
        sample_rand = dataset[rand]
        img_open_cv = tensor_to_cv2(sample_rand["prev_imgs"][-3:, :, :].mul(255.))
        cv2.imwrite("Mthesis/database/my_database/encoder_check/{}out_img_prev_img.png".format(rand), img_open_cv)
        img_open_cv = tensor_to_cv2(sample_rand["target_box"].mul(255.))
        cv2.imwrite("Mthesis/database/my_database/encoder_check/{}out_img_target.png".format(rand), img_open_cv)

    print('prev_imgs', sample_rand['prev_imgs'].size())
    print('commands_dx', sample_rand['commands_dx'].size())
    print('commands_dy', sample_rand['commands_dy'].size())
    print('pan', sample_rand['pan'].size())
    print('tilt', sample_rand['tilt'].size())
    print('target_box', sample_rand['target_box'].size())
    print('target_box_loss', sample_rand['target_box_loss'].size())
    print('target_box_size_hw', sample_rand['target_box_size_hw'])
    print('img_code_name', sample_rand['img_code_name'])
    print('cmd_x_gound_truth', sample_rand['cmd_x_gound_truth'])
    print('cmd_y_gound_truth', sample_rand['cmd_y_gound_truth'])
    print('heat_map', sample_rand['heat_map'].size())
    assert torch.max(sample_rand["prev_imgs"]).item() <= 1 and torch.min(sample_rand["prev_imgs"]).item() >= -1 \
        and torch.max(sample_rand["target_box"]).item() <= 1 and torch.min(sample_rand["target_box"]).item() >= -1 \
        and  torch.max(sample_rand["tilt"]).item() <= 1 and torch.min(sample_rand["tilt"]).item() >= -1
