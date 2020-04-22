import torch
import random
import cv2
import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils_e2e import tensor_to_cv2
from dataset.utils_dataset import is_large_box, is_central_box, CropPatch
from PIL import Image

class FETripletDataset(Dataset):

    def __init__(self, data_dir, n_of_negatives=5, return_negative_list=False):
        self.data_dir = data_dir
        self.transform = CropPatchToTensor()
        self.n_of_negatives = n_of_negatives
        self.original_img_h = 120
        self.original_img_w = 158
        self.min_patch_size = 25

        self.return_negative_list = return_negative_list #sanity check mode for visualization
        if self.return_negative_list:
            self.n_of_desired_negatives = 5
            assert n_of_negatives >= self.n_of_desired_negatives

        self.boxes_list, self.img_box_path_list = self.load_dataset()


    def load_dataset(self):
        # Read in the files and build a list of boxes and corresponding paths
        boxes_file_path = self.data_dir + "/detector_boxes.txt"

        boxes_list = []
        img_box_path_list = []

        with open(boxes_file_path) as boxes_file:
            line = boxes_file.readline()
            while line:
                boxes, img_box_path = line.split("--")
                # if there are boxes
                if len(boxes) > 0:
                    boxes_coords = boxes.split(", ")[:-1]
                    box = []
                    for coord in boxes_coords:
                        box.append(float(coord))
                        if len(box) == 4 :
                            is_large_box_ = is_large_box(box, (self.original_img_h, self.original_img_w), self.min_patch_size)
                            is_central_box_ = is_central_box(box, (self.original_img_h, self.original_img_w), edge_thresh=0.02)
                            if is_large_box_ and is_central_box_:
                                boxes_list.append(box)
                                img_box_path_list.append(self.data_dir + img_box_path[-23:])
                            box = []

                line = boxes_file.readline()

        return boxes_list, img_box_path_list


    def __len__(self):
        return len(self.img_box_path_list)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        box = self.boxes_list[idx]

        anchor_name = self.img_box_path_list[idx][:-1]
        anchor_img = np.array(Image.open(anchor_name).convert("RGB"))
        assert self.original_img_h, self.original_img_w == anchor_img.shape[:2]

        positive_name = anchor_name[:-10] + "apositive.png"
        positive_img = np.array(Image.open(positive_name).convert("RGB"))

        if not self.return_negative_list:
            random_neg_idx = random.randint(0, self.n_of_negatives-1)
            negative_name = anchor_name[:-10] + "negative_%03d.png" %random_neg_idx
            negative_img = np.array(Image.open(negative_name).convert("RGB"))

            sample = {'anchor_img': anchor_img, 'positive_img': positive_img, 'negative_img': negative_img}

        else:
            prev_random_idx = []
            negative_list = []
            for _ in range(self.n_of_desired_negatives):
                random_neg_idx = random.randint(0, self.n_of_negatives-1)
                while random_neg_idx in prev_random_idx:
                    # if we already chose this sample, extract another one
                    random_neg_idx = random.randint(0, self.n_of_negatives-1)
                prev_random_idx.append(random_neg_idx)

                negative_name = anchor_name[:-10] + "negative_%03d.png" %random_neg_idx
                negative_img = np.array(Image.open(negative_name).convert("RGB"))
                negative_list.append(negative_img)

            sample = {'anchor_img': anchor_img, 'positive_img': positive_img, 'negative_img_list': negative_list}


        sample = self.transform(sample, box)
        return sample


class CropPatchToTensor(object):
    # crop patch at box location, for both anchor, pos and neg
    # resize cropped patch to net input size (interpolate)
    # convert to tensor, (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # divide by 255
    # --> data is in range [0, 1]

    def __init__(self):
        # try to keep initial img ratio here (120/158)
        self.out_h = 64
        self.out_w = 80

    def __call__(self, sample, box_coords):
        assert 0.0 <= box_coords[0] <= box_coords[2] <= 1.0 and 0.0 <= box_coords[1] <= box_coords[3] <= 1.0
        if 'negative_img' in sample:
            assert sample['anchor_img'].shape == sample['positive_img'].shape == sample['negative_img'].shape
        elif 'negative_img_list' in sample:
            assert sample['anchor_img'].shape == sample['positive_img'].shape == sample['negative_img_list'][0].shape

        im_h, im_w = sample['anchor_img'].shape[:2]

        x1_crop = int(round(box_coords[0] * im_w))
        y1_crop = int(round(box_coords[1] * im_h))
        x2_crop = int(round(box_coords[2] * im_w))
        y2_crop = int(round(box_coords[3] * im_h))

        h_crop = y2_crop - y1_crop
        w_crop = x2_crop - x1_crop

        crop_anchor = CropPatch(out_size_hw=(h_crop, w_crop), top=box_coords[1], left=box_coords[0])
        delta = random.random() * 0.85
        crop_patch = CropPatch(out_size_hw=(h_crop, w_crop), top=box_coords[1], left=box_coords[0], delta=delta)
        to_tensor = transforms.ToTensor()
        transform_anchor = transforms.Compose([crop_anchor, to_tensor])
        transform_patch = transforms.Compose([crop_patch, to_tensor])


        if 'negative_img' in sample:
            transformed_sample = {'anchor': transform_anchor(sample['anchor_img']),
                                  'positive': transform_patch(sample['positive_img']),
                                  'negative': transform_patch(sample['negative_img'])}

        elif 'negative_img_list' in sample:
            negative_list = []
            for negative_img in sample['negative_img_list']:
                negative_list.append(transform_patch(negative_img))

            pred_positive_img = np.copy(sample['positive_img'])

            transformed_sample = {'anchor': transform_anchor(sample['anchor_img']),
                                  'positive': transform_patch(sample['positive_img']),
                                  'negative_list': negative_list,
                                  'pred_positive_img': to_tensor(pred_positive_img)}

        for key in transformed_sample:
            if key == 'anchor' or key == 'positive' or key == 'negative':
                # Input dimensions in the form: mini-batch x channels x [optional depth] x [optional height] x width
                transformed_sample[key] = torch.nn.functional.interpolate(transformed_sample[key].unsqueeze(0),
                                                                          size=(self.out_h, self.out_w),
                                                                          align_corners=False,
                                                                          mode='bilinear').squeeze(0)
            elif key == 'negative_list':
                for neg_idx in range(len(transformed_sample['negative_list'])):
                    # Input dimensions in the form: mini-batch x channels x [optional depth] x [optional height] x width
                    transformed_sample['negative_list'][neg_idx] = torch.nn.functional.interpolate(transformed_sample['negative_list'][neg_idx].unsqueeze(0),
                                                                                                   size=(self.out_h, self.out_w),
                                                                                                   align_corners=False,
                                                                                                   mode='bilinear').squeeze(0)


        # add original patch size and patch location in original img
        transformed_sample['anchor_patch_original_size_hw'] = (h_crop, w_crop)
        transformed_sample['target_patch_location_tl'] = (y1_crop, x1_crop)

        # all tensors are of type Float
        return transformed_sample



if __name__ == "__main__":
    # this is just a test for the code

    data_dir = "Mthesis/database/my_database/datasets/fe_triplets_ds"
    dataset = FETripletDataset(data_dir)
    print("database length: ", len(dataset))
    for i in range(20):
        rand_idx = np.random.randint(0, len(dataset))
        sample_rand = dataset[rand_idx]
        for key in ['anchor', 'positive', 'negative']:
            assert torch.max(sample_rand[key]).item() <= 1 and torch.min(sample_rand[key]).item() >= 0
            img_open_cv = tensor_to_cv2(sample_rand[key].mul(255.))
            cv2.imwrite("Mthesis/database/my_database/FE_triplet_check/out_img_{}_{}.png".format(rand_idx, key), img_open_cv)

    print('anchor', sample_rand['anchor'].size())
    print('positive', sample_rand['positive'].size())
    print('negative', sample_rand['negative'].size())
    print('anchor_patch_original_size_hw', sample_rand['anchor_patch_original_size_hw'])



    # print()
    # print('FETripletSanityCheckDataset')
    # data_dir = "Mthesis/database/my_database/datasets/fe_triplets_val_ds"
    # dataset = FETripletDataset(data_dir, return_negative_list=True)
    # print("database length: ", len(dataset))
    # for i in range(20):
    #     sample_rand = dataset[np.random.randint(0, len(dataset))]
    # print('anchor', sample_rand['anchor'].size())
    # print('positive', sample_rand['positive'].size())
    # print('pred_positive_img', sample_rand['pred_positive_img'].size())
    # print('anchor_patch_original_size_hw', sample_rand['anchor_patch_original_size_hw'])
    # print('target_patch_location_tl', sample_rand['target_patch_location_tl'])
    # print('negative_list', "length {}, size of element {}".format(len(sample_rand['negative_list']), sample_rand['negative_list'][0].size()) )
    # assert torch.max(sample_rand['negative_list'][0]).item() <= 1 and torch.min(sample_rand['negative_list'][0]).item() >= 0
    # for key in ['anchor', 'positive', 'pred_positive_img']:
    #     assert torch.max(sample_rand[key]).item() <= 1 and torch.min(sample_rand[key]).item() >= 0
