import torch
import random
import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dataset.utils_dataset import center_crop, RandomCrop, img_to_ss, count_semseg_classes_dataset

class PredictorDataset(Dataset):
    def __init__(self, data_dir, prev_img_number, mean_rgb_list=None, semseg=False):
        if mean_rgb_list is None:
            # mean_rgb_list = [107, 110, 110]
            mean_rgb_list = [0, 0, 0]
        self.mean_rgb_list = mean_rgb_list
        self.data_dir = data_dir
        self.prev_img_number = prev_img_number
        self.semseg = semseg
        if semseg:
            self.transform = ScaleResizeToTensorSemSeg()
        else:
            self.transform = ScaleResizeToTensor(mean_rgb_list=mean_rgb_list)


    def __len__(self):
        count = self.count_data_length(self.data_dir)
        return count

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        imgs_counted_prev = 0
        prev_imgs_stacked = np.zeros((200, 263, 3 * self.prev_img_number), dtype=np.uint8)
        target_img = np.zeros((200,263,3), dtype=np.uint8)
        dx_list = []
        dy_list = []
        pan_list = []
        tilt_list = []
        img_path_list = []
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for dir_name in dirs:
                imgs_counted = self.count_data_length(os.path.join(self.data_dir, dir_name))
                if idx < imgs_counted_prev + imgs_counted:
                    for i in range(self.prev_img_number):
                        img_path = os.path.join(self.data_dir, dir_name) + "/imgs/" + dir_name[-3:] +  "_%05d.png" %(idx - imgs_counted_prev + i)
                        if self.semseg:
                            img_path = os.path.join(self.data_dir, dir_name) + "/imgs/" + dir_name[-3:] +  "_ss_%05d.png" %(idx - imgs_counted_prev + i)
                        img_path_list.append(img_path)
                        prev_imgs_stacked[:,:, 3*i : 3*i+3] = np.array(Image.open(img_path))

                    dx_file_path = os.path.join(self.data_dir, dir_name) + "/dx_list.txt"
                    if self.semseg:
                        dx_file_path = os.path.join(self.data_dir, dir_name) + "/dx_list_ss.txt"
                    with open(dx_file_path) as dx_file:
                        line = dx_file.readline()
                        count = 0
                        while line:
                            if count >= idx - imgs_counted_prev:
                                dx, img_path = line.split(", ")
                                assert img_path[-25:-1] == img_path_list[len(dx_list)][-24:]
                                dx_list.append(float(dx))
                                if len(dx_list) == self.prev_img_number:
                                    break
                            line = dx_file.readline()
                            count += 1

                    dy_file_path = os.path.join(self.data_dir, dir_name) + "/dy_list.txt"
                    if self.semseg:
                        dy_file_path = os.path.join(self.data_dir, dir_name) + "/dy_list_ss.txt"
                    with open(dy_file_path) as dy_file:
                        line = dy_file.readline()
                        count = 0
                        while line:
                            if count >= idx - imgs_counted_prev:
                                dy, img_path = line.split(", ")
                                assert img_path[-25:-1] == img_path_list[len(dy_list)][-24:]
                                dy_list.append(float(dy))
                                if len(dy_list) == self.prev_img_number:
                                    break
                            line = dy_file.readline()
                            count += 1

                    pan_file_path = os.path.join(self.data_dir, dir_name) + "/pan_list.txt"
                    if self.semseg:
                        pan_file_path = os.path.join(self.data_dir, dir_name) + "/pan_list_ss.txt"
                    with open(pan_file_path) as pan_file:
                        line = pan_file.readline()
                        count = 0
                        while line:
                            if count >= idx - imgs_counted_prev:
                                pan, img_path = line.split(", ")
                                assert img_path[-25:-1] == img_path_list[len(pan_list)][-24:]
                                pan_list.append(float(pan))
                                if len(pan_list) == self.prev_img_number:
                                    # add one more for target pan, used when reversing
                                    pan, img_path = pan_file.readline().split(", ")
                                    pan_list.append(float(pan))
                                    break
                            line = pan_file.readline()
                            count += 1

                    tilt_file_path = os.path.join(self.data_dir, dir_name) + "/tilt_list.txt"
                    if self.semseg:
                        tilt_file_path = os.path.join(self.data_dir, dir_name) + "/tilt_list_ss.txt"
                    with open(tilt_file_path) as tilt_file:
                        line = tilt_file.readline()
                        count = 0
                        while line:
                            if count >= idx - imgs_counted_prev:
                                tilt, img_path = line.split(", ")
                                assert img_path[-25:-1] == img_path_list[len(tilt_list)][-24:]
                                tilt_list.append(float(tilt))
                                if len(tilt_list) == self.prev_img_number:
                                    # add one more for target tilt, used when reversing
                                    tilt, img_path = tilt_file.readline().split(", ")
                                    tilt_list.append(float(tilt))
                                    break
                            line = tilt_file.readline()
                            count += 1

                    img_path = os.path.join(self.data_dir, dir_name) + "/imgs/" + dir_name[-3:] +  "_%05d.png" %(idx - imgs_counted_prev + self.prev_img_number)
                    if self.semseg:
                        img_path = os.path.join(self.data_dir, dir_name) + "/imgs/" + dir_name[-3:] +  "_ss_%05d.png" %(idx - imgs_counted_prev + self.prev_img_number)
                    target_img = np.array(Image.open(img_path))

                    break

                imgs_counted_prev += imgs_counted

        sample = {'prev_imgs': prev_imgs_stacked,
                  'commands_dx': dx_list,
                  'commands_dy': dy_list,
                  'pan': pan_list,
                  'tilt': tilt_list,
                  'target': target_img,
                  }

        sample = self.random_invert_sequence(sample)

        sample = self.transform(sample)
        return sample

    def random_invert_sequence(self, sample):
        # inverts sequence of images
        # inverts the commands order and sign
        if bool(random.getrandbits(1)):
            index_list = []
            for i in reversed(range(1, self.prev_img_number)):
                index_list.append(3*i)
                index_list.append(3*i+1)
                index_list.append(3*i+2)

            prev_imgs = np.take(sample['prev_imgs'], index_list, axis=2)
            prev_imgs = np.concatenate((sample['target'], prev_imgs), axis=2)
            sample_out = {'prev_imgs': prev_imgs,
                          'commands_dx': - 1.0 *  np.flip(sample['commands_dx']),
                          'commands_dy': - 1.0 * np.flip(sample['commands_dy']),
                          'pan': 1.0 * np.flip(sample['pan'])[:-1], #remove the last one, corresponds to old first img, now target
                          'tilt': 1.0* np.flip(sample['tilt'])[:-1],
                          'target': np.take(sample['prev_imgs'], [0,1,2], axis=2),
                          }
        else:
            sample_out = sample
            #remove the last one, corresponds to target img
            sample_out['pan'] = sample_out['pan'][:-1]
            sample_out['tilt'] = sample_out['tilt'][:-1]
        return sample_out


    def count_data_length(self, dir_path):
        count = 0
        for root, dirs, files in os.walk(dir_path, topdown=False):
            imgs_in_folder = 0
            for name in files:
                if ".png" in name and "target_box" not in name and "heat_map" not in name and "ss" not in name:
                    imgs_in_folder += 1
                    if imgs_in_folder > self.prev_img_number:
                        count += 1
        return count

class ScaleResizeToTensor(object):
    # remove mean of entire dataset from images, per each single colour channel
    # resize to self.img_size
    # convert to tensor, (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # divide by 255
    # --> data is in range [-1, 1] with ~0 mean

    def __init__(self, mean_rgb_list=None):
        if mean_rgb_list is None:
            mean_rgb_list = [107,110,110]
        self.mean_rgb_list = mean_rgb_list
        self.random_cropped_size_hw = (160, 208)

    def __call__(self, sample):
        prev_imgs, target = sample['prev_imgs'], sample['target']
        prev_imgs, target = prev_imgs.astype(np.int16), target.astype(np.int16) # transform from uint8 to int16 to subtract mean

        # remove mean of each color channel
        for channel in range(len(self.mean_rgb_list)):
            target[:,:,channel] -= self.mean_rgb_list[channel]
            for img_idx in range(int(prev_imgs.shape[2]/3)):
                prev_imgs[:,:,channel + 3* img_idx] -= self.mean_rgb_list[channel]


        rand_x = random.random()
        rand_y = random.random()
        t_0 = RandomCrop(out_size_hw=self.random_cropped_size_hw, rand_x=rand_x, rand_y=rand_y)
        t_1 = transforms.ToTensor()
        transform = transforms.Compose([t_0, t_1])

        transformed_sample = {'prev_imgs': transform(prev_imgs).div(255.0),
                              'commands_dx': torch.tensor(sample['commands_dx'], dtype = torch.float),
                              'commands_dy': torch.tensor(sample['commands_dy'], dtype = torch.float),
                              'pan': torch.tensor(sample['pan'], dtype = torch.float).div(450.0),
                              'tilt': torch.tensor(sample['tilt'], dtype = torch.float).div(450.0),
                              'target': center_crop(transform(target), (120,158)).div(255.0)
                              }
        # all tensors are of type Float
        return transformed_sample


class ScaleResizeToTensorSemSeg(object):
    # remove mean of entire dataset from images, per each single colour channel
    # resize to self.img_size
    # convert to tensor, (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # divide by 255
    # --> data is in range [-1, 1] with ~0 mean

    def __init__(self):
        self.random_cropped_size_hw = (160, 208)

    def __call__(self, sample):
        # convert from three to one channel, in format (H, W, 1)
        prev_imgs = img_to_ss(sample['prev_imgs'])
        target = img_to_ss(sample['target'])

        rand_x = random.random()
        rand_y = random.random()
        t_0 = RandomCrop(out_size_hw=self.random_cropped_size_hw, rand_x=rand_x, rand_y=rand_y)
        t_1 = transforms.ToTensor()
        transform = transforms.Compose([t_0, t_1])

        transformed_sample = {'prev_imgs': transform(prev_imgs).float().div(12.0),
                              'commands_dx': torch.tensor(sample['commands_dx'], dtype = torch.float),
                              'commands_dy': torch.tensor(sample['commands_dy'], dtype = torch.float),
                              'pan': torch.tensor(sample['pan'], dtype = torch.float).div(450.0),
                              'tilt': torch.tensor(sample['tilt'], dtype = torch.float).div(450.0),
                              'target': center_crop(transform(target), (120,158)).long().squeeze()
                              }
        # all tensors are of type Float
        return transformed_sample

def eval_mean_rgb_dataset(data_dir):
    mean_rgb_img = np.zeros(3)
    mean_rgb_dataset = np.zeros(3)
    n_of_imgs = 0
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if ".png" in name:
                img_name = os.path.join(root, name)
                img = np.array(Image.open(img_name))
                # cumulative moving average formula
                for channel in range(3):
                    mean_rgb_img[channel] = np.mean(np.reshape(img[:,:,channel], (img.shape[0] * img.shape[1])))
                    mean_rgb_dataset[channel] += (mean_rgb_img[channel] - mean_rgb_dataset[channel]) / (n_of_imgs + 1)
                    assert 0 <= mean_rgb_dataset[channel] <= 255
                n_of_imgs += 1

    mean_rgb_dataset = np.rint(mean_rgb_dataset).astype(np.uint8).tolist()
    return mean_rgb_dataset

def revert_transform(mean_rgb_list, img_tensor):
    # reverts the transform by re-adding the mean, but then divides again by 255 to leave images in [0,1]
    for channel in range(3):
        img_tensor[channel,:,:] = img_tensor[channel,:,:].mul(255.0).add(mean_rgb_list[channel]).div(255.0)
    return img_tensor

if __name__ == "__main__":
    # this is just a test for the code

    # mean_rgb_list = eval_mean_rgb_dataset(data_dir)
    # classes_count_dataset = count_semseg_classes_dataset(data_dir)
    # print("classes_count_dataset: ", classes_count_dataset)
    # print("mean rgb of entire dataset: ", mean_rgb_list)

    data_dir = "Mthesis/database/my_database/datasets/predictor_ds"
    dataset = PredictorDataset(data_dir, prev_img_number=5)
    print("database length: ", len(dataset))
    for _ in range(100):
        sample_rand = dataset[np.random.randint(0, len(dataset))]
    print("prev_imgs", sample_rand["prev_imgs"].size())
    print("commands_dx", sample_rand["commands_dx"].size())
    print("commands_dy", sample_rand["commands_dy"].size())
    print("pan", sample_rand["pan"].size())
    print("tilt", sample_rand["tilt"].size())
    print("target", sample_rand["target"].size())
    assert torch.max(sample_rand["prev_imgs"]).item() <= 1 and torch.min(sample_rand["prev_imgs"]).item() >= -1 and torch.max(sample_rand["target"]).item() <= 1 and torch.min(sample_rand["target"]).item() >= -1