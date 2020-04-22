import torch
import random
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils_e2e import tensor_to_cv2

class FeaturesExtractDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = ScaleResizeToTensor()

    def __len__(self):
        count = self.count_data_length(self.data_dir)
        return count

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        img = np.zeros((200,263,3), dtype=np.uint8)
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for dir_name in dirs:
                img_name = os.path.join(self.data_dir, dir_name) + "/img%05d.png" %idx
                img = Image.open(img_name).convert("RGB")
                break

        img = self.transform(img)
        return img

    def count_data_length(self, dir_path):
        count = 0
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                if ".png" in name:
                    count += 1
        return count

class ScaleResizeToTensor(object):
    # random crop of random size
    # resize cropped patch to max_img_size (necessary to stack samples together)
    # convert to tensor, (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    # divide by 255
    # --> data is in range [0, 1]

    def __init__(self):
        self.min_h = 32
        # try to keep initial img ratio here (120/158)
        self.out_h = 64
        self.out_w = 80

    def __call__(self, img):

        h_crop = random.randint(self.min_h, self.out_h)
        w_crop = int(h_crop * self.out_w / self.out_h)

        t_0 = transforms.RandomCrop(size=(self.out_h, self.out_w))
        # t_1 = transforms.RandomHorizontalFlip()
        # t_2 = transforms.Resize(size=(self.out_h, self.out_w))
        # t_2 = transforms.Resize(size=(160, 208))
        t_3 = transforms.ToTensor()
        transform = transforms.Compose([t_0, t_3])

        transformed_img = transform(img)

        # all tensors are of type Float
        return transformed_img

if __name__ == "__main__":
    # this is just a test for the code

    data_dir = "Mthesis/database/my_database/inference_target_imgs_single"
    dataset = FeaturesExtractDataset(data_dir)
    print("database length: ", len(dataset))
    sample_rand = dataset[np.random.randint(0, len(dataset))]
    for key in sample_rand:
        print(key, sample_rand[key].size())
    assert torch.max(sample_rand).item() <= 1 and torch.min(sample_rand).item() >= 0

    img_open_cv = tensor_to_cv2(sample_rand.mul(255.))
    cv2.imwrite("Mthesis/database/my_database/out_img.png", img_open_cv)