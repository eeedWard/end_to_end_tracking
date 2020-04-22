import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
from termcolor import colored
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from predictor.predictor import Predictor
from feature_extractor.fe_triplet import FETriplet
from dataset.encoder_dataset import EncoderDataset
from dataset.utils_dataset import center_crop
from utils_e2e import MSE_distance

class HeatMapGenerator:
    def __init__(self, prev_img_number):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prev_img_number = prev_img_number

        self.predictor = Predictor(prev_img_number=prev_img_number, inference_mode=True)
        self.feat_extr = FETriplet(inference_mode=True)

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.dataset_dir = self.dataset_general_dir + "/datasets/encoder_ds"
        self.dataset = EncoderDataset(self.dataset_dir, prev_img_number=prev_img_number, generate_ground_truth=True)

        print(colored("\nGenerating heatmaps: did you set randx and randy to 0.5 in encoder_ds?\n", "yellow"))
        print(colored("\nGenerating heatmaps: did you use same threshold values in encoder_ds as when training encoder?\n", "yellow"))

    def run(self, batch_size=32):
        data_loader = torch.utils.data.DataLoader(self.dataset , batch_size=batch_size,
                                                  shuffle=False, num_workers=8, pin_memory=True)

        for batch_idx, sample_batched in enumerate(data_loader):

            print("Analysing batch {}/{}".format(batch_idx+1, self.dataset.__len__() // batch_size))

            sample_batched['prev_imgs'] = sample_batched['prev_imgs'].to(self.device)
            sample_batched['target_box'] = sample_batched['target_box'].to(self.device) #(80, 104)
            sample_batched['target_box_loss'] = sample_batched['target_box_loss'].to(self.device) #(64, 80)
            sample_batched['target_box_size_hw'] = sample_batched['target_box_size_hw'].to(self.device)
            sample_batched['commands_dx'] = sample_batched['commands_dx'][:, :-1].to(self.device)
            sample_batched['commands_dy'] = sample_batched['commands_dy'][:, :-1].to(self.device)
            sample_batched['pan'] = sample_batched['pan'].to(self.device)
            sample_batched['tilt'] = sample_batched['tilt'].to(self.device)

            features_target = self.feat_extr.extract_features(sample_batched['target_box_loss'])

            fe_area_hw = (64, 80)
            self.generate_cmds_heat_map(sample_batched['prev_imgs'],
                                        sample_batched['commands_dx'],
                                        sample_batched['commands_dy'],
                                        sample_batched['pan'],
                                        sample_batched['tilt'],
                                        features_target,
                                        fe_area_hw,
                                        sample_batched['img_code_name'])

    def generate_cmds_heat_map(self, prev_imgs, prev_cmds_x, prev_cmds_y,
                               prev_pan, prev_tilt, target_features, fe_crop_hw, img_code_name):
        assert len(target_features.size()) == 2 and len(prev_imgs.size()) == 4 and prev_cmds_x.size()[1] \
               == prev_cmds_y.size()[1] == prev_tilt.size()[1] -1 == self.prev_img_number-1

        max_cmd = 0.2
        n_of_samples = 60
        neg_cmds_coords = np.linspace(-max_cmd, 0, n_of_samples//2)
        pos_cmds_coords = np.linspace(0, max_cmd, n_of_samples//2)
        coords = np.flip(np.concatenate([neg_cmds_coords, pos_cmds_coords[1:]]))

        x_grid, y_grid = np.meshgrid(coords, coords)

        batch_size = prev_imgs.size()[0]
        cost_grid_tensor = torch.zeros((batch_size, x_grid.shape[0], x_grid.shape[1]))

        loop_count = 0
        for i in range(x_grid.shape[0]):
            for j in range(y_grid.shape[1]):
                cmd_x = x_grid[i,j]
                cmd_y = y_grid[i,j]
                cmd = torch.ones((batch_size, 2)).mul(torch.tensor([cmd_x, cmd_y])).to(self.device)
                prev_cmds_x_extended = torch.cat([prev_cmds_x, cmd[:, 0].unsqueeze(0).t()], 1)
                prev_cmds_y_extended = torch.cat([prev_cmds_y, cmd[:, 1].unsqueeze(0).t()], 1)

                sample_predictor = {'prev_imgs': prev_imgs,
                                    'commands_dx': prev_cmds_x_extended,
                                    'commands_dy': prev_cmds_y_extended,
                                    'pan': prev_pan,
                                    'tilt': prev_tilt,
                                    }

                with torch.no_grad():
                    pred_frame = self.predictor.predict(sample_predictor)
                    pred_frame_centre = center_crop(pred_frame, fe_crop_hw)

                    features = self.feat_extr.extract_features(pred_frame_centre) #upsample carried out here, if required

                    cost_grid_tensor[:, i, j] = MSE_distance(features, target_features)
                loop_count += 1
                print("cmds map coordinate [{}/{}]".format(loop_count, x_grid.shape[0] * x_grid.shape[1]), end='\r')

        for cost_grid_idx in range(batch_size):
            cost_grid = cost_grid_tensor[cost_grid_idx, :, :].numpy()

            best_cmd_x_idx, best_cmd_y_idx = np.where(cost_grid == cost_grid.min())
            best_cmd_x_idx, best_cmd_y_idx = best_cmd_x_idx[0], best_cmd_y_idx[0] #this is needed for some numpy bs
            # find the actual value of cmd(x, y)
            best_cmd_x = x_grid[best_cmd_x_idx, best_cmd_y_idx]
            best_cmd_y = y_grid[best_cmd_x_idx, best_cmd_y_idx]

            color_map = matplotlib.cm.get_cmap('Reds')
            normed_grid = (cost_grid - np.amin(cost_grid)) / (np.amax(cost_grid) - np.amin(cost_grid))
            mapped_data = cv2.resize(color_map(normed_grid)[:,:,:-1], (prev_imgs.size()[3], prev_imgs.size()[2]), interpolation=cv2.INTER_NEAREST)
            heat_map = cv2.cvtColor(mapped_data.astype('float32'), cv2.COLOR_RGBA2BGR)*255.0

            img_name = img_code_name[cost_grid_idx]
            im_dir = img_name[:-26]

            cv2.imwrite(im_dir + '{}_heat_map_{}'.format(img_name[-26:-10], img_name[-9:]), heat_map)
            with open(im_dir[:-5] + "/cmd_x_ground_truth.txt", 'a') as filehandle:
                filehandle.write("%s,%s\n"%(best_cmd_x, img_name))
            with open(im_dir[:-5] + "/cmd_y_ground_truth.txt", 'a') as filehandle:
                filehandle.write("%s,%s\n"%(best_cmd_y, img_name))

if __name__ == "__main__":
    heat_map_generator = HeatMapGenerator(prev_img_number=5)

    print('dataset length: {} \n'.format(len(heat_map_generator.dataset)))
    # heat_map_generator.run()
