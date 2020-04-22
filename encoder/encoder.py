import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from encoder_net import EncoderNet
from end_to_end.predictor.predictor import Predictor
from feature_extractor.fe_triplet import FETriplet
from dataset.encoder_dataset import EncoderDataset
from dataset.utils_dataset import center_crop
from utils_e2e import save_network_details, count_parameters, create_image_grid, plot_grad_weight, draw_rect_on_tensor, cv2_to_tensor, tensor_to_cv2, draw_arrow, MSE_distance
from trial import fov_to_pt

class Encoder:
    def __init__(self, prev_img_number, inference_mode=False, notes=""):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = EncoderNet(prev_img_number=prev_img_number).to(self.device)
        self.prev_img_number = prev_img_number

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.dataset_dir = self.dataset_general_dir + "/datasets/encoder_ds"
        self.val_dataset_dir = self.dataset_general_dir + "/datasets/encoder_val_ds"

        if not inference_mode:
            self.predictor = Predictor(prev_img_number=prev_img_number, inference_mode=True)
            self.feat_extr = FETriplet(inference_mode=True)
            self.loss = nn.MSELoss()
            self.train_set = EncoderDataset(self.dataset_dir, prev_img_number=prev_img_number)
            self.val_set = EncoderDataset(self.val_dataset_dir, prev_img_number=prev_img_number)

        self.load_model_path = self.dataset_general_dir + '/model_encoder.pth'
        self.notes = notes
        self.best_eval_loss = 10e6

        if inference_mode:
            self.net.eval()
            self.load_model_path = self.dataset_general_dir + '/model_encoder_cs.pth'
            self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])

    def train(self,
              load_prev_model=False,
              warm_start=False,
              train_dataset=None,
              val_dataset=None,
              load_model_path=None,
              epochs=51,
              batch_size=32,
              learning_rate=1e-4,
              log_interval=5,
              save_model_epoch_interval=5):

        train_dataset = self.train_set if train_dataset is None else train_dataset
        val_dataset = self.val_set if val_dataset is None else val_dataset
        load_model_path = self.load_model_path if load_model_path is None else load_model_path

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        lr_lambda = lambda epoch: 0.96 ** epoch if epoch > 13 else 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 drop_last=True, pin_memory=True)

        initial_time = datetime.now()
        log_dir = "/tensorboard_logs/encoder/" + "lr_{}_bs_{}_".format(learning_rate, batch_size) + initial_time.strftime("%m-%d_%H-%M-%S_train") + "/"
        loss = None
        epoch_init = 0
        if load_prev_model:
            checkpoint = torch.load(load_model_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict']) #fixme when not doing warm start
            # epoch_init = checkpoint['epoch'] + 1
            # loss = checkpoint['loss']
            log_dir = checkpoint['log_dir']
            if not warm_start and log_dir[-4:] != "_cs/": log_dir = log_dir[:-1] + "_cs_better_data/" #fixme for different cs of same wm model
            print("Model loaded at epoch {}".format(checkpoint['epoch']))

        save_model_folder = self.dataset_general_dir + log_dir + 'models'
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)
        if not load_prev_model:
            dataset_size_list = [len(train_dataset) + len(val_dataset), len(train_dataset), len(val_dataset)]
            save_network_details(self.dataset_general_dir + log_dir, self.notes, self.net, optimizer, scheduler, batch_size,
                                 dataset_size_list)

        writer = SummaryWriter(self.dataset_general_dir + log_dir)

        last_lr = None
        # t_end_batch = datetime.now() #just to initialize the variable
        for epoch in range(epoch_init, epochs):
            self.net.train()
            t0_epoch = datetime.now()
            running_loss = []
            imgs_in_grid = 0 #counter to avoid placing more than 8 imgs in a single grid (imgs get lost)
            for batch_idx, sample_batched in enumerate(train_loader):
                # t0_batch = datetime.now()
                # print("batch loaded in: ", (t0_batch-t_end_batch).total_seconds(), " seconds")

                prev_imgs_clone = torch.clone(sample_batched['prev_imgs']) # these are for visual purposes (keep on cpu)

                sample_batched['prev_imgs'] = sample_batched['prev_imgs'].to(self.device)
                sample_batched['target_box'] = sample_batched['target_box'].to(self.device)
                sample_batched['target_box_loss'] = sample_batched['target_box_loss'].to(self.device)
                sample_batched['target_box_size_hw'] = sample_batched['target_box_size_hw'].to(self.device)

                # set last dx and dy as 0.0 (we are trying to find them!)
                sample_batched['commands_dx'] = sample_batched['commands_dx'].to(self.device)
                sample_batched['commands_dx'][:, -1] = 0.0
                sample_batched['commands_dy'] = sample_batched['commands_dy'].to(self.device)
                sample_batched['commands_dy'][:, -1] = 0.0

                sample_batched['pan'] = sample_batched['pan'].to(self.device)
                sample_batched['tilt'] = sample_batched['tilt'].to(self.device)

                # writer.add_graph(self.net, sample_batched)

                cmd_out = self.net(sample_batched)
                # cmd_out = torch.ones_like(cmd_out).mul(torch.tensor([0.2, 0.2]).to(self.device))
                sample_batched['commands_dx'][:, -1] = cmd_out[:, 0]
                sample_batched['commands_dy'][:, -1] = cmd_out[:, 1]

                if warm_start:
                    cmd_ground_truth = torch.zeros_like(cmd_out)
                    cmd_ground_truth[:, 0] = sample_batched['cmd_x_gound_truth']
                    cmd_ground_truth[:, 1] = sample_batched['cmd_y_gound_truth']

                    loss = self.loss(cmd_out, cmd_ground_truth)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % log_interval == 0:
                        n_data_analysed = epoch * len(train_dataset) + (batch_idx + 1) * batch_size
                        random_sample = np.random.randint(0,prev_imgs_clone.size()[0])
                        prev_imgs_clone[random_sample, 3*self.prev_img_number-3:3*self.prev_img_number, :, :] = draw_rect_on_tensor(prev_imgs_clone[random_sample, 3*self.prev_img_number-3:3*self.prev_img_number, :, :], (64,80))
                        cmd_img = torch.zeros_like(prev_imgs_clone[random_sample, 0:3, :, :]).cpu()
                        cmd_img = cv2_to_tensor(draw_arrow(tensor_to_cv2(cmd_img), cmd_ground_truth[random_sample, 0], cmd_ground_truth[random_sample, 1]), cmd_img.device)

                        img_grid = create_image_grid(mean_rgb_list=[0,0,0],
                                                     prev_imgs=prev_imgs_clone[random_sample, :, :, :],
                                                     dx_list=sample_batched['commands_dx'][random_sample, :].cpu(),
                                                     dy_list=sample_batched['commands_dy'][random_sample, :].cpu(),
                                                     img_1=sample_batched['target_box_loss'][random_sample, :, :, :].cpu(),
                                                     img_2=cmd_img, #ground truth commands
                                                     img_3=sample_batched['heat_map'][random_sample, :, :, :].cpu())
                        writer.add_image("Epoch_{}/training_{}".format(epoch, imgs_in_grid//8), img_grid, n_data_analysed)
                        imgs_in_grid += 1

                        # for _ in range(7):
                        #     # this is a check for the predictor + fe, random commands generate different predicted images.
                        #     # We can then compare them with the loaded heat map
                        #     rand_cmd = np.linspace(0, 0.2, 10)
                        #     rand_idx_x = random.randint(0, rand_cmd.shape[0] - 1)
                        #     rand_idx_y = random.randint(0, rand_cmd.shape[0] - 1)
                        #
                        #     cmds_x = sample_batched['commands_dx'][random_sample, :]
                        #     cmds_y = sample_batched['commands_dy'][random_sample, :]
                        #     cmds_x[-1] = rand_cmd[rand_idx_x]*random.choice((-1, 1))
                        #     cmds_y[-1] = rand_cmd[rand_idx_y]*random.choice((-1, 1))
                        #     sample = {'commands_dx': cmds_x.unsqueeze(0),
                        #               'commands_dy': cmds_y.unsqueeze(0),
                        #               'pan': sample_batched['pan'][random_sample, :].unsqueeze(0),
                        #               'tilt': sample_batched['tilt'][random_sample, :].unsqueeze(0),
                        #               'prev_imgs': sample_batched['prev_imgs'][random_sample, :].unsqueeze(0)
                        #     }
                        #
                        #     pred_img = self.predictor.predict(sample)
                        #
                        #     img_grid = create_image_grid(mean_rgb_list=[0,0,0],
                        #                                  prev_imgs=prev_imgs[random_sample, :, :, :].cpu(),
                        #                                  dx_list=cmds_x.cpu(),
                        #                                  dy_list=cmds_y.cpu(),
                        #                                  img_1=sample_batched['target_box_loss'][random_sample, :, :, :].cpu(),
                        #                                  img_2=pred_img.squeeze().cpu(),
                        #                                  img_3=sample_batched['heat_map'][random_sample, :, :, :].cpu())
                        #     writer.add_image("Epoch_{}/training_batch_{}".format(epoch, batch_idx), img_grid, n_data_analysed)


                else: #if not warm_start
                    pred_frame = self.predictor.predict(sample_batched) # (120, 158)

                    fe_area_hw_t = torch.ones_like(sample_batched['target_box_size_hw']).mul(torch.tensor([64, 80]).to(self.device)) #(64, 80) same val used to generate gt
                    pred_frame_centre = self.centre_crop_upsample(pred_frame, fe_area_hw_t, self.feat_extr.patch_in_size_hw)

                    assert pred_frame_centre.size() == sample_batched['target_box_loss'].size() # same as checking fe_area_hw_t == sample_batched['target_box_loss'].size(), why do we even use fe_area_hw_t? fixme

                    features = self.feat_extr.extract_features(pred_frame_centre)
                    with torch.no_grad():
                        features_target = self.feat_extr.extract_features(sample_batched['target_box_loss'])

                    loss = self.loss(features, features_target)
                    optimizer.zero_grad()
                    loss.backward()
                    #   plot_grad_weight(self.net.named_parameters(), writer, epoch, 0.02)
                    optimizer.step()

                    if batch_idx % log_interval == 0:
                        n_data_analysed = epoch * len(train_dataset) + (batch_idx + 1) * batch_size
                        random_sample = np.random.randint(0,prev_imgs_clone.size()[0])
                        pred_frame_draw = draw_rect_on_tensor(pred_frame[random_sample, :, :, :], fe_area_hw_t[random_sample, :])
                        prev_imgs_clone[random_sample, 3*self.prev_img_number-3:3*self.prev_img_number, :, :] = draw_rect_on_tensor(prev_imgs_clone[random_sample, 3*self.prev_img_number-3:3*self.prev_img_number, :, :],fe_area_hw_t[random_sample, :])
                        heat_map = torch.zeros_like(pred_frame_draw).cpu()
                        if epoch >= 15 and (batch_idx // log_interval) % 8 == 0:
                            # heat_map = self.generate_patch_heat_map(sample_batched['target_box_loss'][random_sample, :, :, :],
                            #                                         pred_frame[random_sample, :, :, :],
                            #                                         sample_batched['target_box_size_hw'][random_sample, :],
                            #                                         (fe_area_hw_t[random_sample, 0].item(), fe_area_hw_t[random_sample, 1].item()))
                            heat_map = self.generate_cmds_heat_map(sample_batched['prev_imgs'][random_sample, :, :, :],
                                                                   sample_batched['commands_dx'][random_sample, :-1],
                                                                   sample_batched['commands_dy'][random_sample, :-1],
                                                                   sample_batched['pan'][random_sample, :],
                                                                   sample_batched['tilt'][random_sample, :],
                                                                   features_target[random_sample, :],
                                                                   (fe_area_hw_t[random_sample, 0].item(), fe_area_hw_t[random_sample, 1].item()))
                        img_grid = create_image_grid([0.0, 0.0, 0.0],
                                                     prev_imgs_clone[random_sample, :, :, :],
                                                     dx_list=sample_batched['commands_dx'][random_sample, :].cpu(),
                                                     dy_list=sample_batched['commands_dy'][random_sample, :].cpu(),
                                                     img_1=sample_batched['target_box_loss'][random_sample, :, :, :].cpu(),
                                                     img_2=pred_frame_draw.cpu(),
                                                     img_3=heat_map)
                        writer.add_image("Epoch_{}/training_{}".format(epoch, imgs_in_grid//8), img_grid, n_data_analysed)
                        imgs_in_grid += 1

                running_loss.append(loss.item())
                if batch_idx % log_interval == 0:
                    n_data_analysed = epoch * len(train_dataset) + (batch_idx + 1) * batch_size
                    print('Epoch [{}/{}], Step [{}/{}], Data: {}k, Loss: {:.4f}'
                          .format(epoch, epochs, batch_idx, len(train_loader), n_data_analysed//1000, loss.item()))

                    # ...log the running loss
                    training_mode = "_warm_start" if warm_start else ""
                    writer.add_scalar('Loss/training{}'.format(training_mode),
                                      sum(running_loss) / len(running_loss),
                                      n_data_analysed)
                    running_loss = []


                # print("batch analysed in: ", (datetime.now()-t0_batch).total_seconds(), " seconds")
                # t_end_batch = datetime.now()

            print("Epoch train in: ", (datetime.now()-t0_epoch).total_seconds(), " seconds. Optimizer: {}".format(optimizer))

            n_data_analysed = (epoch + 1) * len(train_dataset)
            eval_loss, eval_out_sample_list = self.eval(val_loader, warm_start)
            training_mode = "_warm_start" if warm_start else ""
            writer.add_scalar('Loss/validation{}'.format(training_mode),
                              eval_loss,
                              n_data_analysed)

            for sample_out_eval in eval_out_sample_list:
                cmd_img = torch.zeros_like(sample_out_eval['prev_imgs'][0:3, :, :])
                cmd_img = cv2_to_tensor(draw_arrow(tensor_to_cv2(cmd_img), sample_out_eval['cmd_ground_truth'][0], sample_out_eval['cmd_ground_truth'][1]), cmd_img.device)
                img_grid = create_image_grid(mean_rgb_list=[0,0,0],
                                                 prev_imgs=sample_out_eval['prev_imgs'],
                                                 dx_list=sample_out_eval['commands_dx'],
                                                 dy_list=sample_out_eval['commands_dy'],
                                                 img_1=sample_out_eval['target_box_loss'],
                                                 img_2=cmd_img,
                                                 img_3=sample_out_eval['heat_map'])
                writer.add_image("Epoch_{}/eval".format(epoch), img_grid, n_data_analysed)

            if epoch > 10 and (epoch % save_model_epoch_interval == 0 or eval_loss < self.best_eval_loss):
                if not os.path.exists(save_model_folder):
                    os.makedirs(save_model_folder)
                path = save_model_folder + '/model_epoch_{}.pth'.format(epoch)
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    path = save_model_folder + '/best_model_epoch_{}.pth'.format(epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'log_dir': log_dir
                }, path)
                print("Model saved \n")

            scheduler.step()
            print(scheduler.state_dict())
            if scheduler._last_lr[0] != last_lr:
                if epoch != 0:
                    print("Reducing lr \n")
                    with open(self.dataset_general_dir + log_dir + "AAA_network_details.txt", 'a') as filehandle:
                        filehandle.write("lr changed from {} to {} at the end of epoch {}, n_data_analysed {} \n"
                                         .format(last_lr, scheduler._last_lr[0], epoch, n_data_analysed))
                last_lr = scheduler._last_lr[0]

            writer.flush()
            # t_end_batch = datetime.now()

        writer.close()
        total_training_time = (datetime.now() - initial_time).total_seconds()
        print()
        print("Total training time in: ", total_training_time, " seconds")

    def eval(self, data_loader, warm_start):
        t0 = datetime.now()
        loss_list = []

        self.net.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(data_loader):

                sample_batched['prev_imgs'] = sample_batched['prev_imgs'].to(self.device)
                sample_batched['target_box'] = sample_batched['target_box'].to(self.device)
                sample_batched['target_box_loss'] = sample_batched['target_box_loss'].to(self.device)
                sample_batched['target_box_size_hw'] = sample_batched['target_box_size_hw'].to(self.device)

                # set last dx and dy as 0.0 (we are trying to find them!)
                sample_batched['commands_dx'] = sample_batched['commands_dx'].to(self.device)
                sample_batched['commands_dx'][:, -1] = 0.0
                sample_batched['commands_dy'] = sample_batched['commands_dy'].to(self.device)
                sample_batched['commands_dy'][:, -1] = 0.0

                sample_batched['pan'] = sample_batched['pan'].to(self.device)
                sample_batched['tilt'] = sample_batched['tilt'].to(self.device)
                sample_batched['heat_map'] = sample_batched['heat_map'].to(self.device)

                cmd_out = self.net(sample_batched)

                sample_batched['commands_dx'][:, -1] = cmd_out[:, 0]
                sample_batched['commands_dy'][:, -1] = cmd_out[:, 1]

                if warm_start:
                    cmd_ground_truth = torch.zeros_like(cmd_out)
                    cmd_ground_truth[:, 0] = sample_batched['cmd_x_gound_truth']
                    cmd_ground_truth[:, 1] = sample_batched['cmd_y_gound_truth']

                    loss = self.loss(cmd_out, cmd_ground_truth)
                    loss_list.append(loss.item())

                else:
                    pred_frame = self.predictor.predict(sample_batched) # (120, 158)
                    fe_area_hw_t = torch.ones_like(sample_batched['target_box_size_hw']).mul(torch.tensor([64, 80]).to(self.device)) #(64, 80) same val used to generate gt
                    pred_frame_centre = self.centre_crop_upsample(pred_frame, fe_area_hw_t, self.feat_extr.patch_in_size_hw)

                    assert pred_frame_centre.size() == sample_batched['target_box_loss'].size() # same as checking fe_area_hw_t == target_box_loss.size(), why do we even use fe_area_hw_t? fixme

                    features = self.feat_extr.extract_features(pred_frame_centre)
                    features_target = self.feat_extr.extract_features(sample_batched['target_box_loss'])

                    loss = self.loss(features, features_target)
                    loss_list.append(loss.item())

            eval_loss = np.mean(loss_list)

            eval_out_sample_list = []
            for i in range(8):
                random_sample_idx = np.random.randint(0, sample_batched['prev_imgs'].size()[0])
                sample_out_eval = {'prev_imgs': sample_batched['prev_imgs'][random_sample_idx, :, :, :].cpu(),
                                   'target_box': sample_batched['target_box'][random_sample_idx, :, :, :].cpu(),
                                   'target_box_loss': sample_batched['target_box_loss'][random_sample_idx, :, :,:].cpu(),
                                   'commands_dx': sample_batched['commands_dx'][random_sample_idx, :].cpu(),
                                   'commands_dy': sample_batched['commands_dy'][random_sample_idx, :].cpu(),
                                   'pan': sample_batched['pan'][random_sample_idx, :].cpu(),
                                   'tilt': sample_batched['tilt'][random_sample_idx, :].cpu(),
                                   'heat_map': sample_batched['heat_map'][random_sample_idx, :].cpu(),
                                   'cmd_ground_truth': torch.zeros_like(cmd_out[random_sample_idx, :])
                                   }

                if warm_start:
                    sample_out_eval['cmd_ground_truth'] = cmd_ground_truth[random_sample_idx, :]

                eval_out_sample_list.append(sample_out_eval)


        print("Model eval in: ", (datetime.now() - t0).total_seconds(), " seconds", "eval loss: {}".format(eval_loss))
        return eval_loss, eval_out_sample_list

    def generate_patch_heat_map(self, anchor_patch, pred_image, anchor_patch_original_size_hw, patch_size_hw):
        assert len(pred_image.size()) == 3
        anchor_patch_original_size_hw = (anchor_patch_original_size_hw[0].item(), anchor_patch_original_size_hw[1].item())
        pred_image_centre_hw = (pred_image.size()[1]/2.0, pred_image.size()[2]/2.0)
        target_patch_location_tl = (int(pred_image_centre_hw[0] - anchor_patch_original_size_hw[0]/2.0),
                                    int(pred_image_centre_hw[1] - anchor_patch_original_size_hw[1]/2.0))
        # generates hit map of distance cost between target patwch and generated img
        x_grid, y_grid, cost_grid = self.feat_extr.compute_saliency_map(anchor_patch,
                                                                        pred_image,
                                                                        target_patch_location_tl,
                                                                        anchor_patch_original_size_hw,
                                                                        patch_size_hw)

        color_map = matplotlib.cm.get_cmap('Reds')
        normed_grid = (cost_grid - np.amin(cost_grid)) / (np.amax(cost_grid) - np.amin(cost_grid))
        mapped_data = color_map(normed_grid)[:,:,:-1]

        heat_map = torch.from_numpy(mapped_data).permute(2,1,0) #rows and columns are swapped in the cost matrix
        return heat_map

    def generate_cmds_heat_map(self, prev_imgs, prev_cmds_x, prev_cmds_y, prev_pan, prev_tilt, target_features, fe_crop_hw):
        assert len(target_features.size()) == 1 and len(prev_imgs.size()) == 3 and prev_cmds_x.size()[0] == prev_cmds_y.size()[0] == self.prev_img_number-1

        max_cmd = 0.2
        n_of_samples = 60 # samples for creating a discreate grid
        neg_cmds_coords = np.linspace(-max_cmd, 0, n_of_samples//2)
        pos_cmds_coords = np.linspace(0, max_cmd, n_of_samples//2)
        coords = np.flip(np.concatenate([neg_cmds_coords, pos_cmds_coords[1:]])) #

        x_grid, y_grid = np.meshgrid(coords, coords)
        cost_grid = np.ones_like(x_grid) * 150.0  #use this high value to spot obious errors

        loop_count = 0
        for i in range(x_grid.shape[0]):
            for j in range(y_grid.shape[1]):
                cmd_x = torch.tensor([x_grid[i,j]]).to(self.device)
                cmd_y = torch.tensor([y_grid[i,j]]).to(self.device)
                prev_cmds_x_extended = torch.cat([prev_cmds_x, cmd_x])
                prev_cmds_y_extended = torch.cat([prev_cmds_y, cmd_y])

                sample_predictor = {'prev_imgs': prev_imgs.unsqueeze(0),
                                    'commands_dx': prev_cmds_x_extended.unsqueeze(0),
                                    'commands_dy': prev_cmds_y_extended.unsqueeze(0),
                                    'pan': prev_pan.unsqueeze(0),
                                    'tilt': prev_tilt.unsqueeze(0)
                                    }

                with torch.no_grad():
                    pred_frame = self.predictor.predict(sample_predictor)
                    pred_frame_centre = center_crop(pred_frame, fe_crop_hw)

                    features = self.feat_extr.extract_features(pred_frame_centre) #upsample carried out here, if required
                    cost_grid[i,j] = MSE_distance(features, target_features).item()

                loop_count += 1
                print("cmds map [{}/{}]".format(loop_count, x_grid.shape[0] * x_grid.shape[1]), end='\r')

        color_map = matplotlib.cm.get_cmap('Reds')
        normed_grid = (cost_grid - np.amin(cost_grid)) / (np.amax(cost_grid) - np.amin(cost_grid))
        mapped_data = cv2.resize(color_map(normed_grid)[:,:,:-1], (prev_imgs.size()[2], prev_imgs.size()[1]), interpolation=cv2.INTER_NEAREST )
        heat_map = torch.from_numpy(mapped_data).permute(2,0,1)
        heat_map= draw_rect_on_tensor(heat_map, fe_crop_hw)

        # This plot is for thesis presentation only
        plot_for_thesis_presentation = False
        if plot_for_thesis_presentation:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.tick_params(pad=0.0, labelsize=9)
            ax.set_ylim(-0.20, 0.20)
            ax.set_xlim(-0.20, 0.20)
            surf = ax.plot_surface(x_grid, y_grid, cost_grid.swapaxes(0,1), cmap=matplotlib.cm.coolwarm, linewidth=0.2, antialiased=True)
            plt.xlabel("Δ pitch / vertical FOV", fontsize=12) #this is actually delta tilt, so - delta pitch
            plt.ylabel("Δ yaw / horizontal FOV", fontsize=12)
            plt.title("MSE_distance", fontsize=12)
            fig.colorbar(surf, shrink=0.5, aspect=13)
            return mapped_data, ax, cost_grid #return ax to plot ground truth

        return heat_map

    def centre_crop_upsample(self, img_t, crop_size_hw_t, out_size_hw):
        upsample = torch.nn.Upsample(size=out_size_hw)

        # tensor is N x C x H x W
        h = img_t.size()[2]
        w = img_t.size()[3]

        tensor_out = torch.zeros((img_t.size()[0], img_t.size()[1], out_size_hw[0], out_size_hw[1]), dtype=img_t.dtype).to(self.device)

        for t in range(img_t.size()[0]):
            assert crop_size_hw_t[t, 0] <= out_size_hw[0] and crop_size_hw_t[t, 1] <= out_size_hw[1]
            h_crop = crop_size_hw_t[t, 0]
            w_crop = crop_size_hw_t[t, 1]
            i_0 = (h-h_crop)//2
            i_1 = i_0 + (h-h_crop)%2
            j_0 = (w-w_crop)//2
            j_1 = j_0 + (w-w_crop)%2
            tensor_crop = img_t[t, :, i_0:-i_1, j_0:-j_1]
            assert tensor_crop.size()[1] == h_crop and tensor_crop.size()[2] == w_crop

            tensor_out[t, :, :, :] = upsample(tensor_crop.unsqueeze(0)).squeeze(0)

        return tensor_out

    def inference(self, sample):
        cmd_out = self.net(sample)
        return cmd_out


if __name__ == "__main__":
    encoder = Encoder(prev_img_number=5,
                      notes="Using smart random crops, larger min vehicle size. This just for cold start")

    # mode = "train"
    mode = "generate_cmds_heat_map" # used to generate heatmaps shown in thesis presentation

    if mode == "train":
        # print(predictor.net)
        # print('number of trainable parameters %s millions ' % (count_parameters(encoder.net) / 1e6))
        print('dataset length: {},  train: {}, validation: {} \n'.format(len(encoder.train_set) + len(encoder.val_set), len(encoder.train_set),
                                                                         len(encoder.val_set)))
        # encoder.train(load_prev_model=False, warm_start=True)
        encoder.train(load_prev_model=True, warm_start=False)

    elif mode == "generate_cmds_heat_map":

        # ids of (random) samples used to generate maps
        idx_list = [305, 367, 123, 23, 15, 78, 34, 400, 225, 200, 279, 43, 52, 325, 384, 416, 271, 120]
        for idx  in idx_list:
            sample_batched = encoder.val_set[idx]

            prev_imgs = sample_batched['prev_imgs'].to(encoder.device)

            sample_batched['commands_dx'] = sample_batched['commands_dx'].to(encoder.device)
            prev_cmds_x = sample_batched['commands_dx'][:-1]

            sample_batched['commands_dy'] = sample_batched['commands_dy'].to(encoder.device)
            prev_cmds_y = sample_batched['commands_dy'][:-1]

            prev_pan = sample_batched['pan'].to(encoder.device)
            prev_tilt = sample_batched['tilt'].to(encoder.device)

            target_features = encoder.feat_extr.extract_features(sample_batched['target_box_loss'].unsqueeze(0).to(encoder.device)).squeeze()

            fe_crop_hw = (64, 80)
            heat_map, ax, cost_matrix = encoder.generate_cmds_heat_map(prev_imgs, prev_cmds_x, prev_cmds_y, prev_pan, prev_tilt, target_features, fe_crop_hw)
            # red point on patch target centre, where the minimum should be

            x = -(sample_batched['cmd_x_gound_truth'] - 0.5)
            y = -(sample_batched['cmd_y_gound_truth'] - 0.5)
            pan, tilt = fov_to_pt(x, y, prev_pan[-1].cpu()*450, prev_tilt[-1].cpu()*450, 63)
            ground_truth_cmd_x = (pan - prev_pan[-1].cpu() * 450) / 63.0
            ground_truth_cmd_y = (tilt - prev_tilt[-1].cpu()*450) / (63.0 * 9.0 / 16.0)

            cv2.imwrite("repos/out/{} heatmap.png".format(idx), heat_map[:, :, :3] * 255.)
            last_img = tensor_to_cv2(sample_batched['prev_imgs'][-3:, :, :].mul(255.).cpu())
            cv2.imwrite("repos/out/{}_last_img.png".format(idx), last_img)
            target = tensor_to_cv2(sample_batched['target_box_loss'].mul(255.).cpu())
            cv2.imwrite("repos/out/{}_target_img.png".format(idx), target)

            plt.savefig("repos/out/{}_plot.png".format(idx))
            # ax.scatter(ground_truth_cmd_y, ground_truth_cmd_x, 0.0, s=85, c='red')
            ax.quiver(0.2, ground_truth_cmd_x, np.min(cost_matrix), ground_truth_cmd_y-0.2, 0.0, 0.0,  scale=0.1, color='red')
            ax.quiver(ground_truth_cmd_y, -0.2, np.min(cost_matrix), 0.0, ground_truth_cmd_x+0.2, 0.0, color='red')
            plt.savefig("repos/out/{}plot_with__gt.png".format(idx))
            print("\ndone with", idx)
            # plt.show()
