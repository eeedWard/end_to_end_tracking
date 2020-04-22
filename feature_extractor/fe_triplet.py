import torch
import torchvision
import os
import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils_e2e import save_network_details, count_parameters, MSE_distance
from dataset.utils_dataset import crop_tensor
from dataset.fe_triplet_dataset import FETripletDataset
from feature_extractor.feat_extr_net import TripletExtractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class FETriplet:
    def __init__(self, inference_mode=False, notes=""):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = TripletExtractor().to(self.device)

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.train_dataset_dir = self.dataset_general_dir + "/datasets/fe_triplets_ds"
        self.val_dataset_dir = self.dataset_general_dir + "/datasets/fe_triplets_val_ds"

        self.load_model_path = self.dataset_general_dir + '/model_fe_tripl.pth'

        if not inference_mode:
            self.train_set = FETripletDataset(self.train_dataset_dir)
            self.val_set = FETripletDataset(self.val_dataset_dir)
            self.loss = TripletMarginLoss()
            self.best_eval_loss = 10e6

        if inference_mode:
            self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
            self.net.eval()

        self.notes = notes

        self.patch_in_size_hw = (64, 80)
        self.upsample = torch.nn.Upsample(size=self.patch_in_size_hw)


    def train(self,
              load_prev_model=False,
              train_dataset=None,
              val_dataset=None,
              load_model_path="",
              epochs=210,
              batch_size=454,
              learning_rate = 1e-3,
              log_interval=5,
              save_model_epoch_interval = 15):

        train_dataset = self.train_set if train_dataset is None else train_dataset
        val_dataset = self.val_set if val_dataset is None else val_dataset
        load_model_path = self.load_model_path if load_model_path is None else load_model_path

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6,
                                                               cooldown=6, verbose=True, threshold=0.01, min_lr=1e-6)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)

        initial_time = datetime.now()
        log_dir = "/tensorboard_logs/feature_extractor/" + "lr_{}_bs_{}_".format(learning_rate, batch_size) + initial_time.strftime("%m-%d_%H-%M-%S_train") + "/"
        loss = None
        epoch_init = 0
        if load_prev_model:
            checkpoint = torch.load(load_model_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch_init = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            log_dir = checkpoint['log_dir']
            print("Model loaded at epoch {}".format(epoch_init))

        save_model_folder = self.dataset_general_dir + log_dir + 'models'
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)
        if not load_prev_model:
            dataset_size_list = [len(train_dataset) + len(val_dataset), len(train_dataset), len(val_dataset)]
            save_network_details(self.dataset_general_dir + log_dir, self.notes, self.net, optimizer, scheduler, batch_size, dataset_size_list)

        writer = SummaryWriter(self.dataset_general_dir + log_dir)

        last_lr = None
        running_loss = 0.0
        # t_end_batch = datetime.now() #just to initialize the variable
        for epoch in range(epoch_init, epochs):
            self.net.train()
            t0_epoch = datetime.now()
            for batch_idx, sample_batched in enumerate(train_loader):
                # t0_batch = datetime.now()
                # print("batch loaded in: ", (t0_batch-t_end_batch).total_seconds(), " seconds")

                sample_batched['anchor'] = sample_batched['anchor'].to(self.device)
                sample_batched['positive'] = sample_batched['positive'].to(self.device)
                sample_batched['negative'] = sample_batched['negative'].to(self.device)

                # writer.add_graph(self.net, sample_batched)

                anchor, positive, negative = self.net(sample_batched)
                loss = self.loss(anchor, positive, negative)
                optimizer.zero_grad()
                loss.backward()
                # plot_grad_weight(self.net.named_parameters(), writer, epoch, 0.02)
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % log_interval == 0:
                    n_data_analysed = epoch * len(train_dataset) + (batch_idx + 1) * batch_size
                    print('Epoch [{}/{}], Step [{}/{}], Data: {}k, Loss: {:.4f}'
                          .format(epoch+1, epochs, batch_idx, len(train_loader), n_data_analysed//1000, loss.item()))

                    # ...log the running loss
                    writer.add_scalar('Loss/training',
                                      running_loss / log_interval,
                                      n_data_analysed)

                    running_loss = 0.0

                # print("batch analysed in: ", (datetime.now()-t0_batch).total_seconds(), " seconds")
                # t_end_batch = datetime.now()

            print("Epoch train in: ", (datetime.now()-t0_epoch).total_seconds(), " seconds. Optimizer: {}".format(optimizer))

            n_data_analysed = (epoch + 1) * len(train_dataset)
            eval_loss = self.eval(val_loader)
            writer.add_scalar('Loss/validation',
                              eval_loss,
                              n_data_analysed)

            if epoch > 50 and (epoch % save_model_epoch_interval == 0 or eval_loss < self.best_eval_loss):
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

            list_of_img_lists = self.visualize_validation_triplets()
            for img_list in list_of_img_lists:
                img_grid = torchvision.utils.make_grid(img_list, nrow=4)
                writer.add_image("Epoch_{}/validation".format(epoch), img_grid, n_data_analysed)

            writer.flush()
            scheduler.step(eval_loss)
            if scheduler._last_lr[0] != last_lr:
                if epoch != 0:
                    print("Reducing lr \n")
                    with open(self.dataset_general_dir + log_dir + "AAA_network_details.txt", 'a') as filehandle:
                        filehandle.write("lr changed from {} to {} at the end of epoch {}, n_data_analysed {} \n"
                                         .format(last_lr, scheduler._last_lr[0], epoch, n_data_analysed))
                last_lr = scheduler._last_lr[0]
            # t_end_batch = datetime.now()

        writer.close()
        total_training_time = (datetime.now() - initial_time).total_seconds()
        print()
        print("Total training time in: ", total_training_time, " seconds")

    def eval(self, data_loader):
        t0 = datetime.now()
        loss_list = []

        self.net.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(data_loader):
                sample_batched['anchor'] = sample_batched['anchor'].to(self.device)
                sample_batched['positive'] = sample_batched['positive'].to(self.device)
                sample_batched['negative'] = sample_batched['negative'].to(self.device)

                anchor, positive, negative = self.net(sample_batched)
                loss = self.loss(anchor, positive, negative)
                loss_list.append(loss.item())

            eval_loss = np.mean(loss_list)

        print("Model eval in: ", (datetime.now()-t0).total_seconds(), " seconds", " loss: {}".format(eval_loss))
        return eval_loss

    def visualize_validation_triplets(self):
        # reads some validation patches and ranks them based on distance

        dataset_dir_val = self.dataset_general_dir + "/datasets/fe_triplets_val_ds"
        sanity_check_dataset = FETripletDataset(dataset_dir_val, return_negative_list=True)

        n_of_val_tests = 4

        list_of_img_lists = []

        self.net.eval()
        with torch.no_grad():
            for _ in range(n_of_val_tests):
                random_sample_idx = np.random.randint(0,len(sanity_check_dataset))
                sample_vis = sanity_check_dataset[random_sample_idx]

                sample_vis['anchor'] = sample_vis['anchor'].unsqueeze(0).to(self.device)
                sample_vis['positive'] = sample_vis['positive'].unsqueeze(0).to(self.device)
                for neg_idx in range(len(sample_vis['negative_list'])):
                    sample_vis['negative_list'][neg_idx] = sample_vis['negative_list'][neg_idx].unsqueeze(0).to(self.device)

                anchor_features = self.net.forward_once(sample_vis['anchor'])
                pos_features = self.net.forward_once(sample_vis['positive'])

                # mark positive sample with green rectangle
                sample_vis['positive'][:, 1, :8, :8] = 1.0
                dist_list = [MSE_distance(anchor_features,pos_features).item()]
                patches_img_list = [sample_vis['positive']]
                for neg_idx in range(len(sample_vis['negative_list'])):
                    neg_features = self.net.forward_once(sample_vis['negative_list'][neg_idx])
                    dist_list.append(MSE_distance(anchor_features,neg_features).item())
                    patches_img_list.append(sample_vis['negative_list'][neg_idx])

                dist_arr = np.array(dist_list)
                index_sort = np.argsort(dist_arr)
                patches_img_sorted = [sample_vis['anchor'].squeeze(0)]
                dist_sorted = []
                for index in index_sort:
                    patches_img_sorted.append(patches_img_list[index].squeeze(0))
                    dist_sorted.append(dist_list[index])

                list_of_img_lists.append(patches_img_sorted)

        return list_of_img_lists

    def val_saliency_map(self):
        self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
        dataset_dir_val = self.dataset_general_dir + "/datasets/fe_triplets_val_ds"
        sanity_check_dataset = FETripletDataset(dataset_dir_val, return_negative_list=True)
        n_of_val_tests = 4

        fig = plt.figure()

        self.net.eval()
        with torch.no_grad():
            for test_n in range(n_of_val_tests):
                ax = fig.add_subplot(1, n_of_val_tests, test_n+1, projection='3d')
                random_sample_idx = np.random.randint(0,len(sanity_check_dataset))
                sample_vis = sanity_check_dataset[random_sample_idx]
                anchor_patch = sample_vis['anchor'].to(self.device)
                pred_positive_img = sample_vis['pred_positive_img'].to(self.device)
                target_patch_location_tl = sample_vis['target_patch_location_tl']
                anchor_patch_original_size_hw = sample_vis['anchor_patch_original_size_hw']

                patch_location_yx =(target_patch_location_tl[0] + anchor_patch_original_size_hw[0]/2.0,
                      target_patch_location_tl[1] + anchor_patch_original_size_hw[1]/2.0)

                with torch.no_grad():
                    x_grid, y_grid, cost_grid = self.compute_saliency_map(anchor_patch,
                                                                          pred_positive_img,
                                                                          target_patch_location_tl,
                                                                          anchor_patch_original_size_hw,
                                                                          patch_size_hw=(anchor_patch.size()[1], anchor_patch.size()[2]))
                # plotting
                surf = ax.plot_surface(x_grid, y_grid, cost_grid, cmap=cm.coolwarm, linewidth=0.2, antialiased=True)
                # red point on patch target centre, where the minimum should be
                ax.scatter(patch_location_yx[1], patch_location_yx[0], 0, s=85, c='red')
                plt.xlabel("Img width")
                plt.ylabel("Img height")
                plt.xlim(0, sample_vis['pred_positive_img'].size()[2])
                plt.ylim(sample_vis['pred_positive_img'].size()[1], 0)
                plt.title("Cost")
                fig.colorbar(surf, shrink=0.5, aspect=13)

        plt.show()

    def compute_saliency_map(self, anchor_patch, image, target_patch_original_location_tl, anchor_patch_original_size_hw, patch_size_hw = None):
        # todo this function is messy and should be cleaned up
        if patch_size_hw is None:
            patch_size_hw = anchor_patch_original_size_hw

        img_size_hw = (image.size()[1], image.size()[2])
        # chose stride 1 and high steps to convolve with the whole image
        stride = 1
        steps = 900

        anchor_features = self.net.forward_once(anchor_patch.unsqueeze(0))

        patch_tl = (target_patch_original_location_tl[0] - int(round((patch_size_hw[0] - anchor_patch_original_size_hw[0]) / 2.0)),
                    target_patch_original_location_tl[1] - int(round((patch_size_hw[1] - anchor_patch_original_size_hw[1]) / 2.0)))

        # generate location list
        top_list = []
        left_list = []
        for y_step in range(2 * steps + 1):
            top = patch_tl[0] - steps * stride + y_step * stride
            if top >= 0 and  top + patch_size_hw[0] <= img_size_hw[0]:
                top_list.append(top)
        for x_step in range(2* steps + 1):
            left = patch_tl[1] - steps * stride + x_step * stride
            if left >=0 and left + patch_size_hw[1] <= img_size_hw[1]:
                left_list.append(left)

        top_grid, left_grid = np.meshgrid(top_list, left_list)

        # convolve at chosen locations
        cost_grid = np.ones_like(top_grid) * 150.0 #use this high value to spot obious errors

        for i in range(top_grid.shape[0]):
            for j in range(top_grid.shape[1]):
                test_patch = crop_tensor(image, (top_grid[i, j], left_grid[i, j]), patch_size_hw).unsqueeze(0)
                # resize to network input size
                test_patch = torch.nn.functional.interpolate(test_patch,
                                                             size=self.patch_in_size_hw,
                                                             align_corners=False,
                                                             mode='bilinear')
                patch_features = self.net.forward_once(test_patch)
                cost_grid[i,j] = MSE_distance(anchor_features, patch_features).item()


        # obtain patch centre coordinates
        y_grid = top_grid + patch_size_hw[0] / 2.0
        x_grid = left_grid + patch_size_hw[1] / 2.0

        return x_grid, y_grid, cost_grid

    def extract_features(self, img_patch):
        if (img_patch.size()[2], img_patch.size()[3]) !=  self.patch_in_size_hw:
            assert img_patch.size()[2] <= self.patch_in_size_hw[0] and  img_patch.size()[3] <= self.patch_in_size_hw[1]
            img_patch = self.upsample(img_patch)

        target_features = self.net.forward_once(img_patch)
        return target_features

class TripletMarginLoss:
    def __init__(self, margin=1.0):
        self.margin = margin
    def __call__(self, anchor, positive, negative):
        # pow(0.5) makes it equal to the nn.TripletMarginLoss
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

if __name__ == "__main__":
    feature_extractor = FETriplet(inference_mode=False,
                                      notes="using squared euclidean distance for triplet loss")
    print(feature_extractor.net)
    print('number of trainable parameters %s millions ' % (count_parameters(feature_extractor.net) / 1e6))
    print('dataset length: train: {}, validation: {} \n'.format(len(feature_extractor.train_set), len(feature_extractor.val_set)))

    feature_extractor.train(load_prev_model=False)
    # feature_extractor.val_saliency_map()
