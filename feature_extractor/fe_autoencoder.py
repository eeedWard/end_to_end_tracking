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
from utils_e2e import save_network_details, count_parameters
from dataset.feat_extr_dataset import FeaturesExtractDataset
from dataset.sanity_check_dataset import SanityCheckDataset, ResizeToTensor
from feature_extractor.feat_extr_net import AutoEnc
from PIL import Image


class FEAutoEncoder:
    def __init__(self, inference_mode=False, notes=""):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = AutoEnc().to(self.device)
        self.loss = nn.MSELoss()

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.dataset_dir = self.dataset_general_dir + "/inference_target_imgs"
        validation_split = 10
        self.dataset = FeaturesExtractDataset(self.dataset_dir)
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [len(self.dataset) - int(len(self.dataset) / validation_split), int(len(self.dataset) / validation_split)])

        self.load_model_path = self.dataset_general_dir + '/model_fe.pth'

        self.features = None

        if inference_mode:
            self.net.dconv_down4.register_forward_hook(self.get_features_from_layer)
            self.flatten = torch.nn.Flatten()
            self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
            self.net.eval()

        self.notes = notes

        self.patch_in_size = (64,80)
        self.upsample = torch.nn.Upsample(size=self.patch_in_size)


    def train(self,
              load_prev_model=False,
              train_dataset=None,
              val_dataset=None,
              load_model_path=None,
              epochs=5000,
              batch_size=16,
              learning_rate = 1e-3,
              log_interval=5,
              save_model_epoch_interval = 5):

        train_dataset = self.train_set if train_dataset is None else train_dataset
        val_dataset = self.val_set if val_dataset is None else val_dataset
        load_model_path = self.load_model_path if load_model_path is None else load_model_path

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13, 20, 27, 35], gamma=0.1)
        scheduler = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        initial_time = datetime.now()
        log_dir = "/tensorboard_logs/feature_extractor/" + initial_time.strftime("%Y-%m-%d_%H-%M-%S_train") + "/"
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
            dataset_size_list = [len(self.dataset), len(train_dataset), len(val_dataset)]
            save_network_details(self.dataset_general_dir + log_dir, self.notes, self.net, optimizer, scheduler, batch_size, dataset_size_list)

        writer = SummaryWriter(self.dataset_general_dir + log_dir)

        running_loss = 0.0
        # t_end_batch = datetime.now() #just to initialize the variable
        for epoch in range(epoch_init, epochs):
            self.net.train()
            t0_epoch = datetime.now()
            for batch_idx, sample_batched in enumerate(train_loader):
                # t0_batch = datetime.now()
                # print("batch loaded in: ", (t0_batch-t_end_batch).total_seconds(), " seconds")

                sample_batched = sample_batched.to(self.device)
                # writer.add_graph(self.net, sample_batched)

                net_out = self.net(sample_batched)
                loss = self.loss(net_out, sample_batched)
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

                    if (batch_idx/log_interval) % 4 == 0:
                        random_sample = np.random.randint(0,sample_batched.size()[0])
                        img_list = [sample_batched[random_sample, :, :, :], net_out[random_sample, :, :, :]]
                        img_grid = torchvision.utils.make_grid(img_list, nrow=4)
                        writer.add_image("Epoch_{}/training".format(epoch), img_grid, n_data_analysed)

                    running_loss = 0.0

                # print("batch analysed in: ", (datetime.now()-t0_batch).total_seconds(), " seconds")
                # t_end_batch = datetime.now()

            print("Epoch train in: ", (datetime.now()-t0_epoch).total_seconds(), " seconds. Optimizer: {}".format(optimizer))

            if epoch % save_model_epoch_interval == 0:
                path = save_model_folder + '/model_epoch_{}.pth'.format(epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'log_dir': log_dir
                }, path)

            n_data_analysed = (epoch + 1) * len(train_dataset)
            eval_loss, eval_out_dict_list  = self.eval(val_loader)
            writer.add_scalar('Loss/validation',
                              eval_loss,
                              n_data_analysed)

            img_list = []
            for eval_dict in eval_out_dict_list:
                img_list.append(eval_dict['target'])
                img_list.append(eval_dict['net_out'])

            img_grid = torchvision.utils.make_grid(img_list, nrow=4)
            writer.add_image("Epoch_{}/validation".format(epoch), img_grid, n_data_analysed)

            writer.flush()
            # scheduler.step()
            # t_end_batch = datetime.now()

        total_training_time = (datetime.now() - initial_time).total_seconds()
        print()
        print("Total training time in: ", total_training_time, " seconds")

    def eval(self, data_loader):
        t0 = datetime.now()
        loss_list = []
        eval_out_dict_list = []

        self.net.eval()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(data_loader):
                sample_batched = sample_batched.to(self.device)

                net_out = self.net(sample_batched)
                loss = self.loss(net_out, sample_batched)
                loss_list.append(loss.item())

            for i in range(4):
                random_sample_idx = np.random.randint(0,sample_batched.size()[0])

                eval_out_dict = {
                    'target': sample_batched[random_sample_idx, :, :, :],
                    'net_out': net_out[random_sample_idx, :, :, :]
                }
                eval_out_dict_list.append(eval_out_dict)

        eval_loss = np.mean(loss_list)
        print("Model eval in: ", (datetime.now()-t0).total_seconds(), " seconds", " loss: {}".format(eval_loss))

        return eval_loss, eval_out_dict_list

    def get_features_from_layer(self, layer_class, input_tuple, output):
        # input tuple = (input_tensor_to_layer)
        # output = layer_output_tensor
        self.features = output

    def sanity_check(self):
        dataset_dir = self.dataset_general_dir + '/sanity_check_imgs'
        dataset = SanityCheckDataset(dataset_dir)
        print('dataset length: {}'.format(len(dataset)))
        sanity_check_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        transform = ResizeToTensor()
        img_name = self.dataset_general_dir + '/target_box.png'
        target_img = Image.open(img_name).convert("RGB")
        target_img = transform(target_img).unsqueeze(0).to(self.device)

        self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
        self.net.dconv_down4.register_forward_hook(self.get_features_from_layer)
        cos = torch.nn.CosineSimilarity()
        flatten = torch.nn.Flatten()

        initial_time = datetime.now()
        log_dir = "/tensorboard_logs/sanity_check/" + initial_time.strftime("%Y-%m-%d_%H-%M-%S_sc") + "/"
        writer = SummaryWriter(self.dataset_general_dir + log_dir)

        patch_features_list = []
        patch_list = []
        self.net.eval()
        with torch.no_grad():

            _ = self.net(target_img)
            target_features = flatten(self.features)

            for batch_idx, patch_batched in enumerate(sanity_check_loader):
                # print(patch_batched)
                patch_batched = patch_batched.to(self.device)
                _ = self.net(patch_batched)
                patch_features_list.append(flatten(self.features))
                patch_list.append(patch_batched[0])

        cos_distance_list = []
        for patch_features in patch_features_list:
            cosine_distance = cos(target_features, patch_features)
            cos_distance_list.append(cosine_distance.item())

        cos_distance_arr = np.array(cos_distance_list)
        index_sort = np.argsort(- cos_distance_arr)
        patches_sorted = []
        cos_distance_sorted = []

        for index in index_sort:
            patches_sorted.append(patch_list[index])
            cos_distance_sorted.append(cos_distance_list[index])


        patches_sorted.insert(0, target_img.squeeze(0))
        img_grid = torchvision.utils.make_grid(patches_sorted, nrow=12, pad_value=1)
        writer.add_image("AAA_patches_sorted/cosine_distance", img_grid)
        writer.close()

        print(cos_distance_sorted)

    def extract_features(self, img_patch):
        if (img_patch.size()[2], img_patch.size()[3]) !=  self.patch_in_size:
            img_patch = self.upsample(img_patch)

        _ = self.net(img_patch)
        target_features = self.flatten(self.features)
        return target_features



if __name__ == "__main__":
    feature_extractor = FEAutoEncoder(inference_mode=False,
                                         notes="")
    # print(feature_extractor.net)
    # print('number of trainable parameters %s millions ' % (count_parameters(feature_extractor.net) / 1e6))
    print('dataset length: {},  train: {}, validation: {} \n'.format(len(feature_extractor.dataset), len(feature_extractor.train_set), len(feature_extractor.val_set)))
    feature_extractor.train(load_prev_model=False)
    # feature_extractor.sanity_check()


