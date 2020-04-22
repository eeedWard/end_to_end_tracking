import os
import sys
sys.path.insert(0,os.getcwd())
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from predictor_net import UNetV2, UNetV2_small
from dataset.predictor_dataset import PredictorDataset
from dataset.utils_dataset import center_crop
from utils_e2e import save_network_details, count_parameters, create_image_grid, plot_grad_weight, tensor_to_cv2, draw_rect_on_tensor


class Predictor:
    def __init__(self, prev_img_number, inference_mode=False, model=UNetV2, notes=""):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model(prev_img_number=prev_img_number).to(self.device)

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.dataset_dir = self.dataset_general_dir + "/datasets/predictor_ds"
        self.val_dataset_dir = self.dataset_general_dir + "/datasets/predictor_val_ds"
        if notes == "Using linear interp":
            print("LINEAAAAR")
            self.dataset_dir = self.dataset_general_dir + "/datasets/predictor_ds_linear_interp"
            self.val_dataset_dir = self.dataset_general_dir + "/datasets/predictor_val_linear_ds"

        # validation_split = 10
        # self.dataset = PredictorDataset(self.dataset_dir, prev_img_number=prev_img_number)
        # self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [
        #     len(self.dataset) - int(len(self.dataset) / validation_split), int(len(self.dataset) / validation_split)])

        if not inference_mode:
            self.train_set = PredictorDataset(self.dataset_dir, prev_img_number=prev_img_number)
            self.val_set = PredictorDataset(self.val_dataset_dir, prev_img_number=prev_img_number)
            self.loss = nn.MSELoss()

        self.load_model_path = self.dataset_general_dir + '/model_predictor.pth'
        self.out_crop_size_hw = (120, 158)
        self.best_eval_loss = 10e6

        if inference_mode:
            self.net.eval()
            self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])

        self.notes = notes

    def train(self,
              load_prev_model=False,
              train_dataset=None,
              val_dataset=None,
              load_model_path=None,
              epochs=36,
              batch_size=16,
              learning_rate=1e-3,
              log_interval=5,
              save_model_epoch_interval=3):

        train_dataset = self.train_set if train_dataset is None else train_dataset
        val_dataset = self.val_set if val_dataset is None else val_dataset
        load_model_path = self.load_model_path if load_model_path is None else load_model_path

        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=1,
                                                               verbose=True, threshold=0.04, min_lr=1e-6)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 drop_last=True, pin_memory=True)

        initial_time = datetime.now()
        log_dir = "/tensorboard_logs/predictor/" + "lr_{}_bs_{}_".format(learning_rate, batch_size) + initial_time.strftime("%m-%d_%H-%M-%S_train") + "/"
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
            print("Model loaded at epoch {}, {}".format(checkpoint['epoch'], log_dir))

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
            imgs_in_grid = 0
            for batch_idx, sample_batched in enumerate(train_loader):
                # t0_batch = datetime.now()
                # print("batch loaded in: ", (t0_batch-t_end_batch).total_seconds(), " seconds")

                for key in sample_batched:
                    sample_batched[key] = sample_batched[key].to(self.device)

                # writer.add_graph(self.net, sample_batched)

                prev_imgs, commands_dx, commands_dy, target = sample_batched['prev_imgs'], \
                                                              sample_batched['commands_dx'], \
                                                              sample_batched['commands_dy'], \
                                                              sample_batched['target']
                net_out = self.net(sample_batched)
                net_out = center_crop(net_out, self.out_crop_size_hw)
                loss = self.loss(net_out, target)
                optimizer.zero_grad()
                loss.backward()
                # plot_grad_weight(self.net.named_parameters(), writer, epoch, 0.02)
                optimizer.step()

                running_loss.append(loss.item())
                if batch_idx % log_interval == 0:
                    n_data_analysed = epoch * len(train_dataset) + (batch_idx + 1) * batch_size
                    print('Epoch [{}/{}], Step [{}/{}], Data: {}k, Loss: {:.4f}'
                          .format(epoch, epochs, batch_idx, len(train_loader), n_data_analysed // 1000,
                                  loss.item()))

                    # ...log the running loss
                    writer.add_scalar('Loss/training',
                                      sum(running_loss) / len(running_loss),
                                      n_data_analysed)
                    running_loss = []

                    if (batch_idx / log_interval) % 4 == 0:
                        random_sample = np.random.randint(0, prev_imgs.size()[0])
                        grid_lines_size = (target.size()[-2]//3, target.size()[-1]//3)
                        img_grid = create_image_grid(self.train_set.mean_rgb_list,
                                                     prev_imgs[random_sample, :, :, :].cpu(),
                                                     dx_list=commands_dx[random_sample, :].cpu(),
                                                     dy_list=commands_dy[random_sample, :].cpu(),
                                                     img_2=draw_rect_on_tensor(target[random_sample, :, :, :].cpu(), grid_lines_size),
                                                     img_3=draw_rect_on_tensor(net_out[random_sample, :, :, :].cpu(), grid_lines_size))
                        imgs_in_grid += 1
                        writer.add_image("Epoch_{}/training_{}".format(epoch, imgs_in_grid//9), img_grid, n_data_analysed)

                # print("batch analysed in: ", (datetime.now()-t0_batch).total_seconds(), " seconds")
                # t_end_batch = datetime.now()

            print("Epoch train in: ", (datetime.now() - t0_epoch).total_seconds(), " seconds. Optimizer: {}".format(optimizer))

            n_data_analysed = (epoch + 1) * len(train_dataset)
            eval_loss, change_cmd_out_dict_list = self.eval(val_loader)
            writer.add_scalar('Loss/validation',
                              eval_loss,
                              n_data_analysed)

            for eval_dict in change_cmd_out_dict_list:
                grid_lines_size = (eval_dict['target'].size()[-2]//3, eval_dict['target'].size()[-1]//3)
                for sample in range(eval_dict['net_out'].size()[0]):
                    img_grid = create_image_grid(self.train_set.mean_rgb_list,
                                                 eval_dict['prev_imgs'][sample, :, :, :].cpu(),
                                                 dx_list=eval_dict['commands_dx'][sample, :].cpu(),
                                                 dy_list=eval_dict['commands_dy'][sample, :].cpu(),
                                                 img_2=draw_rect_on_tensor(eval_dict['target'][sample, :, :, :].cpu(), grid_lines_size),
                                                 img_3=draw_rect_on_tensor(eval_dict['net_out'][sample, :, :, :].cpu(), grid_lines_size))
                    writer.add_image("Epoch_{}/validation".format(epoch), img_grid, n_data_analysed)
            writer.flush()

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

            scheduler.step(eval_loss)
            print(scheduler.state_dict())
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

                for key in sample_batched:
                    sample_batched[key] = sample_batched[key].to(self.device)

                net_out = center_crop(self.net(sample_batched), self.out_crop_size_hw)
                loss = self.loss(net_out, sample_batched['target'])
                loss_list.append(loss.item())

            change_cmd_out_dict_list = []
            for i in range(3):
                random_sample_idx = np.random.randint(0, sample_batched['prev_imgs'].size()[0])
                change_cmd_out_dict = self.eval_change_cmd(sample_batched, random_sample_idx)
                change_cmd_out_dict_list.append(change_cmd_out_dict)

        eval_loss = np.mean(loss_list)
        print("Model eval in: ", (datetime.now() - t0).total_seconds(), " seconds", "eval loss: {}".format(eval_loss))
        return eval_loss, change_cmd_out_dict_list

    def eval_change_cmd(self, sample_batched, sample_idx):
        # t0 = datetime.now()

        cmds_change_factor = [1.0, 0.0, -1.0]
        mod_number = len(cmds_change_factor)
        prev_imgs_number = sample_batched['commands_dx'][sample_idx, :].size()[0]
        img_size_in = sample_batched['prev_imgs'][sample_idx, :, :, :].size()[1:]
        img_size_out = sample_batched['target'][sample_idx, :, :, :].size()[1:]

        prev_imgs_mod = torch.zeros(mod_number, prev_imgs_number * 3, img_size_in[0], img_size_in[1]).to(self.device)
        commands_dx_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        commands_dy_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        pan_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        tilt_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        target_mod = torch.zeros(mod_number, 3, img_size_out[0], img_size_out[1]).to(self.device)

        for tensor in range(mod_number):
            prev_imgs_mod[tensor, :, :, :] = sample_batched['prev_imgs'][sample_idx, :, :, :]
            target_mod[tensor, :, :, :] = sample_batched['target'][sample_idx, :, :, :]
            pan_mod[tensor, :] = sample_batched['pan'][sample_idx, :]
            tilt_mod[tensor, :] = sample_batched['tilt'][sample_idx, :]
            # perturb last command
            commands_dx_mod[tensor, :] = sample_batched['commands_dx'][sample_idx, :]
            commands_dx_mod[tensor, -1] = commands_dx_mod[tensor, -1].mul(cmds_change_factor[tensor])
            commands_dy_mod[tensor, :] = sample_batched['commands_dy'][sample_idx, :]
            commands_dy_mod[tensor, -1] = commands_dy_mod[tensor, -1].mul(cmds_change_factor[tensor])

        sample_mod = {
            'prev_imgs': prev_imgs_mod,
            'commands_dx': commands_dx_mod,
            'commands_dy': commands_dy_mod,
            'pan': pan_mod,
            'tilt': tilt_mod,
            'target': target_mod
        }

        net_out = self.net(sample_mod)
        net_out = center_crop(net_out, self.out_crop_size_hw)

        change_cmd_out_dict = {
            'prev_imgs': sample_mod['prev_imgs'],
            'commands_dx': sample_mod['commands_dx'],
            'commands_dy': sample_mod['commands_dy'],
            'target': sample_mod['target'],
            'net_out': net_out,
        }

        # print("Model eval change cmd in: ", (datetime.now()-t0).total_seconds(), " seconds")
        return change_cmd_out_dict

    def run(self, sample):
        self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])

        prev_imgs, commands_dx, commands_dy, target = sample['prev_imgs'], \
                                                      sample['commands_dx'], \
                                                      sample['commands_dy'], \
                                                      sample['target']
        for key in sample:
            sample[key] = sample[key].unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            predicted_img = self.net(sample)
            predicted_img = center_crop(predicted_img, self.out_crop_size_hw).squeeze(0)

        init_time = datetime.now()
        log_directory = "/tensorboard_logs/predictor/" + init_time.strftime("%Y-%m-%d_%H-%M-%S_run") + "/"
        writer = SummaryWriter(predictor.dataset_general_dir + log_directory)

        img_grid = create_image_grid(self.train_set.mean_rgb_list,
                                     prev_imgs[:, :, :].cpu(),
                                     dx_list=commands_dx.cpu(),
                                     dy_list=commands_dy.cpu(),
                                     img_2=target[:, :, :].cpu(),
                                     img_3=predicted_img[:, :, :].cpu())
        writer.add_image('images', img_grid)
        writer.add_graph(predictor.net, sample)
        writer.close()

    def inference_on_dataset(self):
        # generate dataset for autoencoder architecture
        self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
        inference_loader = torch.utils.data.DataLoader(self.train_set, batch_size=16, shuffle=True, num_workers=8,
                                                       pin_memory=True)
        img_folder = self.dataset_general_dir + '/datasets/fe_autoenc_ds'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        self.net.eval()
        with torch.no_grad():
            img_idx = 0
            for batch_idx, sample_batched in enumerate(inference_loader):
                for key in sample_batched:
                    sample_batched[key] = sample_batched[key].to(self.device)
                prev_imgs, commands_dx, commands_dy, target = sample_batched['prev_imgs'], \
                                                              sample_batched['commands_dx'], \
                                                              sample_batched['commands_dy'], \
                                                              sample_batched['target']
                net_out = center_crop(self.net(sample_batched), self.out_crop_size_hw)

                for sample in range(target.size()[0]):
                    net_out_img = tensor_to_cv2(net_out[sample, :, :, :].mul(255.).cpu())
                    target_img = tensor_to_cv2(target[sample, :, :, :].mul(255.).cpu())
                    cv2.imwrite(img_folder + '/img%05d.png' % img_idx, net_out_img)
                    cv2.imwrite(img_folder + '/img%05d.png' % (img_idx + 1), target_img)
                    img_idx += 2
                print("batch {} done".format(batch_idx))

    def generate_triplets_ds(self):
        # generate dataset for triplet loss architecture
        self.net.load_state_dict(torch.load(self.load_model_path)['model_state_dict'])
        max_delta = 0.2
        min_delta = 0.05
        max_cmd_out = 0.2
        self.dataset_dir = self.dataset_general_dir + "/datasets/predictor_inference_fe_ds"
        self.dataset = PredictorDataset(self.dataset_dir, prev_img_number=self.net.prev_img_number)
        triplets_loader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=False, num_workers=8,
                                                      pin_memory=True)
        img_folder = self.dataset_general_dir + '/datasets/fe_triplets_ds/imgs'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        len_dataset = len(self.dataset)
        self.net.eval()
        with torch.no_grad():
            img_idx = 0
            for batch_idx, sample_batched in enumerate(triplets_loader):
                for key in sample_batched:
                    sample_batched[key] = sample_batched[key].to(self.device)

                positive = center_crop(self.net(sample_batched), self.out_crop_size_hw)

                negative_list = []
                actual_batch_size = sample_batched['commands_dx'][:,-1].size()[0] #the last batch could be smaller than batch_size
                for neg_idx in range(5):
                    # perturb last command with a delta. Delta in [-max, -min] U [min, max].
                    prob_tensor = torch.ones(actual_batch_size).mul(0.5) #used to generate random signs
                    sign_x = torch.bernoulli(prob_tensor).mul(2).add(-1) #random 1s or -1s
                    sign_y = torch.bernoulli(prob_tensor).mul(2).add(-1)
                    delta_x = torch.rand(actual_batch_size).mul(max_delta-min_delta).add(min_delta).mul(sign_x).to(self.device)
                    delta_y = torch.rand(actual_batch_size).mul(max_delta-min_delta).add(min_delta).mul(sign_y).to(self.device)
                    assert min_delta <= abs(delta_x[0].item()) <= max_delta

                    new_cmd_x = sample_batched['commands_dx'][:,-1] + delta_x
                    new_cmd_y = sample_batched['commands_dy'][:,-1] + delta_y

                    # clamp commands and check if they still differ from previous commands
                    new_cmd_x = torch.clamp(new_cmd_x, -max_cmd_out, max_cmd_out)
                    new_cmd_y = torch.clamp(new_cmd_y, -max_cmd_out, max_cmd_out)
                    close_idx_x = (new_cmd_x - sample_batched['commands_dx'][:,-1]).abs() < min_delta
                    close_idx_y = (new_cmd_y - sample_batched['commands_dy'][:,-1]).abs() < min_delta
                    if close_idx_x.any():
                        new_cmd_x[close_idx_x] = sample_batched['commands_dx'][close_idx_x,-1] + delta_x[close_idx_x].mul(-1)
                    if close_idx_y.any():
                        new_cmd_y[close_idx_y] = sample_batched['commands_dy'][close_idx_y,-1] + delta_y[close_idx_y].mul(-1)

                    sample_batched['commands_dx'][:,-1] = new_cmd_x
                    sample_batched['commands_dy'][:,-1] = new_cmd_y
                    assert (sample_batched['commands_dx'][:,-1].abs() <= max_cmd_out).all() and (sample_batched['commands_dy'][:,-1].abs() <= max_cmd_out).all()

                    negative = center_crop(self.net(sample_batched), self.out_crop_size_hw)
                    negative_list.append(negative)

                # save imgs
                for sample in range(actual_batch_size):
                    anchor_img = tensor_to_cv2(sample_batched['target'][sample, :, :, :].mul(255.).cpu())
                    positive_img = tensor_to_cv2(positive[sample, :, :, :].mul(255.).cpu())
                    cv2.imwrite(img_folder + '/%05d_anchor.png' %img_idx, anchor_img)
                    cv2.imwrite(img_folder + '/%05d_apositive.png' %img_idx, positive_img)
                    for negative_idx in range(len(negative_list)):
                        negative_img = tensor_to_cv2(negative_list[negative_idx][sample, :, :, :].mul(255.).cpu())
                        cv2.imwrite(img_folder + '/%05d_negative_%03d.png' % (img_idx, negative_idx), negative_img)
                    img_idx += 1

                print("batch {}/{} done".format(batch_idx, len_dataset//triplets_loader.batch_size))

    def predict(self, sample):
        pred_frame = self.net(sample)
        return center_crop(pred_frame, self.out_crop_size_hw)

if __name__ == "__main__":
    predictor = Predictor(prev_img_number=5,
                          notes="")
    mode = "train"
    # mode = "run"
    # mode = "inference_on_dataset"
    # mode = "generate_triplets_ds"

    if mode == "train":
        # print(predictor.net)
        print('number of trainable parameters %s millions ' % (count_parameters(predictor.net) / 1e6))
        print('dataset length: train: {}, validation: {} \n'.format(len(predictor.train_set),
                                                                    len(predictor.val_set)))
        predictor.train(load_prev_model=False, learning_rate=1e-3, batch_size=64)
        # del predictor
        # predictor = Predictor(prev_img_number=5)
        # predictor.train(load_prev_model=False, learning_rate=1e-3, batch_size=64)
        # del predictor
        # predictor = Predictor(prev_img_number=5, notes="Plateau scheduler + different batch sizes")
        # predictor.train(load_prev_model=False, learning_rate=0.5e-3, batch_size=16)
        # del predictor
        # predictor = Predictor(prev_img_number=5, notes="Plateau scheduler + different batch sizes, pan & tilt")
        # predictor.train(load_prev_model=False, learning_rate=1e-4, batch_size=32)

    if mode == "run":
        dataset = PredictorDataset(predictor.dataset_dir, prev_img_number=5)
        random_idx = np.random.randint(0, len(dataset))
        print("chosen image index:", random_idx)
        run_sample = dataset[random_idx]

        predictor.run(run_sample)

    if mode == "inference_on_dataset":
        predictor.inference_on_dataset()

    if mode == 'generate_triplets_ds':
        predictor.generate_triplets_ds()
