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
from predictor_net import UNetV2_SemSeg
from dataset.predictor_dataset import PredictorDataset
from dataset.utils_dataset import center_crop, ss_to_img
from utils_e2e import save_network_details, count_parameters, create_image_grid, create_image_grid_ss, plot_grad_weight, tensor_to_cv2, draw_rect_on_tensor


class Predictor:
    def __init__(self, prev_img_number, inference_mode=False, notes=""):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = UNetV2_SemSeg(prev_img_number=prev_img_number).to(self.device)
        # input has to be a Tensor of size either (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K)
        # This criterion expects a class index in the range [0, C-1]
        # Target: (N)(N) where each value is 0 <= targets[i] <= C−1 , or (N, d_1, d_2, ..., d_K)

        # [0.01758005 0.0846384  0.0011241  0.00686975 0.00059465 0.01023665
        #  0.03058421 0.55855576 0.19537849 0.0188431  0.04627842 0.02788447
        #  0.00143195]
        self.out_crop_size_hw = (120, 158)
        weights = torch.ones(13, dtype=torch.float).div(
            torch.tensor([0.01758005, 0.0846384, 0.0011241, 0.00686975, 0.00059465, 0.01023665,
                          0.03058421, 0.55855576, 0.19537849, 0.0188431, 0.04627842, 0.02788447, 0.00143195],
                         dtype=torch.float)).mul(
            self.out_crop_size_hw[0] * self.out_crop_size_hw[1])
        # self.loss = nn.CrossEntropyLoss(weight=weights.to(self.device))
        self.loss = nn.CrossEntropyLoss()

        self.dataset_general_dir = "Mthesis/database/my_database"
        self.dataset_dir = self.dataset_general_dir + "/datasets/predictor_ds"
        self.val_dataset_dir = self.dataset_general_dir + "/datasets/predictor_val_ds"

        # validation_split = 10
        # self.dataset = PredictorDataset(self.dataset_dir, prev_img_number=prev_img_number)
        # self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [
        #     len(self.dataset) - int(len(self.dataset) / validation_split), int(len(self.dataset) / validation_split)])

        self.train_set = PredictorDataset(self.dataset_dir, prev_img_number=prev_img_number, semseg=True)
        self.val_set = PredictorDataset(self.val_dataset_dir, prev_img_number=prev_img_number, semseg=True)

        self.load_model_path = self.dataset_general_dir + '/model_predictor_ss.pth'

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
              save_model_epoch_interval=5):

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
        log_dir = "/tensorboard_logs/predictor_ss/" + "lr_{}_bs_{}_".format(learning_rate, batch_size) + initial_time.strftime("%m-%d_%H-%M-%S_train") + "/"
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
                # print("net out size", net_out.size())
                # print("target size", target.size())
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
                        out_ss = torch.argmax(net_out, dim=1)
                        prev_imgs_rgb = torch.zeros((3 * prev_imgs.size()[1], prev_imgs.size()[2], prev_imgs.size()[3]))
                        for i in range(prev_imgs.size()[1]):
                            prev_imgs_rgb[3*i:3*i+3, :, :] = torch.from_numpy(ss_to_img(prev_imgs[random_sample, i, :, :].mul(12.0).cpu(),out_chw=True)).div(255)
                        target_rgb = torch.from_numpy(ss_to_img(target[random_sample, :, :].cpu(),out_chw=True)).div(255)
                        out_rgb = torch.from_numpy(ss_to_img(out_ss[random_sample, :, :].cpu(), out_chw=True)).div(255)
                        img_grid = create_image_grid(self.train_set.mean_rgb_list,
                                                     prev_imgs_rgb,
                                                     dx_list=commands_dx[random_sample, :].cpu(),
                                                     dy_list=commands_dy[random_sample, :].cpu(),
                                                     img_2=draw_rect_on_tensor(target_rgb, grid_lines_size),
                                                     img_3=draw_rect_on_tensor(out_rgb, grid_lines_size))
                        writer.add_image("Epoch_{}/training".format(epoch), img_grid, n_data_analysed)

                # print("batch analysed in: ", (datetime.now()-t0_batch).total_seconds(), " seconds")
                # t_end_batch = datetime.now()
            #
            print("Epoch train in: ", (datetime.now() - t0_epoch).total_seconds(), " seconds. Optimizer: {}".format(optimizer))

            if epoch != 0 and epoch % save_model_epoch_interval == 0:
                if not os.path.exists(save_model_folder):
                    os.makedirs(save_model_folder)
                path = save_model_folder + '/model_epoch_{}.pth'.format(epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'log_dir': log_dir
                }, path)
                print("Model saved \n")


            n_data_analysed = (epoch + 1) * len(train_dataset)
            eval_loss, change_cmd_out_dict_list = self.eval(val_loader)
            writer.add_scalar('Loss/validation',
                              eval_loss,
                              n_data_analysed)

            for eval_dict in change_cmd_out_dict_list:
                grid_lines_size = (eval_dict['target'].size()[-2]//3, eval_dict['target'].size()[-1]//3)
                out_ss = torch.argmax(eval_dict['net_out'], dim=1)
                for sample in range(out_ss.size()[0]):
                    prev_imgs_rgb = torch.zeros((3 * eval_dict['prev_imgs'].size()[1], eval_dict['prev_imgs'].size()[2], eval_dict['prev_imgs'].size()[3]))
                    for i in range(eval_dict['prev_imgs'].size()[1]):
                        prev_imgs_rgb[3*i:3*i+3, :, :] = torch.from_numpy(ss_to_img(eval_dict['prev_imgs'][sample, i, :, :].mul(12.0).cpu(),out_chw=True)).div(255)
                    target_rgb = torch.from_numpy(ss_to_img(eval_dict['target'][sample, :, :].cpu(),out_chw=True)).div(255)
                    out_rgb = torch.from_numpy(ss_to_img(out_ss[sample, :, :].cpu(), out_chw=True)).div(255)
                    img_grid = create_image_grid(self.train_set.mean_rgb_list,
                                                 prev_imgs_rgb,
                                                 dx_list=eval_dict['commands_dx'][sample, :].cpu(),
                                                 dy_list=eval_dict['commands_dy'][sample, :].cpu(),
                                                 img_2=draw_rect_on_tensor(target_rgb, grid_lines_size),
                                                 img_3=draw_rect_on_tensor(out_rgb, grid_lines_size))
                    writer.add_image("Epoch_{}/validation".format(epoch), img_grid, n_data_analysed)

            writer.flush()
            scheduler.step(eval_loss)
            print(scheduler.state_dict())
            if scheduler._last_lr[0] != last_lr:
                if epoch != 0:
                    print("Reducing lr \n")
                    with open(self.dataset_general_dir + log_dir + "AAA_network_details.txt", 'a') as filehandle:
                        filehandle.write("lr changed from {} to {} at the end of epoch {}, n_data_analysed {} \n"
                                         .format(last_lr, scheduler._last_lr[0], epoch, n_data_analysed))
                last_lr = scheduler._last_lr[0]

            t_end_batch = datetime.now()

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
        img_size_out = sample_batched['target'][sample_idx, :, :].size()[:]

        prev_imgs_mod = torch.zeros(mod_number, prev_imgs_number, img_size_in[0], img_size_in[1]).to(self.device)
        commands_dx_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        commands_dy_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        pan_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        tilt_mod = torch.zeros(mod_number, prev_imgs_number).to(self.device)
        target_mod = torch.zeros(mod_number, img_size_out[0], img_size_out[1]).to(self.device)

        for tensor in range(mod_number):
            prev_imgs_mod[tensor, :, :, :] = sample_batched['prev_imgs'][sample_idx, :, :, :]
            target_mod[tensor, :, :] = sample_batched['target'][sample_idx, :, :]
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


    def predict(self, sample):
        pred_frame = self.net(sample)
        return center_crop(pred_frame, self.out_crop_size_hw)

if __name__ == "__main__":
    predictor = Predictor(prev_img_number=5,
                          notes="No weights in loss")
    mode = "train"
    # mode = "run"
    # mode = "inference_on_dataset"
    # mode = "generate_triplets_ds"

    if mode == "train":
        # print(predictor.net)
        print('number of trainable parameters %s millions ' % (count_parameters(predictor.net) / 1e6))
        print('dataset length: train: {}, validation: {} \n'.format(len(predictor.train_set),
                                                                    len(predictor.val_set)))
        predictor.train(load_prev_model=False, learning_rate=1e-3, batch_size=32)
        del predictor
        predictor = Predictor(prev_img_number=5, notes="No weights in loss")
        predictor.train(load_prev_model=False, learning_rate=1e-4, batch_size=32)
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
