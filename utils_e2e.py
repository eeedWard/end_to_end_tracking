import cv2
import torch
import torchvision
import numpy as np
from dataset.predictor_dataset import revert_transform
from datetime import datetime
from dataset.utils_dataset import ss_to_img

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param


def save_network_details(log_dir, notes, net, optimizer, scheduler, batch_size, dataset_size_list):
    with open(log_dir + "AAA_network_details.txt", 'w') as filehandle:
        filehandle.write("notes: {} \n".format(notes))
        filehandle.write("batch_size: %s \n" %batch_size)
        filehandle.write("optimizer %s \n" %optimizer)
        if scheduler:
            filehandle.write("scheduler %s \n" %scheduler.state_dict())
        filehandle.write("dataset size {}, train: {}, validation: {} \n".format(dataset_size_list[0], dataset_size_list[1], dataset_size_list[2]))
        filehandle.write("Network details: \n")
        filehandle.write("\n")
        filehandle.write(str(net) + "\n")
    return

def create_image_grid(mean_rgb_list, prev_imgs, dx_list=None, dy_list= None, img_1 = None, img_2=None, img_3=None):
    # images have to be cpu tensors
    # here we also add back the mean for each rgb channel
    # we also draw an arrow on each img for pan/tilt commands

    img_list = []
    target_img_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    pred_img_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    target_box_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    h = prev_imgs.size()[1]
    w = prev_imgs.size()[2]
    for i in range(int(prev_imgs.size()[0]/3)):
        img = prev_imgs[3*i:3*i+3, :, :]
        img = revert_transform(mean_rgb_list, img)
        if dx_list is not None and dy_list is not None:
            img = cv2_to_tensor(draw_arrow(tensor_to_cv2(img), dx_list[i], dy_list[i]), img.device)
        img_list.append(img)
    if img_1 is not None:
        h_small = img_1.size()[1]
        w_small = img_1.size()[2]
        img_1 = revert_transform(mean_rgb_list, img_1)
        target_box_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_1

    if img_2 is not None:
        h_small = img_2.size()[1]
        w_small = img_2.size()[2]
        img_2 = revert_transform(mean_rgb_list, img_2)
        target_img_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_2

    if img_3 is not None:
        h_small = img_3.size()[1]
        w_small = img_3.size()[2]
        img_3 = revert_transform(mean_rgb_list, img_3)
        pred_img_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_3

    img_list.append(target_box_pad)
    img_list.append(target_img_pad)
    img_list.append(pred_img_pad)

    img_grid = torchvision.utils.make_grid(img_list, nrow=4)
    return img_grid

def create_image_grid_ss(prev_imgs, dx_list=None, dy_list= None, img_1 = None, img_2=None, img_3=None):
    # images have to be cpu tensors
    # here we also add back the mean for each rgb channel
    # we also draw an arrow on each img for pan/tilt commands

    img_list = []
    target_img_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    pred_img_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    target_box_pad = torch.zeros_like(prev_imgs[0:3, :, :])
    h = prev_imgs.size()[1]
    w = prev_imgs.size()[2]
    for i in range(int(prev_imgs.size()[0])):
        img = ss_to_img(prev_imgs[i, :, :], out_chw=True)
        if dx_list is not None and dy_list is not None:
            img = cv2_to_tensor(draw_arrow(tensor_to_cv2(img), dx_list[i], dy_list[i]), img.device)
        img_list.append(img)
    if img_1 is not None:
        img_1 = ss_to_img(img_1, out_chw=True)
        h_small = img_1.size()[1]
        w_small = img_1.size()[2]
        target_box_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_1

    if img_2 is not None:
        img_2 = ss_to_img(img_2, out_chw=True)
        h_small = img_2.size()[1]
        w_small = img_2.size()[2]
        target_img_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_2

    if img_3 is not None:
        img_3 = ss_to_img(img_3, out_chw=True)
        h_small = img_3.size()[1]
        w_small = img_3.size()[2]
        pred_img_pad[:, int((h-h_small)/2):int((h-h_small)/2)+ h_small, int((w-w_small)/2):int((w-w_small)/2)+ w_small] = img_3

    img_list.append(target_box_pad)
    img_list.append(target_img_pad)
    img_list.append(pred_img_pad)

    img_grid = torchvision.utils.make_grid(img_list, nrow=4)
    return img_grid

def draw_arrow(image, dx, dy, max_cmd=0.2):
    img_centre_xy = (int(image.shape[1]/2), int(image.shape[0]/2))
    max_cmd = 0.2 # command value at max_cmd corresponds to img corner
    scale_x = img_centre_xy[0] / max_cmd
    scale_y = img_centre_xy[1] / max_cmd
    arrow_tip = (int(img_centre_xy[0] - scale_x * dx), int(img_centre_xy[1] - scale_y * dy))
    image = cv2.arrowedLine(image, img_centre_xy, arrow_tip, color=(0, 0, 1.0), thickness=2)
    image = cv2.circle(image, img_centre_xy, radius=3, color=(0, 1.0, 0), thickness=-1)
    return image

def tensor_to_cv2(image_tensor):
    image_rgb = image_tensor.permute(1,2,0).numpy()
    image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_cv2

def cv2_to_tensor(image_cv2, device):
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2,0,1).to(device)
    return image_tensor


def plot_grad_weight(named_parameters, writer, epoch, max_check):
    t0 = datetime.now()
    # Plots the gradients flowing through different layers in the net during training.
    # Can be used for checking for possible gradient vanishing / exploding problems.
    #
    # Usage: Plug this function in Trainer class after loss.backwards() as
    # "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    ave_grads_list = []
    max_grads_list= []
    ave_weight_list = []
    max_weight_list = []
    layer_idx = 0
    for name, p in named_parameters:
        if p.requires_grad and ("bias" not in name):
            mean_weight = p.data.mean()
            max_weight = p.data.max()
            mean_grad = p.grad.abs().mean()
            max_grad = p.grad.abs().max()
            ave_weight_list.append(mean_weight)
            max_weight_list.append(max_weight)
            ave_grads_list.append(mean_grad)
            max_grads_list.append(max_grad)
            if max_grad > max_check:
                print("clipping not working: {}".format(max_grad))
                print()

            # writer.add_scalar("weights per layer epoch {}/mean".format(epoch+1), mean_weight, layer_idx)
            # writer.add_scalar("weights per layer epoch {}/max".format(epoch+1), max_weight, layer_idx)
            # writer.add_scalar("grad per layer epoch {}/mean".format(epoch+1), mean_grad, layer_idx)
            # writer.add_scalar("grad per layer epoch {}/max".format(epoch+1), max_grad, layer_idx)
            # layer_idx += 1

    writer.add_histogram("gradients per layer, plot {}/mean".format(epoch//15), np.array(ave_grads_list), epoch +1)
    writer.add_histogram("gradients per layer, plot {}/max".format(epoch//15), np.array(max_grads_list), epoch +1)
    writer.add_histogram("weights per layer, plot {}/mean".format(epoch//15), np.array(ave_weight_list), epoch+1)
    writer.add_histogram("weights per layer, plot {}/max".format(epoch//15), np.array(max_weight_list), epoch+1)

    # print("Plotting time :{}s".format((datetime.now() - t0).total_seconds()))

def conv2d_out_size(input_size, kernel, stride=1, dilation=1, padding=0):
    h = input_size[0]
    w = input_size[1]
    k = kernel
    d = dilation
    s = stride
    p = padding
    if isinstance(kernel, int):
        k = (kernel, kernel)
    if isinstance(stride, int):
        s = (stride, stride)
    if isinstance(dilation, int):
        d = (dilation, dilation)
    if isinstance(padding, int):
        p = (padding, padding)

    h_out = (h + 2 * p[0] - d[0] * (k[0]-1) -1) / s[0] + 1
    w_out = (w + 2 * p[1] - d[1] * (k[1]-1) -1) / s[1] + 1

    return h_out, w_out

def draw_rect_on_tensor(tensor_in, rect_size_hw):
    # tensor_in is (N x H x W)
    h = tensor_in.size()[1]
    w = tensor_in.size()[2]
    y_min = int(h/2 - rect_size_hw[0]/2)
    y_max = int(h/2 + rect_size_hw[0]/2)
    x_min = int(w/2 - rect_size_hw[1]/2)
    x_max = int(w/2 + rect_size_hw[1]/2)
    tensor_out = tensor_in.clone()
    thickness = 3

    tensor_out[:, - thickness + y_min : y_min, :] = 0 #top
    tensor_out[:, y_max : y_max + thickness, :] = 0 #bottom
    tensor_out[:, :, - thickness + x_min : x_min] = 0 #left
    tensor_out[:, :, x_max : x_max + thickness] = 0 #right
    return tensor_out

def MSE_distance(tensor_1, tensor_2):
    # corresponds to torch.nn.MSELoss()
    # same distance as used in TripletMarginLoss(my implementation).However, there we sum, here we mean (no actual difference)
    dist = (tensor_1 - tensor_2).pow(2).mean(dim=1)
    return dist