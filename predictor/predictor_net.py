import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class UNetV2(nn.Module):

    def __init__(self, prev_img_number):
        super(UNetV2, self).__init__()
        self.prev_img_number = prev_img_number

        self.dconv_down1 = self.double_conv_down(prev_img_number * 3, 64)
        self.dconv_down2 = self.double_conv_down(64, 128)
        self.dconv_down3 = self.double_conv_down(128, 256)
        self.dconv_down4 = self.double_conv_down(256, 512)

        self.flatten = nn.Flatten()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = self.double_conv_up(256 + 512, 256)
        self.dconv_up3 = self.double_conv_up(128 + 256, 128)
        self.dconv_up2 = self.double_conv_up(64 + 128, 128)

        self.conv_last = nn.Sequential(
            nn.Conv2d(prev_img_number * 3 + 128, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3)
        )

        flattened_size = 66560

        self.fc_command = nn.Sequential(nn.Linear(prev_img_number * 3, prev_img_number * 3, bias=True),
                                        nn.Linear(prev_img_number * 3, prev_img_number * 3, bias=True),
                                        nn.Linear(prev_img_number * 3, prev_img_number * 3, bias=True),
                                        nn.Linear(prev_img_number * 3, flattened_size, bias=True))

        # im_h, im_w = 160, 208
        # batch_size = 32
        # coords_y = np.linspace(0, im_h-1, im_h, dtype=np.float)
        # coords_x = np.linspace(0, im_w-1, im_w, dtype=np.float)
        # img_coords_x, img_coords_y = np.meshgrid(coords_x, coords_y)
        # img_coords = np.zeros((batch_size, 2, im_h, im_w), dtype = np.float)
        # img_coords[:, 0, :, :] = img_coords_x
        # img_coords[:, 1, :, :] = img_coords_y
        # self.img_coords_t = torch.from_numpy(img_coords).type(torch.FloatTensor).cuda()

    def forward(self, sample):
        prev_imgs, commands_dx, commands_dy, pan, tilt = sample['prev_imgs'], \
                                                         sample['commands_dx'], \
                                                         sample['commands_dy'], \
                                                         sample['pan'], \
                                                         sample['tilt']
        # commands
        cmd = torch.cat((commands_dx, commands_dy, tilt), 1)
        out_cmd = self.fc_command(cmd)

        # This accounts for varying batch sizes
        # prev_imgs = torch.cat((prev_imgs, self.img_coords_t[:prev_imgs.size()[0], :, :, :]), dim=1)

        # encoder
        # print("encoder input", prev_imgs.shape)
        conv1 = self.dconv_down1(prev_imgs)
        # print("conv1 ", conv1.shape)
        conv2 = self.dconv_down2(conv1)
        # print("conv2 ", conv2.shape)
        conv3 = self.dconv_down3(conv2)
        # print("conv3 ", conv3.shape)
        out_encoder = self.dconv_down4(conv3)

        # print("encoder output", out_encoder.shape)
        out_img = self.flatten(out_encoder)
        # print("encoder output flattened", out_img.shape)
        out_mix = torch.mul(out_img, out_cmd)
        out = out_mix.view_as(out_encoder)

        x = self.upsample(out)
        # print("x1", x.shape)
        # print("conv3", conv3.shape)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        # print("x2", x.shape)
        # print("conv2", conv2.shape)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print("x", x.shape)
        # print("conv1", conv1.shape)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up2(x)
        decoder_out = self.upsample(x)
        # print("x", x.shape)
        # print("prev_imgs", prev_imgs.shape)
        decoder_img_mix = torch.cat([decoder_out, prev_imgs], dim=1)
        # prev_imgs_transf = self.stn(prev_imgs, decoder_img_mix)
        # decoder_img_mix = torch.cat([decoder_out, prev_imgs_transf], dim=1)

        out = self.conv_last(decoder_img_mix)
        # print("decoder output", out.shape)

        return out

    def double_conv_down(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.ReLU(inplace=True)
        )

    def double_conv_up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

class UNetV2_SemSeg(nn.Module):

    def __init__(self, prev_img_number):
        super(UNetV2_SemSeg, self).__init__()
        self.prev_img_number = prev_img_number

        self.dconv_down1 = self.double_conv_down(prev_img_number, 64)
        self.dconv_down2 = self.double_conv_down(64, 128)
        self.dconv_down3 = self.double_conv_down(128, 256)
        self.dconv_down4 = self.double_conv_down(256, 512)

        self.flatten = nn.Flatten()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = self.double_conv_up(256 + 512, 256)
        self.dconv_up3 = self.double_conv_up(128 + 256, 128)
        self.dconv_up2 = self.double_conv_up(64 + 128, 128)

        self.conv_last = nn.Sequential(
            nn.Conv2d(prev_img_number + 128, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 13, 3)
        )

        flattened_size = 66560
        self.fc_command = nn.Linear(prev_img_number * 3, flattened_size, bias=True)

        # im_h, im_w = 160, 208
        # batch_size = 32
        # coords_y = np.linspace(0, im_h-1, im_h, dtype=np.float)
        # coords_x = np.linspace(0, im_w-1, im_w, dtype=np.float)
        # img_coords_x, img_coords_y = np.meshgrid(coords_x, coords_y)
        # img_coords = np.zeros((batch_size, 2, im_h, im_w), dtype = np.float)
        # img_coords[:, 0, :, :] = img_coords_x
        # img_coords[:, 1, :, :] = img_coords_y
        # self.img_coords_t = torch.from_numpy(img_coords).type(torch.FloatTensor).cuda()

    def forward(self, sample):
        prev_imgs, commands_dx, commands_dy, pan, tilt = sample['prev_imgs'], \
                                                         sample['commands_dx'], \
                                                         sample['commands_dy'], \
                                                         sample['pan'], \
                                                         sample['tilt']
        # commands
        cmd = torch.cat((commands_dx, commands_dy, tilt), 1)
        out_cmd = self.fc_command(cmd)

        # This accounts for varying batch sizes
        # prev_imgs = torch.cat((prev_imgs, self.img_coords_t[:prev_imgs.size()[0], :, :, :]), dim=1)

        # encoder
        # print("encoder input", prev_imgs.shape)
        conv1 = self.dconv_down1(prev_imgs)
        # print("conv1 ", conv1.shape)
        conv2 = self.dconv_down2(conv1)
        # print("conv2 ", conv2.shape)
        conv3 = self.dconv_down3(conv2)
        # print("conv3 ", conv3.shape)
        out_encoder = self.dconv_down4(conv3)

        # print("encoder output", out_encoder.shape)
        out_img = self.flatten(out_encoder)
        # print("encoder output flattened", out_img.shape)
        out_mix = torch.mul(out_img, out_cmd)
        out = out_mix.view_as(out_encoder)

        x = self.upsample(out)
        # print("x1", x.shape)
        # print("conv3", conv3.shape)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        # print("x2", x.shape)
        # print("conv2", conv2.shape)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print("x", x.shape)
        # print("conv1", conv1.shape)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up2(x)
        decoder_out = self.upsample(x)
        # print("x", x.shape)
        # print("prev_imgs", prev_imgs.shape)
        decoder_img_mix = torch.cat([decoder_out, prev_imgs], dim=1)

        out = self.conv_last(decoder_img_mix)
        # print("decoder output", out.shape)

        return out

    def double_conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def double_conv_up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )