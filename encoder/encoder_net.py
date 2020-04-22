import torch.nn as nn
import torch


class EncoderNet(nn.Module):

    def __init__(self, prev_img_number):
        super(EncoderNet, self).__init__()
        self.prev_img_number = prev_img_number
        self.max_out = 0.2

        self.dconv_down1 = self.double_conv_down(prev_img_number * 3, 64)
        self.dconv_down2 = self.double_conv_down(64 + 32, 128)
        self.dconv_down3 = self.double_conv_down(128, 256)
        self.dconv_down4 = self.double_conv_down(256, 256)

        self.dconv_target = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()

        flattened_size = 33280
        self.fc_command = nn.Linear(prev_img_number * 3, flattened_size, bias=True)
        self.fc = nn.Sequential(nn.Linear(flattened_size, 100),
                                nn.ReLU(inplace=True),
                                nn.Linear(100, 20),
                                nn.ReLU(inplace=True),
                                nn.Linear(20, 2))

        self.tanh = nn.Tanh()

    def forward(self, sample):
        out, commands_dx, commands_dy, tilt, target_box = sample['prev_imgs'], \
                                                          sample['commands_dx'], \
                                                          sample['commands_dy'], \
                                                          sample['tilt'], \
                                                          sample['target_box']
        # commands
        cmd = torch.cat((commands_dx, commands_dy, tilt), 1)
        out_cmd = self.fc_command(cmd)

        # target_box
        conv_target = self.dconv_target(target_box)

        # encoder
        out = out[:, :, :, :208]
        # print("encoder input", prev_imgs.shape)
        # print("target_box input", target_box.shape)
        out = self.dconv_down1(out)
        # print("conv1 ", conv1.shape)
        # print("conv_target ", conv_target.shape)

        # cat with target_box
        out = torch.cat([conv_target, out], dim=1)

        out = self.dconv_down2(out)
        # print("conv2 ", conv2.shape)
        out = self.dconv_down3(out)
        # print("conv3 ", conv3.shape)
        out = self.dconv_down4(out)

        # print("encoder output", out_encoder.shape)
        out = self.flatten(out)
        # print("encoder output flattened", out_img.shape)
        out = torch.mul(out, out_cmd)

        out = self.fc(out)
        # print("decoder output", out.shape)

        out = self.tanh(out).mul(self.max_out)

        return out

    def double_conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )


class EncoderNet_small(nn.Module):

    def __init__(self, prev_img_number):
        super(EncoderNet_small, self).__init__()
        self.prev_img_number = prev_img_number
        self.max_out = 0.2

        self.dconv_down1 = self.conv_down(prev_img_number * 3, 64)
        self.dconv_down2 = self.conv_down(64 + 32, 128)
        self.dconv_down3 = self.conv_down(128, 256)
        self.dconv_down4 = self.conv_down(256, 256)

        self.dconv_target = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()

        flattened_size = 33280
        self.fc_command = nn.Linear(prev_img_number * 3, flattened_size, bias=True)
        self.fc = nn.Sequential(nn.Linear(flattened_size, 100),
                                nn.ReLU(inplace=True),
                                nn.Linear(100, 20),
                                nn.ReLU(inplace=True),
                                nn.Linear(20, 2))

        self.tanh = nn.Tanh()

    def forward(self, sample):
        out, commands_dx, commands_dy, tilt, target_box = sample['prev_imgs'], \
                                                          sample['commands_dx'], \
                                                          sample['commands_dy'], \
                                                          sample['tilt'], \
                                                          sample['target_box']
        # commands
        cmd = torch.cat((commands_dx, commands_dy, tilt), 1)
        out_cmd = self.fc_command(cmd)

        # target_box
        conv_target = self.dconv_target(target_box)

        # encoder
        out = out[:, :, :, :208]
        # print("encoder input", prev_imgs.shape)
        # print("target_box input", target_box.shape)
        out = self.dconv_down1(out)
        # print("conv1 ", conv1.shape)
        # print("conv_target ", conv_target.shape)

        # cat with target_box
        out = torch.cat([conv_target, out], dim=1)

        out = self.dconv_down2(out)
        # print("conv2 ", conv2.shape)
        out = self.dconv_down3(out)
        # print("conv3 ", conv3.shape)
        out = self.dconv_down4(out)

        # print("encoder output", out_encoder.shape)
        out = self.flatten(out)
        # print("encoder output flattened", out_img.shape)
        out = torch.mul(out, out_cmd)

        out = self.fc(out)
        # print("decoder output", out.shape)

        out = self.tanh(out).mul(self.max_out)

        return out

    def conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
