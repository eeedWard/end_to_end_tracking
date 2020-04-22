import torch.nn as nn

class AutoEnc(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = self.conv_down(3, 64)
        self.dconv_down2 = self.conv_down(64, 32)
        self.dconv_down3 = self.conv_down(32, 32)
        self.dconv_down4 = self.conv_down(32, 32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = self.conv_up(32, 32)
        self.dconv_up3 = self.conv_up(32, 32)
        self.dconv_up2 = self.conv_up(32, 32)
        self.dconv_up1 = self.conv_up(32, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, img):

        # encoder
        # print("encoder input", img.shape)
        conv1 = self.dconv_down1(img)
        # print("conv1 ", conv1.shape)
        conv2 = self.dconv_down2(conv1)
        # print("conv2 ", conv2.shape)
        conv3 = self.dconv_down3(conv2)
        # print("conv3 ", conv3.shape)
        out_encoder = self.dconv_down4(conv3)
        # print("out_encoder ", out_encoder.shape)

        x = self.dconv_up4(out_encoder)
        x = self.upsample(x)
        # print("x3", x.shape)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print("x2", x.shape)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        # print("x1", x.shape)
        x = self.dconv_up1(x)
        x = self.upsample(x)
        # print("x0", x.shape)

        out = self.conv_last(x)
        # print("decoder output", out.shape)

        return out

    def conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    def conv_up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class TripletExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        # patch_in_size: (N, 3, 64, 80)
        flattened_size = 1280
        encoding_size = 128

        self.dconv_down1 = self.conv_down(3, 32)
        self.dconv_down2 = self.conv_down(32, 32)
        self.dconv_down3 = self.conv_down(32, 64)
        self.dconv_down4 = self.conv_down(64, 64)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(flattened_size, encoding_size * 3, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(encoding_size * 3, encoding_size, bias=True))


    def forward_once(self, img):

        # encoder
        # print("encoder input", img.shape)
        conv1 = self.dconv_down1(img)
        # print("conv1 ", conv1.shape)
        conv2 = self.dconv_down2(conv1)
        # print("conv2 ", conv2.shape)
        conv3 = self.dconv_down3(conv2)
        # print("conv3 ", conv3.shape)
        conv4 = self.dconv_down4(conv3)
        # print("out_encoder ", out_encoder.shape)

        out_flat = self.flatten(conv4)

        out = self.fc(out_flat)

        return out

    def forward(self, sample):
        anchor, positive, negative = sample['anchor'], sample['positive'], sample['negative']
        anchor = self.forward_once(anchor)
        positive = self.forward_once(positive)
        negative = self.forward_once(negative)
        return anchor, positive, negative

    def conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

