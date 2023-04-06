import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from . import attention, ff_parser

import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    # print('initialization method [%s]' % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x = self.up(x)
        return x


class AttentionUNetFFskip(nn.Module):
    def __init__(self, n_classes=1, in_channel=3, out_channel=1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=in_channel, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        # ------------------FF parser
        # TODO: make this inp dim dynamic - https://stackoverflow.com/a/73469100

        self.ff1 = ff_parser.Conditioning(64, 64)
        self.ff2 = ff_parser.Conditioning(32, 128)
        self.ff3 = ff_parser.Conditioning(16, 256)
        self.ff4 = ff_parser.Conditioning(8, 512)
        # self.ff5 = ff_parser.Conditioning(4, 1024)

        self.ff1_up = ff_parser.Conditioning(64, 64)
        self.ff2_up = ff_parser.Conditioning(32, 128)
        self.ff3_up = ff_parser.Conditioning(16, 256)
        self.ff4_up = ff_parser.Conditioning(8, 512)
        # ------------------

        # -----ff inception in skip conn-----
        self.ff_inception_1 = ff_parser.FourierBlock(64, 64)
        self.ff_inception_2 = ff_parser.FourierBlock(128, 128)

        # -------------Decoder
        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = attention.AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = attention.AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.att3 = attention.AttentionBlock(f_g=128, f_l=128, f_int=64)
        # self.upconv3 = ConvBlock(ch_in=256, ch_out=128)
        self.upconv3 = ConvBlock(ch_in=128 + 384, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.att2 = attention.AttentionBlock(f_g=64, f_l=64, f_int=32)
        # self.upconv2 = ConvBlock(ch_in=128, ch_out=64)
        self.upconv2 = ConvBlock(ch_in=64 + 192, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)
        # self.conv_1x1 = nn.Conv2d(128, out_channel, kernel_size=1, stride=1, padding=0)

        # self.cls = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     # nn.Conv2d(filters[4], 2, 1),
        #     nn.Conv2d(1024, 2, 1),
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Sigmoid(),
        # )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)

        # --------
        x1_ff_out = self.ff1(x1)
        # --------
        x2 = self.maxpool(x1_ff_out)
        x2 = self.conv2(x2)

        # --------
        x2_ff_out = self.ff2(x2)
        # --------
        x3 = self.maxpool(x2_ff_out)
        x3 = self.conv3(x3)

        # --------
        x3_ff_out = self.ff3(x3)
        # --------
        x4 = self.maxpool(x3_ff_out)
        x4 = self.conv4(x4)

        # --------
        x4_ff_out = self.ff4(x4)
        # --------
        x5 = self.maxpool(x4_ff_out)
        x5 = self.conv5(x5)

        # --------
        # x5_ff_out = self.ff5(x5)
        # --------
        # -------------Classification-------------
        # cls_branch = self.cls(x5_ff_out).squeeze(3).squeeze(2)  # (B,N,1,1)->(B,N)
        # cls_branch_max = cls_branch.argmax(dim=1)
        # cls_branch_max = cls_branch_max[:, np.newaxis].float()

        # decoder + concat
        d5 = self.up5(x5)
        # d5 = self.up5(x5_ff_out)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.concat((x4, d5), dim=1)
        d5 = self.upconv5(d5)
        # ----
        d5 = self.ff4_up(d5)
        # ----

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.concat((x3, d4), dim=1)
        d4 = self.upconv4(d4)
        # ----
        d4 = self.ff3_up(d4)
        # ----

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        # ff inception-------
        x2_ff_inception_out = self.ff_inception_2(x2)
        # -------------------
        # d3 = torch.concat((x2, d3), dim=1)
        d3 = torch.concat((x2_ff_inception_out, d3), dim=1)
        d3 = self.upconv3(d3)
        # ----
        d3 = self.ff2_up(d3)
        # ----

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        # ff inception-------
        x1_ff_inception_out = self.ff_inception_1(x1)
        # -------------------
        # d2 = torch.concat((x1, d2), dim=1)
        d2 = torch.concat((x1_ff_inception_out, d2), dim=1)
        d2 = self.upconv2(d2)
        # ----
        d2 = self.ff1_up(d2)
        # ----

        # d2_cgm_out = self.dotProduct(d2, cls_branch_max)
        # d2_cat = torch.concat((d2, d2_cgm_out), dim=1)
        # d1 = self.conv_1x1(d2_cat)

        d1 = self.conv_1x1(d2)
        return F.sigmoid(d1)
