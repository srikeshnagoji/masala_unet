import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from . import attention, ff_parser
from .unet_parts import UpInception, InceptionConv


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


class SkipConBlock(nn.Module):
    def __init__(
        self,
        maxpool_kernel_size,
        maxpool_kernel_stride,
        ch_in,
        ch_out,
        fmap_size_for_ff_parser,
    ):
        super().__init__()

        InceptionConv(in_channels=ch_in, out_channels=ch_out)
        self.skipconv = nn.Sequential(
            # Down convolution
            nn.MaxPool2d(maxpool_kernel_size, maxpool_kernel_stride, ceil_mode=True),
            nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # inception
            # InceptionConv(in_channels=ch_in, out_channels=ch_out),
            # FF
            ff_parser.Conditioning(fmap_size_for_ff_parser, ch_out),
        )

    def forward(self, x):
        x = self.skipconv(x)
        return x


class SkipConUpBlock(nn.Module):
    def __init__(
        self,
        up_sample_scale_factor,
        ch_in,
        ch_out,
        fmap_size_for_ff_parser,
    ):
        super().__init__()

        self.skipconv_up = nn.Sequential(
            # Up sample
            nn.Upsample(scale_factor=up_sample_scale_factor, mode="bilinear"),
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # FF
            ff_parser.Conditioning(fmap_size_for_ff_parser, ch_out),
        )

    def forward(self, x):
        x = self.skipconv_up(x)
        return x


class AttentionUNetFF3p(nn.Module):
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

        self.ff1_up = ff_parser.Conditioning(64, 64)
        self.ff2_up = ff_parser.Conditioning(32, 128)
        self.ff3_up = ff_parser.Conditioning(16, 256)
        self.ff4_up = ff_parser.Conditioning(8, 512)
        # ------------------

        # ------------------skip connection blocks
        self.skip15 = SkipConBlock(8, 8, 64, 64, 8)
        # self.skip14 = SkipConBlock(4, 4, 64, 64, 16)
        # self.skip13 = SkipConBlock(2, 2, 64, 64, 32)

        # self.skip25 = SkipConBlock(4, 4, 128, 128, 8)

        self.skip51d = SkipConUpBlock(8, 512, 512, 64)
        # ------------------

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = attention.AttentionBlock(f_g=512, f_l=512, f_int=256)
        # self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)
        self.upconv5 = ConvBlock(ch_in=1024 + 64, ch_out=512)
        # self.upconv5 = ConvBlock(ch_in=1024 + 64 + 128, ch_out=512)
        # self.upconv5 = ConvBlock(ch_in=1280, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = attention.AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)
        # self.upconv4 = ConvBlock(ch_in=512 + 64, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.att3 = attention.AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.upconv3 = ConvBlock(ch_in=256, ch_out=128)
        # self.upconv3 = ConvBlock(ch_in=256 + 64, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.att2 = attention.AttentionBlock(f_g=64, f_l=64, f_int=32)
        # self.upconv2 = ConvBlock(ch_in=128, ch_out=64)
        self.upconv2 = ConvBlock(ch_in=128 + 512, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)

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

        # decoder + concat
        d5 = self.up5(x5)
        x4_att = self.att5(g=d5, x=x4_ff_out)
        x15_down_sampled = self.skip15(x1_ff_out)
        # x25_down_sampled = self.skip25(x2_ff_out)
        d5 = torch.concat((x4_att, d5, x15_down_sampled), dim=1)
        # d5 = torch.concat((x4_att, d5, x15_down_sampled, x25_down_sampled), dim=1)
        d5 = self.upconv5(d5)
        # ----
        d5 = self.ff4_up(d5)
        # ----

        d4 = self.up4(d5)
        x3_att = self.att4(g=d4, x=x3_ff_out)
        # x14_down_sampled = self.skip14(x1_ff_out)
        d4 = torch.concat((x3_att, d4), dim=1)
        # d4 = torch.concat((x3_att, d4, x14_down_sampled), dim=1)
        d4 = self.upconv4(d4)
        # ----
        d4 = self.ff3_up(d4)
        # ----

        d3 = self.up3(d4)
        x2_att = self.att3(g=d3, x=x2_ff_out)
        # x13_down_sampled = self.skip13(x1_ff_out)
        d3 = torch.concat((x2_att, d3), dim=1)
        # d3 = torch.concat((x2_att, d3, x13_down_sampled), dim=1)
        d3 = self.upconv3(d3)
        # ----
        d3 = self.ff2_up(d3)
        # ----

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1_ff_out)
        x51d_up_sampled = self.skip51d(x4_ff_out)
        # d2 = torch.concat((x1, d2), dim=1)
        d2 = torch.concat((x1, d2, x51d_up_sampled), dim=1)
        d2 = self.upconv2(d2)
        # ----
        d2 = self.ff1_up(d2)
        # ----

        d1 = self.conv_1x1(d2)

        return F.sigmoid(d1)
