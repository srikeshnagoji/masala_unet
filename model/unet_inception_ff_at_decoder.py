import torch.nn.functional as F
from torch import nn

from .unet_parts import UpInception, InceptionConv, Down, DoubleConv, OutConv
from . import attention, ff_parser


class InceptionUNetFFAtDecoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InceptionUNetFFAtDecoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.block1 = InceptionConv(64, 32)
        self.block2 = InceptionConv(128, 64)
        self.block3 = InceptionConv(256, 128)
        self.block4 = InceptionConv(512, 128)

        # ------------------FF parser
        # TODO: make this inp dim dynamic - https://stackoverflow.com/a/73469100

        #  FF at decoder - BRAIN
        self.ff1_up = ff_parser.Conditioning(128, 16)
        self.ff2_up = ff_parser.Conditioning(64, 16)
        self.ff3_up = ff_parser.Conditioning(32, 64)
        self.ff4_up = ff_parser.Conditioning(16, 128)
        # ------------------

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = UpInception(1024 + 512, 256 // factor, bilinear)
        self.up2 = UpInception(896, 128 // factor, bilinear)
        self.up3 = UpInception(448, 32 // factor, bilinear)
        self.up4 = UpInception(208, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = self.up1(x5, x4, block4)
        x = self.ff4_up(x)
        # x = torch.cat(x, block4)
        x = self.up2(x, x3, block3)
        x = self.ff3_up(x)
        # x = torch.cat(x, block3)
        x = self.up3(x, x2, block2)
        x = self.ff2_up(x)
        # x = torch.cat(x, block2)
        x = self.up4(x, x1, block1)
        x = self.ff1_up(x)
        # x = torch.cat(x, block1)
        logits = self.outc(x)
        return F.sigmoid(logits)
