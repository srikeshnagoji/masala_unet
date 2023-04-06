import torch.nn.functional as F
from torch import nn

from .unet_parts import UpInception, InceptionConv, Down, DoubleConv, OutConv
from . import attention, ff_parser


class InceptionUNetFF(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InceptionUNetFF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.block1 = InceptionConv(64, 32)
        self.block2 = InceptionConv(128, 64)
        self.block3 = InceptionConv(256, 128)
        self.block4 = InceptionConv(512, 128)

        # ------------------FF parser
        # TODO: make this inp dim dynamic - https://stackoverflow.com/a/73469100

        self.ff1 = ff_parser.Conditioning(64, 64)
        self.ff2 = ff_parser.Conditioning(32, 128)
        self.ff3 = ff_parser.Conditioning(16, 256)
        self.ff4 = ff_parser.Conditioning(8, 512)

        # TODO: FF at decoder also. Ref unet_attention_with_ff_3p.py
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
        x1_ff_out = self.ff1(x1)
        x2 = self.down1(x1_ff_out)
        # print(x2.shape)
        x2_ff_out = self.ff2(x2)
        x3 = self.down2(x2_ff_out)
        # print(x3.shape)
        x3_ff_out = self.ff3(x3)
        x4 = self.down3(x3_ff_out)
        # print(x4.shape)
        x4_ff_out = self.ff4(x4)
        x5 = self.down4(x4_ff_out)
        # print(x5.shape)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = self.up1(x5, x4, block4)
        # x = torch.cat(x, block4)
        x = self.up2(x, x3, block3)
        # x = torch.cat(x, block3)
        x = self.up3(x, x2, block2)
        # x = torch.cat(x, block2)
        x = self.up4(x, x1, block1)
        # x = torch.cat(x, block1)
        logits = self.outc(x)
        return F.sigmoid(logits)
