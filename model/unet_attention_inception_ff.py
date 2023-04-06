import torch.nn.functional as F
from torch import nn
from . import attention, ff_parser
import torch
from .unet_parts import UpInception, InceptionConv, Down, DoubleConv, OutConv


class InceptionAttentionUNetFF(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InceptionAttentionUNetFF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.block1 = InceptionConv(64, 32)
        self.block2 = InceptionConv(128, 64)
        self.block3 = InceptionConv(256, 128)
        self.block4 = InceptionConv(512, 128)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # ------------Attention blocks
        self.att5 = attention.AttentionBlock(f_g=64, f_l=16, f_int=16)
        self.att4 = attention.AttentionBlock(f_g=128, f_l=32 // factor, f_int=32)
        self.att3 = attention.AttentionBlock(f_g=256, f_l=128 // factor, f_int=64)
        self.att2 = attention.AttentionBlock(f_g=512, f_l=256 // factor, f_int=128)
        # ------------

        # ------------------FF parsers
        self.ff1 = ff_parser.Conditioning(64, 64)
        self.ff2 = ff_parser.Conditioning(32, 128)
        self.ff3 = ff_parser.Conditioning(16, 256)
        self.ff4 = ff_parser.Conditioning(8, 512)
        # ------------------

        self.up1 = UpInception(1024 + 512, 256 // factor, bilinear)
        self.up2 = UpInception(1024, 128 // factor, bilinear)
        self.up3 = UpInception(512, 32 // factor, bilinear)
        self.up4 = UpInception(224, 16, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1_ff_out = self.ff1(x1)
        # --------
        x2 = self.down1(x1_ff_out)
        x2_ff_out = self.ff2(x2)
        # --------
        x3 = self.down2(x2_ff_out)
        x3_ff_out = self.ff3(x3)
        # --------
        x4 = self.down3(x3_ff_out)
        x4_ff_out = self.ff4(x4)
        # --------
        x5 = self.down4(x4_ff_out)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x_up1 = self.up1(x5, x4, block4)
        # ------------s
        att2_x_op = self.att2(g=x4, x=x_up1)
        x_cat2 = torch.concat((att2_x_op, x_up1), dim=1)
        # ------------e
        x_up2 = self.up2(x_cat2, x3, block3)
        # ------------s
        att3_x_op = self.att3(g=x3, x=x_up2)
        x_cat3 = torch.concat((att3_x_op, x_up2), dim=1)
        # ------------e
        x_up3 = self.up3(x_cat3, x2, block2)
        # ------------s
        att4_x_op = self.att4(g=x2, x=x_up3)
        x_cat4 = torch.concat((att4_x_op, x_up3), dim=1)
        # ------------e
        x_up4 = self.up4(x_cat4, x1, block1)
        # ------------s
        att5_x_op = self.att5(g=x1, x=x_up4)
        x_cat5 = torch.concat((att5_x_op, x_up4), dim=1)
        # ------------e
        logits = self.outc(x_cat5)
        return F.sigmoid(logits)
