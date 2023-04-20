import torch
import torch.nn as nn
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.fft import fft2, ifft2
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)

    # def forward(self, x, c):
    def forward(self, x):
        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        # x = fft2(x)
        # x = x * self.ff_parser_attn_map
        # x = ifft2(x).real
        # x = x.type(dtype)
        # ---------------WINDOWS-CUDA
        if torch.cuda.is_available():
            x_ = x
            x_ = fft2(x_)
            x_ = x_ * self.ff_parser_attn_map
            x_ = ifft2(x_).real
            x_ = x_.type(dtype)
        # ---------------
        else:
            # ---------------MAC - M1 - MPS
            cpu = torch.device("cpu")
            mps0 = torch.device("mps:0")
            x_ = x.to(cpu)
            x_ = fft2(x_)
            x_ = x_ * self.ff_parser_attn_map.to(cpu)
            x_ = ifft2(x_).real
            x_ = x_.type(dtype)
            x = x_.to(mps0)
            self.ff_parser_attn_map = self.ff_parser_attn_map.to(mps0)
        # ---------------
        # eq 3 in paper

        # normed_x = self.norm_input(x)
        # normed_c = self.norm_condition(c)
        # c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block
        # (but maybe theres a better place to condition than right before the downsample)

        # return self.block(c)

        return self.block(x)


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # self.cpu = torch.device("cpu")
        # self.mps0 = torch.device("mps:0")

    def forward(self, x):
        x_ = x  # .to(self.cpu)

        # Apply 2D Fourier transform to input feature maps
        x_fft = torch.fft.fft2(x_)

        x_fft_real = x_fft.real  # .to(self.mps0)
        # Branch 1: 1x1 Convolution
        x1 = self.conv1(x_fft_real)
        x1 = self.bn1(x1)

        # Branch 2: 3x3 Convolution
        x2 = self.conv2(x_fft_real)
        x2 = self.bn2(x2)
        x2 = F.relu(x2, inplace=True)

        # Branch 3: 3x3 Convolution followed by another 3x3 Convolution
        x3 = self.conv3(x_fft_real)
        x3 = self.bn3(x3)
        x3 = F.relu(x3, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)

        # Concatenate the outputs of the three branches along the channel dimension
        x_out = torch.cat([x1, x2, x3], dim=1)

        # Apply inverse 2D Fourier transform to output feature maps
        x_out = x_out.to(self.cpu)
        x_ifft = torch.fft.ifft2(x_out)

        # Add input feature maps to output feature maps (skip connection)
        # x_out = x + x_ifft.real #=------ cannot add throws error
        x_out = x_ifft.real
        # x_out = x_out.to(self.mps0)

        return x_out


# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, padding=1, bias=True
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, padding=1, bias=True
#         )
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         # Initialize biases to complex tensors with zero imaginary part
#         nn.init.zeros_(self.conv1.bias)
#         nn.init.zeros_(self.conv2.bias)
#         nn.init.zeros_(self.conv3.bias)

#         self.cpu = torch.device("cpu")
#         self.mps0 = torch.device("mps:0")

#     def forward(self, x):
#         x_ = x.to(self.cpu)
#         # Apply 2D Fourier transform to input feature maps
#         x_fft = torch.fft.fftn(x_, dim=(-2, -1))

#         # Branch 1: 1x1 Convolution
#         x1 = self.conv1(x_fft)
#         x1 = self.bn1(x1)

#         # Branch 2: 3x3 Convolution
#         x2 = self.conv2(x_fft)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2, inplace=True)

#         # Branch 3: 3x3 Convolution followed by another 3x3 Convolution
#         x3 = self.conv3(x_fft)
#         x3 = self.bn3(x3)
#         x3 = F.relu(x3, inplace=True)
#         x3 = self.conv3(x3)
#         x3 = self.bn3(x3)

#         # Concatenate the outputs of the three branches along the channel dimension
#         x_out = torch.cat([x1, x2, x3], dim=1)

#         # Apply inverse 2D Fourier transform to output feature maps
#         x_ifft = torch.fft.ifftn(x_out, dim=(-2, -1))

#         # Add input feature maps to output feature maps (skip connection)
#         x_ifft = x_ifft.to(self.mps0)
#         x_out = x + x_ifft.real

#         return x_out


# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, padding=1, bias=True
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, padding=1, bias=True
#         )
#         self.bn3 = nn.BatchNorm2d(out_channels)

#         # Initialize biases to complex tensors with zero imaginary part
#         nn.init.zeros_(self.conv1.bias)
#         nn.init.zeros_(self.conv2.bias)
#         nn.init.zeros_(self.conv3.bias)

#         self.cpu = torch.device("cpu")
#         self.mps0 = torch.device("mps:0")

#     def forward(self, x):
#         x_ = x.to(self.cpu)
#         # Apply 2D Fourier transform to input feature maps
#         x_fft = torch.fft.rfft(x_, signal_ndim=2, onesided=False)

#         # Split real and imaginary parts of FFT output along the channel dimension
#         x_fft_real = x_fft[..., 0].unsqueeze(-1)
#         x_fft_imag = x_fft[..., 1].unsqueeze(-1)
#         x_fft_concat = torch.cat([x_fft_real, x_fft_imag], dim=-1)

#         # Branch 1: 1x1 Convolution
#         x1 = self.conv1(x_fft_concat)
#         x1 = self.bn1(x1)

#         # Branch 2: 3x3 Convolution
#         x2 = self.conv2(x_fft_concat)
#         x2 = self.bn2(x2)
#         x2 = F.relu(x2, inplace=True)

#         # Branch 3: 3x3 Convolution followed by another 3x3 Convolution
#         x3 = self.conv3(x_fft_concat)
#         x3 = self.bn3(x3)
#         x3 = F.relu(x3, inplace=True)
#         x3 = self.conv3(x3)
#         x3 = self.bn3(x3)

#         # Concatenate the outputs of the three branches along the channel dimension
#         x_out_concat = torch.cat([x1, x2, x3], dim=1)

#         # Inverse FFT
#         x_out_concat = x_out_concat.to(self.mps0)
#         x_out = torch.irfft(x_out_concat, signal_ndim=2, onesided=False)

#         return x_out
