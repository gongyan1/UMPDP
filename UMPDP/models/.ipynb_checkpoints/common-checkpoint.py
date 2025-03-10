 # YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

from typing import Optional
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential
from timm.models.layers import trunc_normal_, DropPath
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)  if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() 

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)
    
class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

#         self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = nn.Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input__ in inputs:
            input__ = input__.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input__)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out__ = self._conv_forward(input__, kernels)
            res.append(out__)
        return torch.cat(res, dim=0)


class CondConv(Conv):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, p, g, act)
        self.conv = CondConv2D(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         print("CondConv")
        
    
class BottleneckCond(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckCond, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = CondConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3OD(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3OD, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[BottleneckCond(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
    
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerLayer(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c, num_heads, window_size=7, shift_size=0, 
                mlp_ratio = 4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        if num_heads > 10:
            drop_path = 0.1
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(c)
        self.attn = WindowAttention(
            c, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(c)
        mlp_hidden_dim = int(c * mlp_ratio)
        self.mlp = Mlp(in_features=c, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = ( (0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.tensor(-100.0)).masked_fill(attn_mask == 0, torch.tensor(0.0))
        return attn_mask

    def forward(self, x):
        b, c, w, h = x.shape
        x = x.permute(0, 3, 2, 1).contiguous() # [b,h,w,c]

        attn_mask = self.create_mask(x, h, w) # [nW, Mh*Mw, Mh*Mw]
        shortcut = x
        x = self.norm1(x)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            # print(f"shift size: {self.shift_size}")
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, self.window_size) # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c) # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)  # [B, H', W', C]
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.permute(0, 3, 2, 1).contiguous()
        return x # (b, self.c2, w, h)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
class C3STR(C3):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])
    
class AddC(nn.Module):
    #  Add two tensors
    def __init__(self, arg, c1):
        super(AddC, self).__init__()
        self.arg = arg
        self.conv1 = Conv(2*c1, c1, 1, 1)

    def forward(self, x):
        return self.conv1(torch.cat((x[0], x[1]),dim=1))
    

class Add_idt(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add_idt, self).__init__()
        self.arg = arg

    def forward(self, x):
        return x[2]


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
#             print(type(x[0]), type(x[1][0]))
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
#             print(type(x[0]), type(x[1][1]))
#             print(x[0].shape, x[1][0].shape)
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])
    
def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class Add3(nn.Module):
    #  Add two tensors
    def __init__(self, c1, arg):
        super(Add3, self).__init__()
        self.arg = arg
        self.conv1 = Conv(2*c1, c1, 1, 1)


    def forward(self, x):
        out = torch.cat((x[0], x[1]), dim=1)
        out = shuffle_channels(out, 2)
        return self.conv1(out)
    

class Add4(nn.Module):
    #  Add two tensors
    def __init__(self, c1, arg):
        super(Add4, self).__init__()
        self.arg = arg
        self.conv1 = Conv(c1, c1//2, 1, 1)
        self.conv2 = Conv(c1, c1//2, 1, 1)


    def forward(self, x):
        out = torch.cat((self.conv1(x[0]), self.conv2(x[1])), dim=1)
        out = shuffle_channels(out, 2)
        return out
    
    
class Add5(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index
        self.conv1 = Conv(2*c1, c1, 1, 1)
        self.conv2 = Conv(2*c1, c1, 1, 1)

    def forward(self, x):
        if self.index == 0:
            sum = torch.cat((x[0], x[1][0]), dim=1)
            return self.conv1(sum)
        elif self.index == 1:
            sum = torch.cat((x[0], x[1][1]), dim=1)
            return self.conv2(sum)

    
        
class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)
class ASFFV5(nn.Module):
    def __init__(self, level, rfb=False, vis=False):

    #  512, 256, 128 -> multiplier=1
    # 256, 128, 64 -> multiplier=0.5

        super(ASFFV5, self).__init__()
        self.level = level
        self.multiplier = 1

        self.dim = [int(1024 * self.multiplier), int(512 * self.multiplier),
                    int(256 * self.multiplier)]


        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * self.multiplier), self.inter_dim, 3, 2)

            self.stride_level_2 = Conv(int(256 * self.multiplier), self.inter_dim, 3, 2)

            self.expand = Conv(self.inter_dim, int(
                1024 * self.multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * self.multiplier), self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(
                int(256 * self.multiplier), self.inter_dim, 3, 2)
            self.expand = Conv(self.inter_dim, int(512 * self.multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * self.multiplier), self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(
                int(512 * self.multiplier), self.inter_dim, 1, 1)
            self.expand = Conv(self.inter_dim, int(
                256 * self.multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(
            self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(
            self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(
            compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 , 1, 1)
        self.cls_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.cls_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out


class DMAF1(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(DMAF1, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv(c1, 2*c1, 1, 2)
        self.conv1_t = Conv(c1, 2*c1, 1, 2)
        self.conv2 = Conv(c1, 2*c1, 3, 2)
        self.conv2_t = Conv(c1, 2*c1, 3, 2)

    def forward(self, x, y):
        fdx = x - y#N,C,H,W
        vx = F.tanh(self.GAP(fdx))#N,C,1,1
        x_res = x + vx * y
        fdy = y - x
        vy = F.tanh(self.GAP(fdy))
        y_res = y + vy * x
        x = self.conv1(x)
        x_res = self.conv2(x_res)
        y = self.conv1_t(y)
        y_res = self.conv2_t(y_res)

        return x+x_res, y+y_res

class DMAF(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(DMAF, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv2 = Conv(c1, c1, 3, 1)
        self.conv2_t = Conv(c1, c1, 3, 1)

    def forward(self, z):
        x, y = z[0], z[1]
        fdx = x - y#N,C,H,W
        vx = F.tanh(self.GAP(fdx))#N,C,1,1
        x_res = y + vx * x
        fdy = y - x
        vy = F.tanh(self.GAP(fdy))
        y_res = x + vy * y
        x_res = self.conv2(x_res)
        y_res = self.conv2_t(y_res)

        return x_res, y_res
    



    
class DMAF_SE_AFF1(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE_AFF1, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        
        inter_channels = int(c1 // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)
        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        x_res = y + y * wei + vx * x 
        x_mix = self.conv1(x_res)
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        y_res = x + x * wei + vy * y
        y_mix = self.conv1(y_res)

        return x_mix, y_mix

class DMAF_SE_AFF(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE_AFF, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        
        inter_channels = int(c1 // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)
        
        sub_xy = x - y
        vx = self.sig(self.sub1(sub_xy))
        y_res =x * wei + vx * x  + y
        y_mix = self.conv1(y_res)
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        x_res = y * (1 - wei) + vy * y + x
        x_mix = self.conv1(x_res)

        return x_mix, y_mix
    
    

class DMAF_SE_AFF2(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE_AFF2, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        
        inter_channels = int(c1 // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)
        
        sub_xy = x - y
        vx = self.sig(self.sub1(sub_xy))
        y_res =xlg * wei + vx * x 
        y_mix = self.conv1(y_res)
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        x_res = xlg * wei + vy * y 
        x_mix = self.conv1(x_res)

        return x_mix, y_mix
    
    
    
class DMAF_SE_AFF3(nn.Module):
    def __init__(self, c1, reduction=2):
        super(DMAF_SE_AFF3, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        
        inter_channels = int(c1 // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.sig = nn.Sigmoid()
       
        
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)
        
        xy_same = 2 * x *wei + 2 * y * (1 - wei)
        
        sub_xy = x - y
        vx = self.sig(self.sub1(sub_xy))
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = torch.cat((vx * sub_xy, xy_same), dim=1)
        x_mix = self.conv1(x_res)
        y_res = torch.cat((vy * sub_yx, xy_same), dim=1)
        y_mix = self.conv1(y_res)

        return x_mix, y_mix

    
class DMAF_SE_AFF4(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE_AFF4, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        
        inter_channels = int(c1 // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
    
        
        self.sig = nn.Sigmoid()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)
        
        sub_xy = x - y
        vx = self.sig(self.sub1(sub_xy))
        x_res = vx * x + xlg * wei
       
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        y_res = vy * y +xlg * wei
        

        return self.conv1(y_res), self.conv2(x_res)
    
    
class DMAF_SE_AFF5(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE_AFF5, self).__init__()
        inter_channels = int(c1 // reduction)
        self.sub1 = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),    
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),    
        )
        
        

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),        
       )
    
        
        self.sig = nn.Sigmoid()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
       
    def forward(self, z):
        
        x, y = z[0], z[1]
        
        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        
        
        sub_xy = x - y
        vx = self.sig(self.sub1(sub_xy))
      
       
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        
        
        x_mix = x + vy * xlg
        y_mix = y + vx * xlg 
        

        return self.conv1(x_mix), self.conv2(y_mix)
    
    
class TN_AFF(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_AFF, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True)
        )

        inter_channels = int(c1 // 4)

        self.local_att = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
        )

        self.sig = nn.Sigmoid()
        self.conv2 = Conv(c1, c1, 3, 1)
        self.conv2_t = Conv(c1, c1, 3, 1)

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midy_fuse(y2x)
        x = x + 0.1 * y2x_feat

        sum_xy = x + y
        xl = self.local_att(sum_xy)
        xg = self.global_att(sum_xy)
        xlg = xl + xg
        wei = self.sig(xlg)

        x_mix = wei * x + x
        y_mix = (1 - wei) * y + y
        
        return x_mix, y_mix

    
class TN_DMAF(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True)
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv2 = Conv(c1, c1, 3, 1)
        self.conv2_t = Conv(c1, c1, 3, 1)

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midy_fuse(y2x)
        x = x + 0.1 * y2x_feat

        fdx = x - y#N,C,H,W
        vx = torch.tanh(self.GAP(fdx))#N,C,1,1
        fdy = y - x
        vy = torch.tanh(self.GAP(fdy))
        x_res = x + vy * y
        y_res = y + vx * x
        x_res = self.conv2(x_res)
        y_res = self.conv2_t(y_res)

        return x_res, y_res
    
class SpatialAttention_c(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_c, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        wei = self.sigmoid(x)
        return wei * res


class SDMAF(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(SDMAF, self).__init__()
        inter_channels = int(c1 // 4)

        self.local_att1 = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, c1, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

        self.conv2 = Conv(c1, c1, 1, 1)
        self.conv2_t = Conv(c1, c1, 1, 1)

    def forward(self, z):
        x, y = z[0], z[1]

        fdx = x - y
        xl = self.local_att1(fdx)
        xg = self.global_att1(fdx)
        xy_wei = self.sigmoid(xl + xg)

        fdy = y - x
        yl = self.local_att2(fdy)
        yg = self.global_att2(fdy)
        yx_wei = self.sigmoid(yl + yg)

        x_res = x + y * yx_wei
        y_res = y + x * xy_wei
        x_res = self.conv2(x_res)
        y_res = self.conv2_t(y_res)


        return x_res, y_res
    

class Adds(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index
        self.spatial1 = SpatialAttention_c()
        self.spatial2 = SpatialAttention_c()

    def forward(self, x):
        if self.index == 0:
           

            return self.spatial1(torch.add(x[0], x[1][0]))
        elif self.index == 1:

            return self.spatial2(torch.add(x[0], x[1][1]))

        
class TN(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midy_fuse(y2x)
        x_mix = y2x_feat
        
        mid_x = self.x_t(x)
        x2y = (alpha + 1) * mid_x + beta
        x2y_feat = self.midy_fuse(x2y)
        y_mix = x2y_feat

        
        return x_mix, y_mix, torch.add(x, y)
    
class DMAF_SE(nn.Module):
    def __init__(self, c1, reduction=4):
        super(DMAF_SE, self).__init__()
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, z):
        x, y = z[0], z[1]
        sub_xy = x - y#N,C,H,W
   
        vx = torch.tanh(self.sub1(sub_xy))
        sub_yx = y - x
        vy = torch.tanh(self.sub2(sub_yx))
        
        x_res = y + vx * x
        x_mix = self.conv1(x_res)
        
        y_res = x + vy * y
        y_mix = self.conv1(y_res)

        return x_mix, y_mix, torch.add(x, y)


    
class TN1(nn.Module):
    def __init__(self, c1):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN1, self).__init__()
        self.x_tr = Conv(c1, c1, 1, 1)
        self.y_tr = Conv(c1, c1, 1, 1)

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x_feat = self.midy_fuse(y2x)
        x_mix = y2x_feat
        
        mid_x = self.x_t(x)
        x2y = (alpha + 1) * mid_x + beta
        x2y_feat = self.midy_fuse(x2y)
        y_mix = x2y_feat

        
        return x_mix, y_mix
    
    


class C3HB(nn.Module):
    # CSP HorBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(HorBlock(c_) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class HorLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError # by iscyy/air
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape gnconv [512]by iscyy/air
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class HorBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = HorLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
    
class TN_DMAF7(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF7, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=1,stride=1,padding=0,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.convx3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1,1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.convy3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
#         y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x)
        x_mix = self.convx3(y2x_feat + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
#         x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y)
        y_mix = self.convy3(x2y_feat + y)
        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y + x + x_mix
        x_fusion = self.conv1(x_res)
        
        y_res = vx * x + y + y_mix
        y_fusion = self.conv2(y_res)
        
        return x_fusion, y_fusion
    
    
    
class TN_DMAF8(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF8, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
#         self.convx3 = nn.Sequential(
#             nn.Conv2d(c1, c1, 3, 1,1),
#             nn.BatchNorm2d(c1),
#             nn.ReLU()
#         )
#         self.convy3 = nn.Sequential(
#             nn.Conv2d(c1, c1, 3, 1, 1),
#             nn.BatchNorm2d(c1),
#             nn.ReLU()
#         )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
#         y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x + x)
#         x_mix = y2x_feat
#         x_mix = self.convx3(y2x_feat + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
#         x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y + y)
#         y_mix = x2y_feat
#         y_mix = self.convy3(x2y_feat + y)
        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y + x + y2x_feat
        x_fusion = self.conv1(x_res)
        
        y_res = vx * x + y + x2y_feat
        y_fusion = self.conv2(y_res)
        
        return x_fusion, y_fusion
    
    
    
class TN_DMAF9(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF9, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.convx3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1,1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.convy3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
#         y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x)
        x_mix = y2x_feat
        x_mix = self.convx3(y2x_feat + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
#         x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y)
        y_mix = x2y_feat
        y_mix = self.convy3(x2y_feat + y)
        
        sub_xy = x_mix - y_mix#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y_mix - x_mix
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y_mix + x_mix + y2x_feat
        x_mix = self.conv1(x_res)
        
        y_res = vx * x_mix + y_mix + x2y_feat
        y_mix = self.conv2(y_res)
        
        return x_mix , y_mix
    
    
class TN_DMAF10(nn.Module):
    def __init__(self, c1,reduction=2):#c1,c2用不到，因为输入和输出通道数是完全一样的，无需改变通道数。为了模型保持一致所以这里加上。
        super(TN_DMAF10, self).__init__()
        self.x_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.y_tr = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )

        self.alpha = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.alpha1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)
        self.beta1 = nn.Conv2d(2*c1,c1,kernel_size=1,stride=1,padding=0,bias=True)

        self.y_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midy_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.x_t = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=True)
        self.midx_fuse = nn.Sequential(
            nn.Conv2d(c1,c1,kernel_size=3,stride=1,padding=1,bias=True),
#             nn.ReLU(inplace=True)
        )
        
        self.sub1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.sub2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(c1, c1 // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(c1 // reduction),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1, stride=1, padding=0, bias=False),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.convx3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1,1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.convy3 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        
        self.sig = nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
    

    def forward(self, z):
        x, y = z[0], z[1]

        x_tr = self.x_tr(x)
        y_tr = self.y_tr(y)

        xy_cat = torch.cat((x_tr, y_tr), dim=1)

        alpha = self.alpha(xy_cat)
        beta = self.beta(xy_cat)
        
        alpha1 = self.alpha1(xy_cat)
        beta1 = self.beta1(xy_cat)

        mid_y = self.y_t(y)
        y2x = (alpha + 1) * mid_y + beta
        y2x = self.relu(y2x)
        y2x_feat = self.midx_fuse(y2x)
#         x_mix = y2x_feat
        x_mix = self.convx3(y2x_feat + x)
        
        mid_x = self.x_t(x)
        x2y = (alpha1 + 1) * mid_x + beta1
        x2y = self.relu(x2y)
        x2y_feat = self.midy_fuse(x2y)
#         y_mix = x2y_feat
        y_mix = self.convy3(x2y_feat + y)
        
        sub_xy = x - y#N,C,H,W
        vx = self.sig(self.sub1(sub_xy))
        
        sub_yx = y - x
        vy = self.sig(self.sub2(sub_yx))
        
        x_res = vy * y + x + x_mix
        x_fusion = self.conv1(x_res)
        
        y_res = vx * x + y + y_mix
        y_fusion = self.conv2(y_res)
        
        return x_fusion, y_fusion
    