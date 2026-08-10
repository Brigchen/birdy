# -*- coding: utf-8 -*-
"""
AI 图像增强模块：Real-ESRGAN 降噪 + OmniSR 锐化（不放大）

串联流水线：
  原图 → Real-ESRGAN (outscale=1.0，仅降噪) → OmniSR (x2 推理后缩回原尺寸，仅锐化) → 输出

模型权重（放入 models/ 目录）：
  - realesr-general-x4v3.pth   （Real-ESRGAN 通用降噪）
  - OmniSR_X2.pth              （OmniSR x2 锐化，官方预训练权重）

依赖（可选，未安装时本模块自动降级跳过）：
  pip install einops safetensors
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  （SRVGGNetCompact / NAFNet / OmniSR 架构均自包含，无需 basicsr/realesrgan）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import piexif
except Exception:
    piexif = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

# ==================== OmniSR 架构（基于官方实现，自包含） ====================
# Source: https://github.com/Francis0625/Omni-SR (CVPR 2023)
# 架构文件: components/OmniSR.py, ops/OSAG.py, ops/OSA.py, ops/esa.py, ops/layernorm.py, ops/pixelshuffle.py

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange, Reduce
    _EINOPS_OK = True
except Exception:
    _EINOPS_OK = False


class _LayerNorm2d(nn.Module):
    """LayerNorm for CHW tensor (from ops/layernorm.py)."""

    class _LayerNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            ctx.eps = eps
            N, C, H, W = x.size()
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + eps).sqrt()
            ctx.save_for_backward(y, var, weight)
            y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            eps = ctx.eps
            N, C, H, W = grad_output.size()
            y, var, weight = ctx.saved_variables
            g = grad_output * weight.view(1, C, 1, 1)
            mean_g = g.mean(dim=1, keepdim=True)
            mean_gy = (g * y).mean(dim=1, keepdim=True)
            gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
            return (
                gx,
                (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
                grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
                None,
            )

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return self._LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class _ESA(nn.Module):
    """Enhanced Spatial Attention (from ops/esa.py)."""

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super().__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# --- OSA helpers (from ops/OSA.py) ---

def _exists(val):
    return val is not None


def _default(val, d):
    return val if _exists(val) else d


class _PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class _Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = _LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class _Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.0):
        super().__init__()
        hidden_features = int(dim * mult)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class _SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class _Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device
        if self.prob == 0.0 or (not self.training):
            return x
        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


class _MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = _Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


def _MBConv(dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        nn.GELU(),
        _SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
    )
    if dim_in == dim_out and not downsample:
        net = _MBConvResidual(net, dropout=dropout)
    return net


class _Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7, with_pe=True):
        super().__init__()
        assert (dim % dim_head) == 0
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(grid, "j ... -> 1 j ...")
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
            self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape,
            x.device,
            self.heads,
        )
        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, "i j h -> h i j")
        attn = self.attend(sim)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class _Channel_Attention(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)", ph=self.ps, pw=self.ps, head=self.heads
            ),
            qkv,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)", h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps, head=self.heads
        )
        out = self.project_out(out)
        return out


class _Channel_Attention_grid(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)", ph=self.ps, pw=self.ps, head=self.heads
            ),
            qkv,
        )
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)", h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps, head=self.heads
        )
        out = self.project_out(out)
        return out


class _OSA_Block(nn.Module):
    """Omni Spatial Attention Block (from ops/OSA.py)."""

    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super().__init__()
        w = window_size
        self.layer = nn.Sequential(
            _MBConv(channel_num, channel_num, downsample=False, expansion_rate=1, shrinkage_rate=0.25),
            Rearrange("b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w),
            _PreNormResidual(
                channel_num, _Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout, window_size=window_size, with_pe=with_pe)
            ),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            _Conv_PreNormResidual(channel_num, _Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            _Conv_PreNormResidual(channel_num, _Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            _Conv_PreNormResidual(channel_num, _Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            Rearrange("b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w),
            _PreNormResidual(
                channel_num, _Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout, window_size=window_size, with_pe=with_pe)
            ),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
            _Conv_PreNormResidual(channel_num, _Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
            _Conv_PreNormResidual(channel_num, _Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            _Conv_PreNormResidual(channel_num, _Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        return self.layer(x)


class _OSAG(nn.Module):
    """Omni Spatial Attention Group (from ops/OSAG.py, 简化为直接使用 OSA_Block)."""

    def __init__(self, channel_num=64, bias=True, block_num=4, window_size=8, pe=True, ffn_bias=False, **kwargs):
        super().__init__()
        group_list = []
        for _ in range(block_num):
            group_list.append(_OSA_Block(channel_num, bias, ffn_bias=ffn_bias, window_size=window_size, with_pe=pe))
        group_list.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = _ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


def _pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False):
    """PixelShuffle upsampling block (from ops/pixelshuffle.py)."""
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size, padding=1, bias=bias)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class OmniSR(nn.Module):
    """OmniSR 主模型 (from components/OmniSR.py)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super().__init__()
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]
        residual_layer = []
        self.res_num = res_num
        for _ in range(res_num):
            residual_layer.append(_OSAG(channel_num=num_feat, **kwargs))
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = _pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)
        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        residual = self.input(x)
        out = self.residual_layer(residual)
        out = torch.add(self.output(out), residual)
        out = self.up(out)
        out = out[:, :, : H * self.up_scale, : W * self.up_scale]
        return out


# OmniSR X2 默认配置（来自 train_yamls/train_OmniSR_X2_DF2K.yaml）
def build_omnisr_x2() -> OmniSR:
    return OmniSR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        res_num=5,
        upsampling=2,
        bias=True,
        block_num=1,
        block_script_name="OSA",  # 保留用于兼容，实际不使用动态导入
        block_class_name="OSA_Block",
        window_size=8,
        pe=True,
        ffn_bias=True,
    )


# ==================== SRVGGNetCompact 架构（Real-ESRGAN 通用降噪/超分） ====================
# Source: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
# 用于 realesr-general-x4v3.pth 权重（num_feat=64, num_conv=32, upscale=4, act_type='prelu'）
# 注意：此权重并非 RRDBNet 架构，而是更轻量的 SRVGGNetCompact。


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    所有 conv 层均在 self.body 中（包括首尾 conv）；PReLU 作为激活函数。
    forward 输出为 (H*upscale, W*upscale)。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        self.body.append(self._make_act(act_type, num_feat))

        # the body convolutions
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_act(act_type, num_feat))

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_act(act_type, num_feat):
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            return nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f'Unsupported act_type: {act_type}')

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


def build_srvgg_realesr_general_x4v3() -> SRVGGNetCompact:
    """realesr-general-x4v3.pth 模型配置（num_feat=64, num_conv=32, upscale=4）。"""
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')


# ==================== NAFNet 架构（ECCV 2022, Simple Baselines for Image Restoration） ====================
# Source: https://github.com/megvii-research/NAFNet
# 优势：比 Real-ESRGAN 降噪更干净，无油画感；速度快 3x；原生降噪（非超分降级）


class _NAFNet_LayerNorm2d(nn.Module):
    """NAFNet 官方 LayerNorm2d（参数名 weight/bias，shape (channels,)）。

    与 basicsr/models/archs/arch_util.py 的 LayerNorm2d 完全一致，
    以兼容官方预训练权重（NAFNet-SIDD-width64.pth）。
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        return y


class _SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = _SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = _NAFNet_LayerNorm2d(c)
        self.norm2 = _NAFNet_LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet 主模型 (from basicsr/models/archs/NAFNet_arch.py)."""

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[_NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[_NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def build_nafnet_sidd_width64() -> NAFNet:
    """NAFNet SIDD 去噪配置（width=64, 官方预训练权重）。"""
    return NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )


# ==================== Tile 分块推理 ====================


def _tile_inference(
    model: nn.Module,
    img_tensor: torch.Tensor,
    tile_size: int = 512,
    tile_pad: int = 16,
    scale: int = 2,
) -> torch.Tensor:
    """
    分块推理，避免大图爆显存。
    img_tensor: (1, C, H, W) on device
    返回: (1, C, H*scale, W*scale)
    """
    _, c, h, w = img_tensor.shape
    if h <= tile_size and w <= tile_size:
        return model(img_tensor)

    out_h, out_w = h * scale, w * scale
    output = img_tensor.new_zeros(1, c, out_h, out_w)

    h_idx = 0
    while h_idx < h:
        w_idx = 0
        h_end = min(h_idx + tile_size, h)
        h_start_pad = max(0, h_idx - tile_pad)
        h_end_pad = min(h, h_end + tile_pad)

        while w_idx < w:
            w_end = min(w_idx + tile_size, w)
            w_start_pad = max(0, w_idx - tile_pad)
            w_end_pad = min(w, w_end + tile_pad)

            tile = img_tensor[:, :, h_start_pad:h_end_pad, w_start_pad:w_end_pad]
            tile_out = model(tile)

            # 裁掉 padding 区域，只保留核心
            crop_h_start = (h_idx - h_start_pad) * scale
            crop_h_end = crop_h_start + (h_end - h_idx) * scale
            crop_w_start = (w_idx - w_start_pad) * scale
            crop_w_end = crop_w_start + (w_end - w_idx) * scale

            output[:, :, h_idx * scale : h_end * scale, w_idx * scale : w_end * scale] = tile_out[
                :, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end
            ]

            w_idx = w_end
        h_idx = h_end

    return output


# ==================== 模型加载（懒加载 + 缓存） ====================

_denoise_srvgg: Optional[SRVGGNetCompact] = None  # SRVGGNetCompact 实例（Real-ESRGAN 降噪）
_denoise_nafnet: Optional[NAFNet] = None
_sharpen_model: Optional[OmniSR] = None
_device = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def _get_denoise_model(model_path: str) -> Optional[SRVGGNetCompact]:
    """加载 Real-ESRGAN (realesr-general-x4v3, SRVGGNetCompact) 降噪模型。"""
    global _denoise_srvgg
    if _denoise_srvgg is not None:
        return _denoise_srvgg

    model = build_srvgg_realesr_general_x4v3()
    try:
        try:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            if "params" in sd:
                sd = sd["params"]
            elif "params_ema" in sd:
                sd = sd["params_ema"]
            elif "state_dict" in sd:
                sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[ai_enhance] Real-ESRGAN 权重加载失败: {e}", flush=True)
        return None

    model.eval()
    model = model.to(_get_device())
    _denoise_srvgg = model
    return model


def _get_denoise_nafnet(model_path: str) -> Optional[NAFNet]:
    """加载 NAFNet (SIDD width64) 降噪模型。"""
    global _denoise_nafnet
    if _denoise_nafnet is not None:
        return _denoise_nafnet
    model = build_nafnet_sidd_width64()
    try:
        try:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            if "params" in sd:
                sd = sd["params"]
            elif "params_ema" in sd:
                sd = sd["params_ema"]
            elif "state_dict" in sd:
                sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[ai_enhance] NAFNet 权重加载失败: {e}", flush=True)
        return None
    model.eval()
    model = model.to(_get_device())
    _denoise_nafnet = model
    return model


def _get_sharpen_model(model_path: str) -> Optional[OmniSR]:
    """加载 OmniSR X2 锐化模型。"""
    global _sharpen_model
    if _sharpen_model is not None:
        return _sharpen_model
    if not _EINOPS_OK:
        print("[ai_enhance] 缺少 einops，无法加载 OmniSR。请 pip install einops", flush=True)
        return None

    model = build_omnisr_x2()
    try:
        try:
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
        # 兼容多种保存格式：{"params": sd}, {"params_ema": sd}, 训练检查点嵌套
        if isinstance(sd, dict):
            if "params" in sd:
                sd = sd["params"]
            elif "params_ema" in sd:
                sd = sd["params_ema"]
            # 训练检查点可能含 state_dict 键
            elif "state_dict" in sd:
                sd = sd["state_dict"]
        # 过滤 thop 性能分析元数据（total_ops / total_params 等非参数键）
        sd = {k: v for k, v in sd.items() if not k.endswith((".total_ops", ".total_params"))}
        sd = {k: v for k, v in sd.items() if k not in ("total_ops", "total_params")}
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[ai_enhance] OmniSR 权重加载失败: {e}", flush=True)
        return None

    model.eval()
    model = model.to(_get_device())
    _sharpen_model = model
    return model


# ==================== 降噪 / 锐化函数 ====================


def _denoise_bgr(
    img_bgr: np.ndarray,
    strength: float = 0.5,
    model_path: str = "",
    tile_size: int = 512,
    tile_pad: int = 10,
) -> np.ndarray:
    """
    Real-ESRGAN (SRVGGNetCompact) 降噪。
    模型原生 x4 超分；推理后缩回原尺寸作为降噪结果。
    strength: 0=原图，1=完全降噪，默认 0.5
    """
    model = _get_denoise_model(model_path)
    if model is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    device = _get_device()

    # BGR -> RGB -> tensor (1,3,H,W) float32 [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                out_tensor = _tile_inference(model, tensor, tile_size=tile_size, tile_pad=tile_pad, scale=4)
        else:
            out_tensor = _tile_inference(model, tensor, tile_size=tile_size, tile_pad=tile_pad, scale=4)

    # SRVGGNetCompact 输出 x4，缩回原尺寸作为降噪结果
    out_tensor = F.interpolate(out_tensor, size=(h, w), mode='bilinear', align_corners=False)

    denoised_np = out_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    denoised_np = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)
    denoised_bgr = cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR)

    # strength 混合：out = orig + strength * (denoised - orig)
    if strength >= 1.0:
        return denoised_bgr
    if strength <= 0.0:
        return img_bgr
    orig = img_bgr.astype(np.float32)
    denoised = denoised_bgr.astype(np.float32)
    mixed = orig + strength * (denoised - orig)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _denoise_bgr_nafnet(
    img_bgr: np.ndarray,
    strength: float = 1.0,
    model_path: str = "",
    tile_size: int = 512,
    tile_pad: int = 16,
) -> np.ndarray:
    """
    NAFNet 降噪（原生降噪，不放大）。
    NAFNet 无内置 strength 参数，通过混合实现：out = orig + strength * (denoised - orig)
    strength: 0=原图，1=完全降噪，默认 1.0
    """
    model = _get_denoise_nafnet(model_path)
    if model is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    device = _get_device()

    # BGR -> RGB -> tensor (1,3,H,W) float32 [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        # NAFNet scale=1（不放大），复用 tile 推理
        if _get_device().type == "cuda":
            with torch.cuda.amp.autocast():
                out_tensor = _tile_inference(model, tensor, tile_size=tile_size, tile_pad=tile_pad, scale=1)
        else:
            out_tensor = _tile_inference(model, tensor, tile_size=tile_size, tile_pad=tile_pad, scale=1)

    denoised_np = out_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    denoised_np = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)
    denoised_bgr = cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR)

    # strength 混合
    if strength >= 1.0:
        return denoised_bgr
    orig_f = img_bgr.astype(np.float32)
    den_f = denoised_bgr.astype(np.float32)
    out = orig_f + strength * (den_f - orig_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sharpen_bgr(
    img_bgr: np.ndarray,
    strength: float = 0.5,
    model_path: str = "",
    tile_size: int = 512,
    tile_pad: int = 16,
) -> np.ndarray:
    """
    OmniSR 锐化（不放大）：
    1. OmniSR x2 推理 → 2 倍大图
    2. Lanczos 缩回原尺寸
    3. 混合: output = original + strength * (sr_downsampled - original)
    """
    model = _get_sharpen_model(model_path)
    if model is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    device = _get_device()

    # BGR -> RGB -> tensor (1,3,H,W) float32 [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_out = _tile_inference(model, tensor, tile_size=tile_size, tile_pad=tile_pad, scale=2)

    # (1,3,2H,2W) -> numpy (2H,2W,3)
    sr_np = sr_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)

    # 缩回原尺寸 (Lanczos)
    sr_down = cv2.resize(sr_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    sr_down = cv2.cvtColor(sr_down, cv2.COLOR_RGB2BGR)

    # 混合
    orig_f = img_bgr.astype(np.float32)
    sr_f = sr_down.astype(np.float32)
    out = orig_f + strength * (sr_f - orig_f)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==================== EXIF 传递 ====================


def _read_exif(path: str) -> Optional[bytes]:
    """读取原图 EXIF bytes。"""
    if piexif is None:
        return None
    try:
        return piexif.load(path)
    except Exception:
        return None


def _write_exif(path: str, exif_data) -> None:
    """将 EXIF 写入图片文件。"""
    if piexif is None or exif_data is None:
        return
    try:
        piexif.insert(piexif.dump(exif_data), path)
    except Exception:
        pass


# ==================== 主类 ====================


class AIEnhancer:
    """
    AI 图像增强：降噪（Real-ESRGAN 或 NAFNet 可选）+ OmniSR 锐化。

    用法:
        enhancer = AIEnhancer(model_dir="models")
        # Real-ESRGAN 降噪
        img = enhancer.enhance_pil(img, denoise=True, denoise_model="realesrgan", denoise_strength=0.5,
                                    sharpen=True, sharpen_strength=0.4)
        # NAFNet 降噪
        img = enhancer.enhance_pil(img, denoise=True, denoise_model="nafnet", denoise_strength=1.0,
                                    sharpen=True, sharpen_strength=0.4)
    """

    # 降噪模型文件名映射
    DENOISE_MODELS = {
        "realesrgan": "realesr-general-x4v3.pth",
        "nafnet": "NAFNet-SIDD-width64.pth",
    }

    def __init__(
        self,
        model_dir: str = "",
        tile_size: int = 512,
        tile_pad: int = 16,
    ):
        self.model_dir = model_dir or str(Path(__file__).resolve().parent.parent / "models")
        self.tile_size = tile_size
        self.tile_pad = tile_pad

    def _resolve_model(self, name: str) -> str:
        """解析模型路径；支持候选列表（| 分隔），返回第一个存在的文件。"""
        candidates = [n.strip() for n in name.split("|") if n.strip()]
        if not candidates:
            candidates = [name]
        for n in candidates:
            p = Path(self.model_dir) / n
            if p.is_file():
                return str(p)
        # 都不存在则返回第一个候选名（保留原行为，由调用方报错）
        return str(Path(self.model_dir) / candidates[0])

    def enhance_bgr(
        self,
        img_bgr: np.ndarray,
        denoise: bool = False,
        denoise_strength: float = 0.5,
        sharpen: bool = False,
        sharpen_strength: float = 0.5,
        denoise_model: str = "realesrgan",
    ) -> np.ndarray:
        """对 BGR numpy 图像执行 AI 增强。denoise_model: "realesrgan" 或 "nafnet"。"""
        if denoise:
            model_file = self.DENOISE_MODELS.get(denoise_model, self.DENOISE_MODELS["realesrgan"])
            path = self._resolve_model(model_file)
            if os.path.isfile(path):
                if denoise_model == "nafnet":
                    img_bgr = _denoise_bgr_nafnet(
                        img_bgr, strength=denoise_strength, model_path=path,
                        tile_size=self.tile_size, tile_pad=self.tile_pad,
                    )
                else:
                    img_bgr = _denoise_bgr(
                        img_bgr, strength=denoise_strength, model_path=path,
                        tile_size=self.tile_size, tile_pad=self.tile_pad,
                    )
            else:
                print(f"[ai_enhance] 降噪模型不存在: {path}", flush=True)

        if sharpen:
            # 支持多种 OmniSR 权重文件名（| 分隔，按顺序查找第一个存在的）
            path = self._resolve_model(
                "OmniSR_X2.pth|epoch896_OmniSR.pth|epoch885_OmniSR.pth"
            )
            if os.path.isfile(path):
                img_bgr = _sharpen_bgr(
                    img_bgr, strength=sharpen_strength, model_path=path,
                    tile_size=self.tile_size, tile_pad=self.tile_pad,
                )
            else:
                print(f"[ai_enhance] 锐化模型不存在: {path}", flush=True)

        return img_bgr

    def enhance_pil(
        self,
        img: Image.Image,
        denoise: bool = False,
        denoise_strength: float = 0.5,
        sharpen: bool = False,
        sharpen_strength: float = 0.5,
        denoise_model: str = "realesrgan",
    ) -> Image.Image:
        """对 PIL 图像执行 AI 增强，返回 PIL 图像。denoise_model: "realesrgan" 或 "nafnet"。"""
        if not denoise and not sharpen:
            return img

        # PIL RGB -> BGR numpy
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        bgr = self.enhance_bgr(
            bgr, denoise=denoise, denoise_strength=denoise_strength,
            sharpen=sharpen, sharpen_strength=sharpen_strength,
            denoise_model=denoise_model,
        )

        # BGR numpy -> PIL RGB
        rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_out)

    def enhance_file(
        self,
        input_path: str,
        output_path: str,
        denoise: bool = False,
        denoise_strength: float = 0.5,
        sharpen: bool = False,
        sharpen_strength: float = 0.5,
        denoise_model: str = "realesrgan",
    ) -> bool:
        """
        对文件执行 AI 增强，保留 EXIF。
        返回是否成功。denoise_model: "realesrgan" 或 "nafnet"。
        """
        exif_data = _read_exif(input_path)

        bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return False

        bgr = self.enhance_bgr(
            bgr, denoise=denoise, denoise_strength=denoise_strength,
            sharpen=sharpen, sharpen_strength=sharpen_strength,
            denoise_model=denoise_model,
        )

        ok = cv2.imwrite(output_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            _write_exif(output_path, exif_data)
        return ok


# ==================== CLI（独立测试用） ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 图像增强 (降噪 + OmniSR 锐化)")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default="", help="输出路径（默认覆盖）")
    parser.add_argument("--denoise", action="store_true", help="启用降噪")
    parser.add_argument("--sharpen", action="store_true", help="启用锐化")
    parser.add_argument("--denoise-model", default="realesrgan", choices=["realesrgan", "nafnet"], help="降噪模型")
    parser.add_argument("--denoise-strength", type=float, default=0.5, help="降噪强度 0-1")
    parser.add_argument("--sharpen-strength", type=float, default=0.5, help="锐化强度 0-1")
    parser.add_argument("--model-dir", default="models", help="模型目录")
    parser.add_argument("--tile-size", type=int, default=512, help="分块大小")
    args = parser.parse_args()

    out = args.output or args.input
    enhancer = AIEnhancer(model_dir=args.model_dir, tile_size=args.tile_size)
    ok = enhancer.enhance_file(
        args.input, out,
        denoise=args.denoise, denoise_strength=args.denoise_strength,
        sharpen=args.sharpen, sharpen_strength=args.sharpen_strength,
        denoise_model=args.denoise_model,
    )
    print(f"{'成功' if ok else '失败'}: {out}")
