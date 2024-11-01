# # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
from torchvision.ops import DeformConv2d


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization module supporting only 'BCHW' format.

    Args:
        shape (int or tuple): The shape of the input tensor to be normalized.
        eps (float): A small value to prevent division by zero. Default: 1e-6.
        format (str): The data format of the input tensor. Only 'channels_first' is supported. Default: 'channels_first'.
    """

    def __init__(self, shape, eps=1e-6, format="BCHW"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.eps = eps
        self.format = format
        if self.format != "BCHW":
            raise NotImplementedError("Only 'BCHW' format is supported")
        self.shape = (shape,)

    def forward(self, x):
        """
        Forward pass of the Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(1, keepdim=True)  # Compute the mean along the channel dimension
        var = (x - mean).pow(2).mean(1, keepdim=True)  # Compute the variance along the channel dimension
        x = (x - mean) / torch.sqrt(var + self.eps)  # Normalize the tensor
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # Apply weight and bias
        return x


class SGFM(nn.Module):
    """
    Spatial-Gated Feature Modulation (SGFM) module.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features):
        super().__init__()
        intermediate_features = num_features * 2

        self.Conv1 = nn.Conv2d(num_features, intermediate_features, 1, 1, 0)
        self.act_layer = nn.GELU()
        self.DWConv1 = nn.Conv2d(num_features, num_features, 7, 1, 7 // 2, groups=num_features)
        self.Conv2 = nn.Conv2d(num_features, num_features, 1, 1, 0)

        self.norm = LayerNorm(num_features, format='BCHW')
        self.scale = nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        Forward pass of the SGFM.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        shortcut = x.clone()
        x = self.Conv1(self.norm(x))
        x = self.act_layer(x)
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class MLDA(nn.Module):
    """
    Multiscale Large Kernel Decomposition Attention (MLDA) module.

    Args:
        num_features (int): Number of input features.
    LSKA:
        https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention.
    """

    def __init__(self, num_features):
        super().__init__()
        intermediate_features = 2 * num_features

        self.num_features = num_features
        self.intermediate_features = intermediate_features

        self.norm = LayerNorm(num_features, format='BCHW')
        self.scale = nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        # Multiscale Large Kernel Decomposition Attention
        self.LSKA7 = nn.Sequential(
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 7), stride=(1, 1), padding=(0, 7 // 2),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(7, 1), stride=(1, 1), padding=(7 // 2, 0),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 9), stride=(1, 1), padding=(0, (9 // 2) * 4),
                      groups=num_features // 3, dilation=4),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(9, 1), stride=(1, 1), padding=((9 // 2) * 4, 0),
                      groups=num_features // 3, dilation=4),
            nn.Conv2d(num_features // 3, num_features // 3, 1, 1, 0)
        )
        self.LSKA5 = nn.Sequential(
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 5), stride=(1, 1), padding=(0, 5 // 2),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(5, 1), stride=(1, 1), padding=(5 // 2, 0),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 7), stride=(1, 1), padding=(0, (7 // 2) * 3),
                      groups=num_features // 3, dilation=3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(7, 1), stride=(1, 1), padding=((7 // 2) * 3, 0),
                      groups=num_features // 3, dilation=3),
            nn.Conv2d(num_features // 3, num_features // 3, 1, 1, 0)
        )
        self.LSKA3 = nn.Sequential(
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0),
                      groups=num_features // 3),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 // 2) * 2),
                      groups=num_features // 3, dilation=2),
            nn.Conv2d(num_features // 3, num_features // 3, kernel_size=(5, 1), stride=(1, 1), padding=((5 // 2) * 2, 0),
                      groups=num_features // 3, dilation=2),
            nn.Conv2d(num_features // 3, num_features // 3, 1, 1, 0)
        )
        # (r + n) × (r + n) dilation-aware depthwise convolutions
        # r=2, n=1
        self.Dw_2 = nn.Conv2d(num_features // 3, num_features // 3, 3, 1, 1, groups=num_features // 3)
        # r=3, n=2
        self.Dw_3 = nn.Conv2d(num_features // 3, num_features // 3, 5, 1, 5 // 2, groups=num_features // 3)
        # r=4, n=3
        self.Dw_4 = nn.Conv2d(num_features // 3, num_features // 3, 7, 1, 7 // 2, groups=num_features // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(num_features, intermediate_features, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1, 1, 0))

    def forward(self, x):
        """
        Forward pass of the MLDA.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        LDA3 = self.LSKA3(a_1) * self.Dw_2(a_1)
        LDA5 = self.LSKA5(a_1) * self.Dw_3(a_2)
        LDA7 = self.LSKA7(a_1) * self.Dw_4(a_3)
        a = torch.cat([LDA3, LDA5, LDA7], dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x


# MLDAM
class MLDAM(nn.Module):
    """
    Multiscale Large Kernel Decomposition Attention Module (MLDAM).

    This module combines the MLDA and SGFM modules to process the input tensor.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features):
        super().__init__()

        self.MLDA = MLDA(num_features=num_features)  # Initialize the MLDA module
        self.SGFM = SGFM(num_features=num_features)  # Initialize the SGFM module

    def forward(self, x):
        """
        Forward pass of the MLDAM.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = self.MLDA(x)
        x = self.SGFM(x)
        return x


# ADFM
class ADFM(nn.Module):
    """
    Anatomical-aware Dynamic Fusion Module (ADFM).

    This module combines pixel-wise convolution, offset generation, and deformable convolution to process the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default: 3.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ADFM, self).__init__()
        self.pixel_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.pixel_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the ADFM.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = self.pixel_conv1(x)
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.pixel_conv2(x)
        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats, res_scale=1.0):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([
            MLDAM(n_feats) \
            for _ in range(n_resblocks)])

        self.body_t = ADFM(in_channels=n_feats, out_channels=n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)
        x = self.body_t(res) + x

        return x


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


# @ARCH_REGISTRY.register()
class MLDAN(nn.Module):
    def __init__(self, n_resblocks=5, n_resgroups=1, n_colors=3, n_feats=48, scale=3, res_scale=1.0):
        super(MLDAN, self).__init__()

        self.n_resgroups = n_resgroups
        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats, res_scale=res_scale)
            for i in range(n_resgroups)])

        if self.n_resgroups > 1:
            self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)
        if self.n_resgroups > 1:
            res = self.body_t(res) + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == '__main__':
    input_images = (torch.randn(4, 3, 48, 48))
    print(input_images.shape)
    model = MLDAN(scale=4, n_resblocks=24, n_resgroups=1, n_colors=3, n_feats=60, res_scale=1.0)
    out_images = model(input_images)
    flops, params = profile(model, (input_images,))
    print("Mult-Adds: {:.2f} GFlops".format(flops / 1e9))
    print('params: ', params)
    print(out_images.shape)
