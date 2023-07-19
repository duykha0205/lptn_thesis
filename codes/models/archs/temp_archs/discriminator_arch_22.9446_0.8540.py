from torch import nn as nn
import torch
import functools

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

from torch.nn import Parameter


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

from codes.models.archs.arch_util import ActNorm
class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=True,
            bias=True, act_layer=nn.ReLU, norm_layer=ActNorm, gate_layer='sigmoid'):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3
        padw = 1
        
        def discriminator_block_kha(in_filters, out_filters, norm_layer):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            # layers = [SpectralNorm(nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1))]
            # layers.append(nn.LeakyReLU(0.2, True))
            layers.append(norm_layer(out_filters, affine=True))
            layers.append(nn.SiLU())
            
            layers.append(SEModule(out_filters))
            
            return layers
    
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            # SpectralNorm(nn.Conv2d(3, 16, 3, stride=2, padding=1)),
            norm_layer(16),
            # nn.LeakyReLU(0.2, True),
            nn.SiLU(),
            
            # Self_Attn(16, "SiLu"),
            *discriminator_block_kha(16, 32, norm_layer),
            *discriminator_block_kha(32, 64, norm_layer),
            *discriminator_block_kha(64, 128, norm_layer),
            *discriminator_block_kha(128, 128, norm_layer),
            nn.Conv2d(128, 1, 8, padding=0)
            # Self_Attn(128, "SiLu"),
            # SpectralNorm(nn.Conv2d(128, 1, 8, padding=0))
        )

    def forward(self, img_input):
        return self.model(img_input)
