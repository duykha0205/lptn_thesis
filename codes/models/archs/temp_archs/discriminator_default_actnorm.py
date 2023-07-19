from torch import nn as nn
from codes.models.archs.arch_util import ActNorm


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
        
        
        def discriminator_block(in_filters, out_filters, norm_layer):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            layers.append(norm_layer(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            
            return layers
    
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            norm_layer(16),
            nn.SiLU(),
            *discriminator_block(16, 32, norm_layer),
            *discriminator_block(32, 64, norm_layer),
            *discriminator_block(64, 128, norm_layer),
            *discriminator_block(128, 128, norm_layer),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)