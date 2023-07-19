from torch import nn as nn
import torch
import functools

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

from torch.nn import Parameter

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # self.query_conv = SpectralNorm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # self.key_conv = SpectralNorm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        # self.value_conv = SpectralNorm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out


def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    # layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.SiLU())
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            # nn.LeakyReLU(0.2),
            nn.SiLU(),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            nn.Conv2d(128, 1, 8, padding=0)
        )

    def forward(self, img_input):
        return self.model(img_input)

class MultiDiscriminator_d(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator_d, self).__init__()

        # def discriminator_block(in_filters, out_filters, normalize=True):
        #     """Returns downsampling layers of each discriminator block"""
        #     layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        #     if normalize:
        #         layers.append(nn.InstanceNorm2d(out_filters))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        def discriminator_block(in_filters, out_filters, normalization=False):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            # layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.SiLU())
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
                    nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    # nn.LeakyReLU(0.2),
                    nn.SiLU(),
                    nn.InstanceNorm2d(16, affine=True),
                    *discriminator_block(16, 32),
                    *discriminator_block(32, 64),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 128),
                    nn.Conv2d(128, 1, 8, padding=0)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.SiLU())
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


from codes.models.archs.arch_util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
            # layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            layers = [SpectralNorm(nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1))]
            # layers.append(nn.LeakyReLU(0.2, True))
            layers.append(norm_layer(out_filters, affine=True))
            layers.append(nn.SiLU())
            return layers
    
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            # nn.Conv2d(3, 16, 3, stride=2, padding=1),
            SpectralNorm(nn.Conv2d(3, 16, 3, stride=2, padding=1)),
            norm_layer(16),
            # nn.LeakyReLU(0.2, True),
            nn.SiLU(),
            
            Self_Attn(16, "SiLu"),
            *discriminator_block_kha(16, 32, norm_layer),
            *discriminator_block_kha(32, 64, norm_layer),
            *discriminator_block_kha(64, 128, norm_layer),
            *discriminator_block_kha(128, 128, norm_layer),
            # nn.Conv2d(128, 1, 8, padding=0)
            # Self_Attn(128, "SiLu"),
            SpectralNorm(nn.Conv2d(128, 1, 8, padding=0))
        )

    def forward(self, img_input):
        return self.model(img_input)
