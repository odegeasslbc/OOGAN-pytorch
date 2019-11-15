import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UnFlatten(nn.Module):
    def __init__(self, block_size):
        super(UnFlatten, self).__init__()
        self.block_size = block_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.block_size, self.block_size)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def cos_similarity(weight):
    weight = weight / weight.norm(dim=-1).unsqueeze(-1)
    cos_distance = torch.mm(weight, weight.transpose(1,0))
    return cos_distance.pow(2).mean()

class OrthorConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0, bias=True, groups=1):
        super(OrthorConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias, groups=groups)
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-5, betas=(0.5, 0.99))
        self.out_channel = out_channel
        self.in_channel = in_channel

    def orthogonal_update(self):
        self.zero_grad()
        loss = cos_similarity(self.conv.weight.view(self.in_channel*self.out_channel//self.groups, -1))
        loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        if self.training:
            self.orthogonal_update()
        return self.conv(feat)

class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw):
        super(OrthorTransform, self).__init__()

        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, c_dim, feat_hw, feat_hw))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.99))

    def orthogonal_update(self):
        self.zero_grad()
        loss = cos_similarity(self.weight.view( self.c_dim, -1))
        loss.backward()
        self.opt_orth.step()

    def forward(self, feat):
        if self.training:
            self.orthogonal_update()
        pred = feat * self.weight.expand_as(feat)
        return pred.mean(-1).mean(-1)


# Q module that utilzes the orthogonal regularized conv and transformer layers
class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_c, feat_hw, prob=True):
        super(CodeReduction, self).__init__()
        if prob:
            c_dim *= 2
        
        self.main = nn.Sequential(
            nn.Conv2d(feat_c, c_dim, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(0.1),
            OrthorConv2d(c_dim, c_dim, 4, 2, 1, bias=True, groups=c_dim)
        )

        self.trans = OrthorTransform(c_dim=c_dim, feat_hw=feat_hw//2)
    
    def forward(self, feat):
        pred_c = self.trans( self.main(feat) )
        return pred_c.view(feat.size(0), -1)


class ChannelAttentionMask(nn.Module):
    def __init__(self, c_dim, feat_c, feat_hw):
        super().__init__()
        self.feat_c = feat_c
        self.feat_hw = feat_hw

        self.instance_attention = nn.Parameter(torch.randn(1,feat_c,feat_hw*feat_hw))
        self.channel_attention = nn.Sequential(
            nn.Linear(c_dim, feat_c), nn.ReLU(), nn.Linear(feat_c, feat_c), UnFlatten(1)
        )
    def forward(self, c):
        #instance_mask = torch.softmax(self.instance_attention, dim=-1).view(1, self.feat_c, self.feat_hw, self.feat_hw)
        channel_mask = self.channel_attention(c)
        return self.feat_c*channel_mask


class Upscale2d(nn.Module):
    def __init__(self, factor):
        super(Upscale2d, self).__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        s = x.size()
        x = x.view(-1, s[1], s[2], 1, s[3], 1)
        x = x.expand(-1, s[1], s[2], self.factor, s[3], self.factor)
        x = x.contiguous().view(-1, s[1], s[2] * self.factor, s[3] * self.factor)
        return x



class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.main = nn.Sequential(
                        Upscale2d(factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
                        nn.BatchNorm2d(out_channel),
                        nn.LeakyReLU(0.1),
                    )
    
    def forward(self, x):
        return self.main(x)


class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel ):
        super().__init__()

        self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True),
                nn.BatchNorm2d(out_channel), 
                nn.AvgPool2d(2, 2),
                nn.LeakyReLU(0.2),)

    def forward(self, x):
        return self.main(x)



 
class OOGANInput(nn.Module):
    """
    The OOGAN input module, the compteting-free input
    ...

    Attributes
    ----------
    c_dim : int
        number of dimensions in control vector c
    z_dim : int
        number of dimensions in noise vector z
    feat_4 : int
        feature's channel dimension at 4x4 level
    feat_8 : int
        feature's channel dimension at 8x8 level

    Methods
    -------
    forward(c=None, z=None): Tensor
        returns the feature map
    """
    def __init__(self, c_dim, z_dim, feat_4, feat_8):
        super().__init__()

        self.init_noise = nn.Parameter(torch.randn(1, feat_4, 4, 4))
        
        self.from_c_4 = nn.Sequential(
            UnFlatten(1), 
            nn.ConvTranspose2d(c_dim, feat_4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(feat_4), 
            nn.LeakyReLU(0.01))

        self.from_c_8 = UpConvBlock(feat_4, feat_8)
        
        self.z_dim = z_dim
        if z_dim > 0:
            self.attn_from_c = ChannelAttentionMask(c_dim, feat_8, 8)
            self.from_z_8 = nn.Sequential(
                UnFlatten(1), 
                nn.ConvTranspose2d(z_dim, z_dim//2, 4, 1, 0, bias=True),
                nn.BatchNorm2d(z_dim//2), 
                nn.LeakyReLU(0.01),
                UpConvBlock(z_dim//2, feat_8)
                )

    def forward(self, c, z=None):
        #feat = self.init_noise.expand(c.size(0), -1, -1, -1)
        feat = self.from_c_4(c) #+ feat
        feat = self.from_c_8(feat)
        if self.z_dim>0 and z is not None:
            attn_from_c = self.attn_from_c(c)
            feat = attn_from_c*self.from_z_8(z) + feat
        return feat

class InfoGANInput(nn.Module):
    def __init__(self, c_dim, z_dim, feat_4, feat_8):
        super().__init__()

        self.z_dim = z_dim

        self.main = nn.Sequential(
            UnFlatten(1), 
            nn.ConvTranspose2d(c_dim+z_dim, feat_4, 4, 1, 0, bias=True),
            nn.BatchNorm2d(feat_4), 
            nn.LeakyReLU(0.01),
            UpConvBlock(feat_4, feat_8))
                
    def forward(self, c, z=None):
        if self.z_dim>0 and z is not None:
            c = torch.cat([c,z], dim=1)
        return self.main(c)
