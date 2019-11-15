import torch

import torch.optim as optim
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt

import random


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module





class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)



class Blur(nn.Module):
    def __init__(self):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(input, self.weight.repeat(input.shape[1], 1, 1, 1), padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


def cos_similarity(weight):
    weight = weight / weight.norm(dim=-1).unsqueeze(-1)
    cos_distance = torch.mm(weight, weight.transpose(1,0))
    return cos_distance.pow(2).mean()

class EqualConvOrth2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-5, betas=(0.5, 0.99))

    def forward(self, input):
        if self.training:
            self.orthogonal_update()
        return self.conv(input)

    def orthogonal_update(self):
        self.zero_grad()
        weight = self.conv.weight_orig
        a,b,c,d = weight.size()
        loss = cos_similarity(self.conv.weight_orig.view(a*b, c*d))
        loss.backward()
        self.opt_orth.step()

class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding,
                 kernel_size2=None, padding2=None,
                 pixel_norm=True, spectral_norm=False, orth=False):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        # we modify the output on Discriminator
        # to have orthogonal groupped conv layers
        if orth:
            conv_module = EqualConvOrth2d
        else:
            conv_module = EqualConv2d

        self.conv = nn.Sequential(conv_module(in_channel, out_channel,
                                            kernel1, padding=pad1),
                                nn.LeakyReLU(0.2),
                                conv_module(out_channel, out_channel,
                                            kernel2, padding=pad2),
                                nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, style_dim=512, initial=False):
        super().__init__()

        self.initial = initial

        if initial:
            ### we modify the stylegan input to function as OOGAN
            self.conv1 = EqualConvTranspose2d(in_channel, out_channel, kernel_size, stride=1, padding=padding)
        else:
            self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        if len(input.shape)==2:
            input = input.unsqueeze(-1).unsqueeze(-1)
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.adain1(out, style)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.adain2(out, style)
        out = self.lrelu2(out)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, control_dim):
        super().__init__()

        self.progression = nn.ModuleList([StyledConvBlock(control_dim, 512, 4, 0, initial=True),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 256, 3, 1),
                                          StyledConvBlock(256, 128, 3, 1)])

        self.to_rgb = nn.ModuleList([EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(256, 3, 1),
                                     EqualConv2d(128, 3, 1)])

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0][0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                # upsample = self.blur(upsample)
                out = conv(upsample, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i][1])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8, control_dim=32):
        super().__init__()

        self.generator = Generator(code_dim, control_dim)
        self.control_dim = control_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
        
        self.controlled_noise = nn.ModuleList([
                EqualLinear(control_dim, 4**2),
                EqualLinear(control_dim, 8**2)])
        
    def forward(self, input, vector_c, noise=None, fix_noise=False, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1)):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            
            for i in range(0, step + 1):
                ### we modify the input module of StyleGAN, letting
                ### 1. the initial noise is now generated by the control vector c
                ### 2. the injected noise is now also generated by control vector c
                ### 3. we leave the z as it is, to keep original styleGAN's features
                size = 4 * 2 ** i
                if size <= 8:
                    noise_from_c = self.controlled_noise[i](vector_c.view(batch, -1)).view(batch, 1, size, size)
                    if i==0:
                        noise_from_c = (vector_c, noise_from_c)
                    noise.append(noise_from_c)
                else:
                    if fix_noise:
                        noise.append(torch.zeros(batch, 1, size, size, device=input[0].device))
                    else:      
                        noise.append(0.1*torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, control_dim=32):
        super().__init__()

        self.control_dim = control_dim

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1),
                                          ConvBlock(256, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(513, 512, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, 128, 1),
                                       EqualConv2d(3, 256, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1)])
        
        ### we modify the output module of StyleGAN, letting
        ### 1. predict c' 
        ### the conv module is orthogonal regulairzed and is groupped convolution
        self.to_c = nn.Sequential(
            ConvBlock(512, 512, 3, 1, orth=False),
            EqualConvOrth2d(512, control_dim*2, 4, 1, 0, groups=control_dim*2))
        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                pred_c_params = self.to_c(out)

                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out, pred_c_params.view(-1, self.control_dim, 2)