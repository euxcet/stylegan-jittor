import jittor as jt
from jittor import nn
from jittor import init
import numpy as np
import random
import math


if jt.has_cuda:
    jt.flags.use_cuda = 1

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()
        self.weight = jt.zeros((1, channel, 1, 1))

    def execute(self, image, noise):
        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.constant = jt.zeros((1, channel, 1, 1))

    def execute(self, x):
        batch_size = x.shape[0]
        out = self.constant.repeat(batch_size, 1, 1, 1)
        return out


'''
class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super(FusedUpsample, self).__init__()
        weight = jt.ops.random((in_channel, out_channel, kernel_size, kernel_size))
        bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

'''

class Blur(nn.Module):
    def __init__(self, channel):
        super(Blur, self).__init__()
        self.weight = jt.float32([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        self.weight = self.weight.view(1, 1, 3, 3)
        self.weight = self.weight / self.weight.sum()
        self.weight.requires_grad = False

    def execute(self, input):
        return nn.conv2d(input, self.weight, padding=1, groups=input.shape[1])
        
    

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.linear = nn.Linear(style_dim, in_channel * 2)

        self.linear.bias.data[:in_channel] = 1
        self.linear.bias.data[in_channel:] = 0

    def execute(self, input, style):
        style = self.linear(style).unsqueeze(2).unsqueeze(3) 
        gamma, beta = style.chunk(2, 1)
        
        out = self.norm(input)
        out = gamma * out + beta

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, downsample=False, fused=False):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding),
            nn.LeakyReLU(0.2)
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
                    nn.Pool(2),
                    nn.LeakyReLU(0.2)
                )
            else:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
                    nn.Pool(2),
                    nn.LeakyReLU(0.2)
                )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding),
                nn.LeakyReLU(0.2)
            )

    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(input)
        return out

 

class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, style_dim=512, initial=False, upsample=False, fused=False):
        super(StyledConvBlock, self).__init__()
        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        #FusedUpsample(in_channel, out_channel, kernel_size, padding=padding)
                        Blur(out_channel)
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)#  todo: equal
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)#  todo: equal
                        Blur(out_channel)
                    )
        self.noise1 = NoiseInjection(out_channel)
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = NoiseInjection(out_channel)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def execute(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)
        return out

class Generator(nn.Module):
    def __init__(self, dim, fused=True):
        super(Generator, self).__init__()
        self.progression = nn.ModuleList([
            StyledConvBlock(512, 512, 3, 1, initial=True),
            StyledConvBlock(512, 512, 3, 1, upsample=True),
            StyledConvBlock(512, 512, 3, 1, upsample=True),
            StyledConvBlock(512, 512, 3, 1, upsample=True),
            StyledConvBlock(512, 256, 3, 1, upsample=True),
            StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),
            StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),
            StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),
            StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused)
        ])

        self.to_rgb = nn.ModuleList([
            nn.Conv2d(512, 3, 1),
            nn.Conv2d(512, 3, 1),
            nn.Conv2d(512, 3, 1),
            nn.Conv2d(512, 3, 1),
            nn.Conv2d(256, 3, 1),
            nn.Conv2d(128, 3, 1),
            nn.Conv2d(64, 3, 1),
            nn.Conv2d(32, 3, 1),
            nn.Conv2d(16, 3, 1),
        ])

    def execute(self, style, noise, step=0, alpha=-1, mixing_range=(-1,-1)):
        out = noise[0]
        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))
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
                out_prev = out

            out = conv(out, style_step, noise[i])
            
            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = nn.interpolate(skip_rgb, scale_factor=2, mode='nearest') # F
                    out = (1 - alpha) * skip_rgb + alpha * out
                break
        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def execute(self, input):
        return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)

class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super(StyledGenerator, self).__init__()
        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(nn.Linear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        
        self.style = nn.Sequential(*layers)

    def execute(self, input, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1,-1)):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]
        
        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(jt.ops.random([batch, 1, size, size]))
        
        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            
            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdims=True)
        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super(Discriminator, self).__init__()
        
        self.progression = nn.ModuleList([
            ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),
            ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),
            ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),
            ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),
            ConvBlock(256, 512, 3, 1, downsample=True),
            ConvBlock(512, 512, 3, 1, downsample=True),
            ConvBlock(512, 512, 3, 1, downsample=True),
            ConvBlock(512, 512, 3, 1, downsample=True),
            ConvBlock(513, 512, 3, 1, 4, 0), # why 513?
        ])

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(nn.Conv2d(3, out_channel, 1), nn.LeakyReLU(0.2))
            else:
                return nn.Conv2d(3, out_channel, 1)
        
        self.from_rgb = nn.ModuleList([
            make_from_rgb(16),
            make_from_rgb(32),
            make_from_rgb(64),
            make_from_rgb(128),
            make_from_rgb(256),
            make_from_rgb(512),
            make_from_rgb(512),
            make_from_rgb(512),
            make_from_rgb(512),
        ])

        self.n_layer = len(self.progression)
        self.linear = nn.Linear(512, 1)

    def execute(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1
            if i == step:
                out = self.from_rgb[index](input)
            if i == 0:
                out_std = jt.sqrt(out.var + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = jt.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = nn.pool(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out
        
