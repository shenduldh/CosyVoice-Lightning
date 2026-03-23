import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torchaudio
from einops import rearrange
from .commons import init_weights, get_padding
from .activations import SnakeBeta
from .resample import UpSample1d, DownSample1d

LRELU_SLOPE = 0.1 

class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x

class AMPBlock0(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock0, self).__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers


        self.activations = nn.ModuleList([
            Activation1d(
                activation=SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
        ])

    def forward(self, x):
    
        # Since we pruned, zip will only iterate once
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, 
                                 self.activations[::2], self.activations[1::2]):
        
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)

            # Residual connection
            x = xt + x[:, :, :xt.shape[2]]
        
        return x
        
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_initial_channel, gin_channels=0):
        super(Generator, self).__init__()

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = AMPBlock0

        self.resblocks = nn.ModuleList()
        for i in range(1):
            ch = upsample_initial_channel//(2**(i))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        x = F.interpolate(x, int(x.shape[-1] * 3), mode='linear')

        xs = self.resblocks[0](x)

        x = self.conv_post(xs)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_initial_channel,
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.segment_size = segment_size

        self.dec = Generator(1, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_initial_channel)


    def forward(self, x):
        y = self.dec(x)
        return y


    @torch.no_grad()
    def infer(self, x, max_len=None):
        o = self.dec(x[:,:,:max_len])
        return o
