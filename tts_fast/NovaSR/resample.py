import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# --- MATH UTILS ---
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: beta = 0.1102 * (A - 8.7)
    elif A >= 21.: beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

# --- FUSED KERNEL (Restored to 12-tap Alignment) ---
@torch.jit.script
def _polyphase_upsample_fused(x: Tensor, weight: Tensor, ratio: int):
    # Original padding for 12-tap center alignment
    x = F.pad(x, (2, 3))
    out = F.conv1d(x, weight, groups=x.shape[1], stride=1)

    B, C_out, L = out.shape
    C = x.shape[1]
    out = out.view(B, C, ratio, L).transpose(2, 3).reshape(B, C, -1)
    # Original slice for 12-tap alignment
    return out[..., 2:-2]

# --- MODULES ---

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = kernel_size

        # Buffer for loading static 12-tap weights
        self.register_buffer("filter", torch.zeros(1, 1, 12))

        # Fast buffer: kernel_size // ratio = 6 taps per phase
        self.register_buffer("f_fast", torch.zeros(channels * ratio, 1, 6), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            w = self.filter * float(self.ratio)
            w = w.view(self.kernel_size) # Should be 12

            # Polyphase decomposition (Length 12 -> two phases of 6)
            p0, p1 = w[0::2], w[1::2]

            fast_w = torch.stack([p0, p1], dim=0).unsqueeze(0).expand(self.channels, -1, -1)
            fast_w = fast_w.reshape(self.channels * self.ratio, 1, 6)
            self.f_fast.copy_(fast_w)
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared and not self.training: self.prepare()
        return _polyphase_upsample_fused(x, self.f_fast[:x.shape[1]*self.ratio], self.ratio)

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = kernel_size

        # Buffer for loading static 12-tap weights
        self.register_buffer("filter", torch.zeros(1, 1, 12))

        # Optimized buffer for execution
        self.register_buffer("f_opt", torch.zeros(channels, 1, 12), persistent=False)
        self._prepared = False

    def prepare(self):
        with torch.no_grad():
            self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True

    def forward(self, x: Tensor):
        if not self._prepared and not self.training: self.prepare()
        C = x.shape[1]
        # Original padding 5 for 12-tap kernel
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=5, groups=C)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, kernel_size, channels)
    def forward(self, x):
        return self.lowpass(x)
