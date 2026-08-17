from typing import Dict, List
import numpy as np
import math
from scipy.signal import get_window
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.distributions.uniform import Uniform
import sys

sys.path.insert(0, "/data1/work/tts/models/cosyvoice/cosyvoice_dev-stream/tts_fast")

from cosyvoice.utils.common import get_padding
from cosyvoice.utils.common import init_weights


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class CausalConv1dUpsample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CausalConv1dUpsample, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert dilation == 1
        self.causal_padding = kernel_size - 1
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)):
        x = self.upsample(x)
        # input_timestep = x.shape[2]
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            # assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dUpsample, self).forward(x)
        # assert input_timestep == x.shape[2]
        return x


class CausalConv1dDownSample(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CausalConv1dDownSample, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            # assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        x = super(CausalConv1dDownSample, self).forward(x)
        return x


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        causal_type: str = "left",
        device=None,
        dtype=None,
    ) -> None:
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert stride == 1
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2
        assert causal_type in ["left", "right"]
        self.causal_type = causal_type

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)):
        # input_timestep = x.shape[2]
        if cache.size(2) == 0:
            cache = torch.zeros(x.shape[0], x.shape[1], self.causal_padding, device=x.device, dtype=x.dtype)
        # assert cache.size(2) == self.causal_padding
        if self.causal_type == "left":
            x = torch.concat([cache, x], dim=2)
        else:
            x = torch.concat([x, cache], dim=2)
        x = super(CausalConv1d, self).forward(x)
        # assert x.shape[2] == input_timestep
        return x


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        causal: bool = False,
    ):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=dilation, padding=get_padding(kernel_size, dilation))
                    if causal is False
                    else CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation, causal_type="left")
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                    if causal is False
                    else CausalConv1d(channels, channels, kernel_size, 1, dilation=1, causal_type="left")
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for idx in range(len(self.convs1)):
            remove_parametrizations(self.convs1[idx], "weight")
            remove_parametrizations(self.convs2[idx], "weight")


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, dim=1, length)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.transpose(1, 2)
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)), device=f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate

        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0

        # generate sine waveforms
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves.transpose(1, 2), uv.transpose(1, 2), noise


class SineGen2(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self, samp_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0, flag_for_pulse=False, causal=False
    ):
        super(SineGen2, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.causal = causal
        if causal is True:
            rand_ini = torch.rand(1, 9)
            rand_ini[:, 0] = 0
            self.register_buffer("rand_ini", rand_ini)
            self.register_buffer("sine_waves", torch.rand(1, 300 * 24000, 9))
        self.register_buffer("f0_mul_weight", torch.FloatTensor([[range(1, self.harmonic_num + 2)]]))

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        if self.training is False and self.causal is True:
            rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini
        else:
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # rad_values = rad_values.transpose(1, 2).contiguous()
            # rad_values = torch.nn.functional.interpolate(rad_values, scale_factor=1 / self.upsample_scale, mode="linear")
            # rad_values = rad_values.transpose(1, 2).contiguous()
            # phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            # phase = phase.transpose(1, 2).contiguous() * self.upsample_scale
            # phase = torch.nn.functional.interpolate(phase, scale_factor=self.upsample_scale, mode="nearest" if self.causal is True else "linear")
            # phase = phase.transpose(1, 2).contiguous()
            # sines = torch.sin(phase)

            # 下采样
            rad_values_down = rad_values[:, :: self.upsample_scale, :]
            # 沿时间维累加相位
            phase_cycles = (torch.cumsum(rad_values_down, dim=1) * self.upsample_scale) % 1.0
            phase = phase_cycles * 2 * np.pi
            # 上采样
            btz, seq_len, dim = phase.shape
            phase = phase.unsqueeze(2).expand(btz, seq_len, self.upsample_scale, dim)
            phase = phase.reshape(btz, seq_len * self.upsample_scale, dim)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        # fundamental component
        fn = torch.multiply(f0, self.f0_mul_weight)

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        if self.training is False and self.causal is True:
            noise = noise_amp * self.sine_waves[:, : sine_waves.shape[1]]
        else:
            noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0, sinegen_type="1", causal=False
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        if sinegen_type == "1":
            self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        else:
            self.l_sin_gen = SineGen2(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod, causal=causal)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        self.causal = causal
        if causal is True:
            self.register_buffer("uv", torch.rand(1, 300 * 24000, 1))

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        if self.training is False and self.causal is True:
            noise = self.uv[:, : uv.shape[1]] * self.sine_amp / 3
        else:
            noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(CausalConv1d(in_channels, cond_channels, kernel_size=4, causal_type="right")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
            weight_norm(CausalConv1d(cond_channels, cond_channels, kernel_size=3, causal_type="left")),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)
        self.causal_padding = int(self.condnet[0].causal_padding)

    def remove_parametrizations(self):
        remove_parametrizations(self.condnet[0], "weight")
        remove_parametrizations(self.condnet[2], "weight")
        remove_parametrizations(self.condnet[4], "weight")
        remove_parametrizations(self.condnet[6], "weight")
        remove_parametrizations(self.condnet[8], "weight")

    def forward(self, x: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        if finalize is True:
            x = self.condnet[0](x)
        else:
            x = self.condnet[0](x[:, :, : -self.causal_padding], x[:, :, -self.causal_padding :])
        for i in range(1, len(self.condnet)):
            x = self.condnet[i](x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class ConvSTFT(nn.Module):
    def __init__(self, n_fft=16, hop_length=4, win_length=16, init_dtype="float64"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_amount = self.n_fft // 2

        np_dtype = getattr(np, init_dtype, np.float32)
        torch_dtype = getattr(torch, init_dtype, torch.float32)

        window = torch.from_numpy(get_window("hann", win_length, fftbins=True).astype(np_dtype))

        k = torch.arange(self.n_fft // 2 + 1, dtype=torch_dtype).view(-1, 1, 1)
        n = torch.arange(self.n_fft, dtype=torch_dtype).view(1, 1, -1)
        idx = 2 * math.pi * k * n / self.n_fft

        real_w = window.view(1, 1, -1) * torch.cos(idx)
        imag_w = window.view(1, 1, -1) * -torch.sin(idx)
        W_fwd = torch.cat([real_w, imag_w], dim=0)

        # 频域到时域需要除以 N
        # 考虑到共轭对称性，1~7 的频率要乘以 2，0 和 8 不乘
        scale = torch.ones(self.n_fft // 2 + 1, 1, 1, dtype=torch_dtype) * (2.0 / self.n_fft)
        scale[0] = 1.0 / self.n_fft
        scale[-1] = 1.0 / self.n_fft

        real_inv = real_w * scale
        imag_inv = imag_w * scale
        W_inv = torch.cat([real_inv, imag_inv], dim=0)

        # iSTFT 重建后需要除以窗口平方的叠加包络以消除幅度调制
        W_env = (window**2).view(1, 1, 16)

        self.register_buffer("W_fwd", W_fwd.to(torch.float32))
        self.register_buffer("W_inv", W_inv.to(torch.float32))
        self.register_buffer("W_env", W_env.to(torch.float32))

    def stft(self, x):
        # PyTorch 的 STFT 默认 center=True 使用 reflect 模式填充
        x_pad = F.pad(x.unsqueeze(1), (self.pad_amount, self.pad_amount), mode="reflect")
        spec = F.conv1d(x_pad, self.W_fwd, stride=self.hop_length)
        real = spec[:, : (self.n_fft // 2 + 1), :]
        imag = spec[:, (self.n_fft // 2 + 1) :, :]
        return real, imag

    def istft(self, magnitude, phase):
        magnitude = torch.clamp(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        X = torch.cat([real, imag], dim=1)

        # iSTFT 转置卷积
        signal_recon = F.conv_transpose1d(X, self.W_inv, stride=self.hop_length)

        # 动态包络抵消
        # ones = torch.ones(1, 1, X.size(-1), dtype=X.dtype, device=X.device)
        ones = torch.ones_like(X[:, :1, :])
        envelope = F.conv_transpose1d(ones, self.W_env, stride=self.hop_length)
        signal_recon = signal_recon / torch.clamp(envelope, min=1e-11)

        # 去除首尾的 Padding 以还原等长时序音频
        out = signal_recon.squeeze(1)
        out = out[:, self.pad_amount : -self.pad_amount]

        return out


class CausalHiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: List[int] = [8, 8],
        upsample_kernel_sizes: List[int] = [16, 16],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        conv_pre_look_right: int = 4,
        f0_predictor: CausalConvRNNF0Predictor = None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=math.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
            sinegen_type="1" if self.sampling_rate == 22050 else "2",
            causal=True,
        )
        self.upsample_rates = upsample_rates
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(CausalConv1d(in_channels, base_channels, conv_pre_look_right + 1, 1, causal_type="right"))

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(CausalConv1dUpsample(base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u)))

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(CausalConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1, causal_type="left"))
            else:
                self.source_downs.append(CausalConv1dDownSample(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u))
            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d, causal=True))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d, causal=True))

        self.conv_post = weight_norm(CausalConv1d(ch, istft_params["n_fft"] + 2, 7, 1, causal_type="left"))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.conv_pre_look_right = conv_pre_look_right
        self.f0_predictor = f0_predictor

        self.causal_padding = int(self.f0_predictor.condnet[0].causal_padding)
        self.stft_crop_len = int(math.prod(self.upsample_rates) * self.conv_pre_look_right)
        self.audio_crop_len = int(math.prod(self.upsample_rates) * self.istft_params["hop_len"])

        self.n_fft = self.istft_params["n_fft"]
        self.hop_len = self.istft_params["hop_len"]
        self.win_len = self.istft_params["n_fft"]
        # self.register_buffer("stft_window", torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)))
        self.convstft_layer = ConvSTFT(self.n_fft, self.hop_len, self.win_len)
        self.high_precision = False

    def remove_parametrizations(self):
        for l in self.ups:
            remove_parametrizations(l, "weight")
        for l in self.resblocks:
            l.remove_parametrizations()
        for l in self.source_resblocks:
            l.remove_parametrizations()
        remove_parametrizations(self.conv_pre, "weight")
        remove_parametrizations(self.conv_post, "weight")
        self.f0_predictor.remove_parametrizations()

    def use_high_precision(self):
        torch.set_float32_matmul_precision("high")
        self.f0_predictor.to(torch.float64)
        self.high_precision = True

    def _stft(self, x):
        # spec = torch.stft(x, self.n_fft, self.hop_len, self.win_len, window=self.stft_window, return_complex=True)
        # return spec.real.contiguous(), spec.imag.contiguous()
        return self.convstft_layer.stft(x)

    def _istft(self, magnitude, phase):
        # magnitude = torch.clip(magnitude, max=1e2)
        # real = magnitude * torch.cos(phase)
        # img = magnitude * torch.sin(phase)
        # inversed = torch.istft(torch.complex(real, img), self.n_fft, self.hop_len, self.win_len, window=self.stft_window)
        # return inversed
        return self.convstft_layer.istft(magnitude, phase)

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0), finalize: bool = True) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize is True:
            x = self.conv_pre(x)
        else:
            x = self.conv_pre(x[:, :, : -self.conv_pre_look_right], x[:, :, -self.conv_pre_look_right :])
            s_stft_real = s_stft_real[:, :, : -self.stft_crop_len]
            s_stft_imag = s_stft_imag[:, :, : -self.stft_crop_len]
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1).to(dtype=x.dtype)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        if finalize is False:
            x = x[:, : -self.audio_crop_len]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, finalize: bool = True):
        # mel -> f0
        _speech_feat = speech_feat.to(torch.float64) if self.high_precision else speech_feat
        f0 = self.f0_predictor(_speech_feat, finalize=finalize).to(speech_feat)
        # f0 -> source
        source = self.f0_upsamp(f0[:, None]).transpose(1, 2).contiguous()  # [bs, n, t]
        source, _, _ = self.m_source(source)
        source = source.transpose(1, 2).contiguous()
        if finalize is True:
            generated_speech = self.decode(x=speech_feat, s=source, finalize=finalize)
        else:
            generated_speech = self.decode(x=speech_feat[:, :, : -self.causal_padding], s=source, finalize=finalize)
        return generated_speech


if __name__ == "__main__":
    import time

    model = CausalHiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
    )
    device = "cuda:3"
    model.to(device).eval()
    model.remove_parametrizations()
    model.use_high_precision()

    torch._dynamo.config.recompile_limit = 32
    for i in range(len(model.source_downs)):
        model.source_downs[i].forward = torch.compiler.disable(model.source_downs[i].forward)
    model.inference = torch.compile(model.inference, dynamic=True, mode="max-autotune-no-cudagraphs")

    dummy_input = torch.randn((1, 80, 200), device=device)
    model.inference(dummy_input, False)

    for i in range(10):
        s = time.perf_counter()
        model.inference(dummy_input, False)
        e = time.perf_counter()
        print(e - s)
