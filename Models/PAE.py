"""
Periodic Autoencoder (PAE): a 1D-convolutional autoencoder that learns per-channel
phase, frequency, amplitude, and offset parameters describing the periodicity of
each IMU signal channel, and reconstructs the input signal from a sinusoidal latent
representation built from those parameters. These phase parameters are used
downstream to gate the MANN / MoE-TCN motion predictors.

This model architecture is adapted from the Periodic Autoencoder introduced in:
Sebastian Starke, Ian Mason, and Taku Komura, "DeepPhase: Periodic Autoencoders for
Learning Motion Phase Manifolds", ACM Transactions on Graphics (SIGGRAPH 2022),
41(4), Article 136. https://dl.acm.org/doi/10.1145/3528223.3530178

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

class LN_v2(nn.Module):
    """Feature-wise layer normalization with learnable scale/shift, applied along the time axis."""
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y
    
class Model(nn.Module):
    """Periodic Autoencoder: Conv1d encoder -> per-channel phase/frequency/amplitude/offset
    extraction (via FFT + learned phase head) -> sinusoidal reconstruction -> Conv1d decoder."""
    def __init__(self, input_channels, embedding_channels, intermediate_channels, time_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.window = window
        self.intermediate_channels = intermediate_channels

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False) #Remove DC frequency

        # intermediate_channels = int(input_channels/3)
        
        self.conv1 = nn.Conv1d(input_channels, self.intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(time_range)
        self.conv2 = nn.Conv1d(self.intermediate_channels, embedding_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        self.deconv1 = nn.Conv1d(embedding_channels, self.intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(time_range)
        self.deconv2 = nn.Conv1d(self.intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    def FFT(self, function, dim):
        """Return per-channel dominant frequency, amplitude, and DC offset of `function` over `dim`."""
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        # Signal Embedding

        # y = y.reshape(y.shape[0], self.input_channels, self.time_range)


        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        # y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        # return y, latent, signal, params
        return params
    




