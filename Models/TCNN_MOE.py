"""
Dynamic-weight TCN Mixture-of-Experts predictor (the "MoETCNN" model used for
training/evaluation in network_training.py). A PAE-phase-driven gating network
(as in Models/MANN.py) produces per-sample expert weights that blend TCN
convolution kernels directly, so the gate reshapes the TCN's effective weights
per sample rather than only blending expert outputs. See
Models/TCNN_MOE_Additional.py for earlier output-blended / shared-body MoE
variants and gating regularizers explored during development.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def Normalize(X, N):
    """Normalize `X` using a (mean, std) pair `N`."""
    mean = N[0]
    std = N[1]
    return (X - mean) / std

# Chomp1d ensures the output length is the same as the input length after convolution.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class GatingNet(nn.Module):
    """ELU MLP gate (as in your MANN), outputs softmax weights over experts."""
    def __init__(self, gating_input: int, gating_hidden: int, num_experts: int, dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.g1 = nn.Linear(gating_input, gating_hidden)
        self.g2 = nn.Linear(gating_hidden, gating_hidden)
        self.g3 = nn.Linear(gating_hidden, num_experts)
        self.dropout = dropout
        self.temperature = temperature
        fab_mean = [-3.2564923e-02, -2.3406364e-02, 1.3240871e+00, 4.1014603e+01,
                    -2.4297564e-01, -1.6370684e-02, -2.4113677e-02, 1.7107403e+00,
                     5.4285000e+01,  4.5630690e-02,  4.6314280e-02,  1.9869128e-02,
                     4.6051321e+00,  1.6846336e+01, -4.2095739e-01, -4.5052882e-02,
                     1.0535752e-01,  3.1186976e+00,  2.3081659e+01, -2.0144255e-01,
                     3.3257127e-02,  2.9932617e-03,  1.4969879e+00,  3.5440731e+01,
                    -1.5683125e-01, -4.0017970e-02, -4.4576611e-02,  2.8906257e+00,
                     2.4773144e+01,  1.8246192e+00]

        
     
        fab_std = [ 0.7077086,   0.70508003,  0.3104338,  16.11017   ,  6.2945657 ,  0.66618043,
                    0.7445238,   0.34715655, 15.923401 ,   6.063388  ,  0.705116  ,  0.70655215,
                    1.9695925,  10.840973  ,  3.1709101,   0.6837597 ,  0.72021097,  2.0887349,
                   14.667681 ,   3.639706  ,  0.7222788,   0.69070977,  0.38787553, 16.480913,
                    5.2354403,   0.7105775 ,  0.7003866,   0.99095803, 14.494578  ,  7.0202174 ]
        
        # Register as buffers so they move with .to(device) / .cuda()
        self.register_buffer("fab_mean", torch.tensor(fab_mean, dtype=torch.float32))
        self.register_buffer("fab_std", torch.tensor(fab_std, dtype=torch.float32))

    def forward(self, g):
        # g: (B, G)
        # Normalize only features 21:50 (Python slicing is end-exclusive)
        # g[:, 20:50] = (g[:, 20:50] - self.fab_mean) / (self.fab_std + 1e-6)
        
        g = F.dropout(g, self.dropout, training=self.training)
        g = F.elu(self.g1(g))
        g = F.dropout(g, self.dropout, training=self.training)
        g = F.elu(self.g2(g))
        g = F.dropout(g, self.dropout, training=self.training)
        
        temperature = torch.clamp(torch.tensor(self.temperature, device=g.device), min=1e-6)
        
        logits = self.g3(g) / temperature
        w = F.softmax(logits, dim=1)  # (B, E)
        return w
        
# ------------------------------
# Dynamic-Weight TCN (expert-weighted conv kernels)
# ------------------------------

# This represents batch wise blending of weights.
# class ExpertConv1d(nn.Module):
#     """
#     Conv1d whose weights are a *mixture of E expert kernels* blended by per-sample
#     gating weights w (B, E). This directly implements "gating dynamically changes the
#     TCN weights" rather than only blending outputs.

#     Shapes
#       - experts: E
#       - W: (E, C_out, C_in, K)
#       - b: (E, C_out)
#       - x: (B, C_in, T)
#       - w: (B, E)
#     """
#     def __init__(self, experts: int, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
#         super().__init__()
#         self.experts = experts
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation

#         # Expert kernels & biases
#         self.W = nn.Parameter(torch.empty(experts, out_ch, in_ch, kernel_size))
#         self.b = nn.Parameter(torch.zeros(experts, out_ch))
#         bound = math.sqrt(6.0 / (in_ch * kernel_size + out_ch))
#         nn.init.uniform_(self.W, -bound, bound)

#     def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         B, C_in, T = x.shape
#         E = self.experts
#         assert w.shape == (B, E), f"Expected w=(B,E) got {tuple(w.shape)}"

#         # Blend expert kernels per-sample: Wb[b] = sum_e w[b,e] * W[e]
#         # Vectorize blending, then apply conv per-sample
#         # blended_W: (B, C_out, C_in, K), blended_b: (B, C_out)
#         blended_W = torch.einsum('be,ecik->bcik', w, self.W)
#         blended_b = torch.einsum('be,eo->bo', w, self.b)

#         outs = []
#         for b in range(B):
#             out_b = F.conv1d(
#                 x[b:b+1], blended_W[b], blended_b[b],
#                 stride=self.stride, padding=self.padding, dilation=self.dilation
#             )
#             outs.append(out_b)
#         return torch.cat(outs, dim=0)  # (B, C_out, T_out)


## This is a globalized weight blending
# class ExpertConv1d(nn.Module):
#     """
#     Conv1d whose weights are a *mixture of E expert kernels* blended by per-sample
#     gating weights w (B, E). This directly implements "gating dynamically changes the
#     TCN weights" rather than only blending outputs.

#     Shapes
#       - experts: E
#       - W: (E, C_out, C_in, K)
#       - b: (E, C_out)
#       - x: (B, C_in, T)
#       - w: (B, E)
#     """
#     def __init__(self, experts: int, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
#         super().__init__()
#         self.experts = experts
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation

#         # Expert kernels & biases
#         self.W = nn.Parameter(torch.empty(experts, out_ch, in_ch, kernel_size))
#         self.b = nn.Parameter(torch.zeros(experts, out_ch))
#         bound = math.sqrt(6.0 / (in_ch * kernel_size + out_ch))
#         nn.init.uniform_(self.W, -bound, bound)
        
        
    
#     ## This is a globalized weight blending
#     def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, C_in, T)
#         w: (E,)  or (1,E)  or (B,E)
#            - If (E,) or (1,E): use as-is (global blend for whole batch)
#            - If (B,E): reduce to a single global w (e.g., mean over batch)
#         """
#         B, C_in, T = x.shape
#         E = self.experts
    
#         # ---- normalize weights ----
#         if w.dim() == 1 and w.numel() == E:          # (E,)
#             w_global = torch.softmax(w, dim=0)       # (E,)
#         elif w.dim() == 2 and w.shape == (1, E):     # (1,E)
#             w_global = torch.softmax(w.squeeze(0), dim=0)  # (E,)
#         elif w.dim() == 2 and w.shape[1] == E:       # (B,E) -> reduce to one w
#             # choose a reduction you like; mean is common:
#             w_global = torch.softmax(w.mean(dim=0), dim=0) # (E,)
#             # alternatives: median, sum then softmax, or a learned reducer
#         else:
#             raise ValueError(f"w must be (E,), (1,E), or (B,E); got {tuple(w.shape)}")
    
#         # ---- blend once for the whole batch ----
#         # blended_W: (C_out, C_in, K), blended_b: (C_out,)
#         blended_W = torch.einsum('e,ecik->cik', w_global, self.W)
#         blended_b = torch.einsum('e,eo->o',     w_global, self.b)
    
#         # ---- single conv for the whole batch ----
#         y = F.conv1d(
#             x, blended_W, blended_b,
#             stride=self.stride, padding=self.padding, dilation=self.dilation
#         )
#         return y
    
class ExpertConv1d(nn.Module):    
    ## Output blending across experts
    """
    Run E expert convs in parallel and blend their *outputs*:
      y = sum_e w_e * Conv_e(x)

    Supports:
      - w: (E,)  -> same mixture for whole batch
      - w: (B,E) -> per-sample mixture
    """
    def __init__(self, experts: int, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 use_weight_norm: bool = True):
        super().__init__()
        self.experts = experts
        convs = []
        for _ in range(experts):
            c = nn.Conv1d(in_ch, out_ch, kernel_size,
                          stride=stride, padding=padding, dilation=dilation, bias=True)
            if use_weight_norm:
                c = weight_norm(c)
            convs.append(c)
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        w: (E,) or (B,E)
        returns: (B, C_out, T_out)
        """
        B = x.size(0)

        # Collect per-expert outputs: list of (B, C_out, T) -> stack -> (B, E, C_out, T)
        ys = [conv(x).unsqueeze(1) for conv in self.convs]
        Y = torch.cat(ys, dim=1)  # (B, E, C_out, T)

        # Normalize / broadcast weights
        if w.dim() == 1:                 # (E,)
            w = F.softmax(w, dim=0).unsqueeze(0).expand(B, -1)  # (B,E)
        elif w.dim() == 2:               # (B,E)
            w = F.softmax(w, dim=1)
        else:
            raise ValueError(f"w must be (E,) or (B,E); got {tuple(w.shape)}")

        # Blend along expert axis: (B,E) ⊗ (B,E,C_out,T) -> (B,C_out,T)
        out = torch.einsum('be,bect->bct', w, Y)
        return out
        
    
    
class ExpertTemporalBlock(nn.Module):
    def __init__(self, experts: int, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ExpertConv1d(experts, n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = ExpertConv1d(experts, n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # expert-mixed 1x1 for residual when channels change
        self.downsample = ExpertConv1d(experts, n_inputs, n_outputs, kernel_size=1, stride=1, padding=0, dilation=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T), w: (B, E)
        out = self.conv1(x, w)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out, w)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x, w)
        return self.relu(out + res)


class ExpertTemporalConvNet(nn.Module):
    """
    Stack of ExpertTemporalBlock with increasing dilation.
    Every conv kernel is dynamically blended by the gating weights per sample.
    """
    def __init__(self, experts: int, num_inputs: int, num_channels, kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                ExpertTemporalBlock(
                    experts, in_ch, out_ch, kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x, w)
        return x


class MANN_TCN_DynamicWeights(nn.Module):
    """
    End-to-end: GatingNet -> per-sample expert weights -> Expert TCN (all convs mixed)
    This exactly matches: "gating network dynamically changes the weights of the TCN".
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_experts: int,
                 tcn_channels,                # e.g., [64,128,128,256]
                 gating_input: int,
                 gating_hidden: int = 128,
                 kernel_size: int = 2,
                 tcn_dropout: float = 0.2,
                 gating_dropout: float = 0.0,
                 temperature: float = 1.0):
        super().__init__()
        self.gate = GatingNet(gating_input, gating_hidden, num_experts, dropout=gating_dropout, temperature=temperature)
        self.tcn = ExpertTemporalConvNet(num_experts, input_size, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        self.head = nn.Linear(tcn_channels[-1], output_size)

    def forward(self, phaseinputs: torch.Tensor, seq_input: torch.Tensor):
        # phaseinputs: (B, G), seq_input: (B, C_in, T)
        w = self.gate(phaseinputs)           # (B, E)
        feats_t = self.tcn(seq_input, w)             # (B, C_last, T)
        feats = feats_t[:, :, -1]
        out = self.head(feats)
        # return {
        #     'pred': out,
        #     'weights': w,
        #     'logits': logits,
        #     'feats': feats,
        # }
        
        return out

class MANN_TCN_DynamicWeights_Forecast(nn.Module):
    """
    Same architecture as `MANN_TCN_DynamicWeights`, but the final linear head predicts
    all `horizon` future timesteps in one forward pass instead of a single step.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 num_experts: int,
                 tcn_channels,                # e.g., [64,128,128,256]
                 gating_input: int,
                 gating_hidden: int = 128,
                 kernel_size: int = 2,
                 tcn_dropout: float = 0.2,
                 gating_dropout: float = 0.0,
                 temperature: float = 1.0):
        super().__init__()
        self.gate = GatingNet(gating_input, gating_hidden, num_experts, dropout=gating_dropout, temperature=temperature)
        self.tcn = ExpertTemporalConvNet(num_experts, input_size, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        self.horizon = horizon
        self.output_size = output_size
        
        self.linear = nn.Linear(tcn_channels[-1], output_size * horizon)

    def forward(self, phaseinputs: torch.Tensor, seq_input: torch.Tensor):
        # phaseinputs: (B, G), seq_input: (B, C_in, T)
        w = self.gate(phaseinputs)           # (B, E)
        feats_t = self.tcn(seq_input, w)             # (B, C_last, T)
        last = feats_t[:, :, -1]
        out = self.linear(last)         # [B, H*C_out]
        out = out.view(seq_input.size(0), self.horizon, self.output_size) \
                 .transpose(1, 2) \
                 .contiguous() 
        # return {
        #     'pred': out,
        #     'weights': w,
        #     'logits': logits,
        #     'feats': feats,
        # }
        
        return out
    

class MANN_TCN_DynamicWeights_Forecast_k(nn.Module):
    """
    TCN-MoE forecaster with temporal conditioning.
    - predict_k(..., k): one-step-at-k (1..H) using FiLM on last features.
    - predict_all(...): vectorized prediction for all steps 1..H.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 num_experts: int,
                 tcn_channels,                # e.g., [64,128,128,256]
                 gating_input: int,
                 gating_hidden: int = 128,
                 kernel_size: int = 2,
                 tcn_dropout: float = 0.2,
                 gating_dropout: float = 0.0,
                 temperature: float = 1.0,
                 t_dim: int = 16):            # <-- temporal embedding size
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.last_dim = tcn_channels[-1]

        # Gate + TCN unchanged
        self.gate = GatingNet(gating_input, gating_hidden, num_experts,
                              dropout=gating_dropout, temperature=temperature)
        self.tcn = ExpertTemporalConvNet(num_experts, input_size, tcn_channels,
                                         kernel_size=kernel_size, dropout=tcn_dropout)

        # Learned embedding for horizons [1..H]
        self.time_emb = nn.Embedding(horizon + 1, t_dim)

        # Simple linear head
        self.head = nn.Linear(self.last_dim + t_dim, output_size)

    def _last_feats(self, phaseinputs: torch.Tensor, seq_input: torch.Tensor):
        """
        Helper: compute last TCN features once.
        returns: feats_last: [B, C_last]
        """
        w = self.gate(phaseinputs)                 # (B, E)
        feats_t = self.tcn(seq_input, w)           # (B, C_last, T)
        feats_last = feats_t[:, :, -1]             # (B, C_last)
        return feats_last

    def forward(self, phaseinputs, seq_input, k: torch.LongTensor):
        """
        phaseinputs: [B, G]
        seq_input:   [B, C_in, T]
        k:           [B] integer tensor in [1..H] (different horizon per sample)
        returns:     [B, D_out]  (prediction at horizon k only)
        """
        
        if isinstance(k, torch.Tensor):
            k = int(k.item())  # make sure it's an int
        
        B = seq_input.size(0)
        feats = self._last_feats(phaseinputs, seq_input)    # [B, C_last]
        
        
        # Build embeddings only up to K
        device = feats.device
        ks = torch.arange(1, k+1, device=device)         # [K]
        tmat = self.time_emb(ks)                           # [K,t_dim]
    
        feats_exp = feats.unsqueeze(1).expand(B, k, -1)    # [B,K,C_last]
        tmat_exp  = tmat.unsqueeze(0).expand(B, -1, -1)    # [B,K,t_dim]
        z = torch.cat([feats_exp, tmat_exp], dim=-1)       # [B,K,C_last+t_dim]
    
        y = self.head(z)                                   # [B,K,D_out]
        return y.transpose(1, 2).contiguous()              # [B,D_out,K]