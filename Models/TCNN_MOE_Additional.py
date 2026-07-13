"""
Mode-Adaptive (Gated) Mixture-of-Experts for TCNs - additional/exploratory variants.
- Full MoE: multiple independent TCN experts, gate blends their predictions
- SharedBody: single TCN trunk, gate blends expert output heads (parameter-efficient)
- A second copy of the dynamic-weight (kernel-blending) TCN, kept for reference

The model actually used for training/evaluation ("MoETCNN" in network_training.py)
is `MANN_TCN_DynamicWeights[_Forecast]` in Models/TCNN_MOE.py; the variants here
were exploratory alternatives and are not wired into the main training pipeline.

Input conventions
- seq_input: (B, C_in, T)
- phaseinputs (gating inputs): (B, G)
- output: (B, O)

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

# ------------------------------
# Core TCN building blocks
# ------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Keep causality by chopping off the padding on the right
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ------------------------------
# Experts and Gating
# ------------------------------

class TCNExpert(nn.Module):
    """One TCN expert with its own head."""
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (B, C_in, T)
        y = self.tcn(x)              # (B, C_last, T)
        feats = y[:, :, -1]          # (B, C_last)
        out = self.head(feats)       # (B, O)
        return out, feats


class GatingNet(nn.Module):
    """ELU MLP gate (as in your MANN), outputs softmax weights over experts."""
    def __init__(self, gating_input: int, gating_hidden: int, num_experts: int, dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.g1 = nn.Linear(gating_input, gating_hidden)
        self.g2 = nn.Linear(gating_hidden, gating_hidden)
        self.g3 = nn.Linear(gating_hidden, num_experts)
        self.dropout = dropout
        self.temperature = temperature

    def forward(self, g):
        # g: (B, G)
        g = F.dropout(g, self.dropout, training=self.training)
        g = F.elu(self.g1(g))
        g = F.dropout(g, self.dropout, training=self.training)
        g = F.elu(self.g2(g))
        g = F.dropout(g, self.dropout, training=self.training)
        logits = self.g3(g) / max(self.temperature, 1e-6)
        w = F.softmax(logits, dim=1)  # (B, E)
        return w, logits


class ExpertLinearVectorized(nn.Module):
    """Vectorized output-blended expert layer (replaces looped ExpertLinear)."""
    def __init__(self, experts: int, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(experts, in_dim, out_dim))   # (E, C, O)
        self.b = nn.Parameter(torch.zeros(experts, 1, out_dim))        # (E, 1, O)
        # Xavier-like uniform
        bound = math.sqrt(6.0 / (in_dim + out_dim))
        nn.init.uniform_(self.W, -bound, bound)

    def forward(self, x, w):
        """
        x: (B, C)
        w: (B, E)
        returns: (B, O)
        """
        x_e = torch.einsum('bc,eco->ebo', x, self.W) + self.b   # (E, B, O)
        x_e = x_e.transpose(0, 1)                               # (B, E, O)
        out = torch.einsum('be,beo->bo', w, x_e)                # (B, O)
        return out


# ------------------------------
# Full MoE model: multiple TCN experts
# ------------------------------

class MANN_TCN_MoE(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_experts: int,
                 tcn_channels,                # e.g. [64, 128, 128, 256]
                 gating_input: int,
                 gating_hidden: int = 128,
                 kernel_size: int = 2,
                 expert_dropout: float = 0.2,
                 gating_dropout: float = 0.0,
                 temperature: float = 1.0):
        super().__init__()
        self.gate = GatingNet(gating_input, gating_hidden, num_experts, dropout=gating_dropout, temperature=temperature)
        self.experts = nn.ModuleList([
            TCNExpert(input_size, output_size, tcn_channels, kernel_size=kernel_size, dropout=expert_dropout)
            for _ in range(num_experts)
        ])

    def forward(self, phaseinputs, seq_input):
        """
        phaseinputs: (B, G)
        seq_input:   (B, C_in, T)
        """
        weights, logits = self.gate(phaseinputs)            # (B, E)
        outs = []
        feats = []
        for expert in self.experts:
            o, f = expert(seq_input)                        # (B, O), (B, C_last)
            outs.append(o.unsqueeze(1))
            feats.append(f.unsqueeze(1))
        outs = torch.cat(outs, dim=1)                       # (B, E, O)
        feats = torch.cat(feats, dim=1)                     # (B, E, C_last)
        blended = torch.einsum('be,beo->bo', weights, outs) # (B, O)
        return {
            'pred': blended,
            'weights': weights,
            'logits': logits,
            'expert_preds': outs,
            'expert_feats': feats,
        }


# ------------------------------
# Shared-body model: one TCN trunk, blended output heads
# ------------------------------

class MANN_TCN_SharedBody(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tcn_channels,                # e.g. [64, 128, 128, 256]
                 gating_input: int,
                 gating_hidden: int = 128,
                 kernel_size: int = 2,
                 tcn_dropout: float = 0.2,
                 gating_dropout: float = 0.0,
                 num_experts: int = 4,
                 temperature: float = 1.0):
        super().__init__()
        self.body = TemporalConvNet(input_size, tcn_channels, kernel_size=kernel_size, dropout=tcn_dropout)
        self.gate = GatingNet(gating_input, gating_hidden, num_experts, dropout=gating_dropout, temperature=temperature)
        self.head = ExpertLinearVectorized(num_experts, tcn_channels[-1], output_size)

    def forward(self, phaseinputs, seq_input):
        feats_t = self.body(seq_input)            # (B, C_last, T)
        feats = feats_t[:, :, -1]                 # (B, C_last)
        w, logits = self.gate(phaseinputs)        # (B, E)
        blended = self.head(feats, w)             # (B, O)
        return {
            'pred': blended,
            'weights': w,
            'logits': logits,
            'feats': feats,
        }


# ------------------------------
# Regularizers for the gate (optional)
# ------------------------------

def gate_entropy(weights: torch.Tensor):
    """Higher is more uniform. Use -lambda * gate_entropy in the loss to encourage smoothing."""
    eps = 1e-8
    return -(weights * (weights + eps).log()).sum(dim=1).mean()


def load_balance_loss(weights: torch.Tensor):
    """Encourage equal average expert usage across the batch."""
    mean_w = weights.mean(dim=0)                  # (E,)
    E = weights.shape[1]
    return ((mean_w - 1.0 / E) ** 2).sum()


# ------------------------------
# Minimal demo / sanity check
# ------------------------------
if __name__ == "__main__":
    B, Cin, T, G = 8, 27, 201, 8
    O, E = 20, 4
    channels = [64, 128, 128, 256]

    seq = torch.randn(B, Cin, T)
    phases = torch.randn(B, G)

    print("Full MoE demo:")
    model_moe = MANN_TCN_MoE(
        input_size=Cin, output_size=O, num_experts=E,
        tcn_channels=channels, kernel_size=3,
        gating_input=G, gating_hidden=128,
        expert_dropout=0.2, gating_dropout=0.1, temperature=1.0
    )
    out_moe = model_moe(phases, seq)
    for k, v in out_moe.items():
        print(k, tuple(v.shape))

    print("\nShared-body demo:")
    model_sb = MANN_TCN_SharedBody(
        input_size=Cin, output_size=O, num_experts=E,
        tcn_channels=channels, kernel_size=3,
        gating_input=G, gating_hidden=128,
        tcn_dropout=0.2, gating_dropout=0.1, temperature=1.0
    )
    out_sb = model_sb(phases, seq)
    for k, v in out_sb.items():
        print(k, tuple(v.shape))

    # Example loss with regularization (fake target)
    y = torch.randn(B, O)
    loss = F.mse_loss(out_sb['pred'], y)
    # Add optional regularizers
    loss = loss - 1e-3 * gate_entropy(out_sb['weights']) + 1e-3 * load_balance_loss(out_sb['weights'])
    print("\nLoss (with regs):", float(loss))


# ------------------------------
# Dynamic-Weight TCN (expert-weighted conv kernels)
# ------------------------------

class ExpertConv1d(nn.Module):
    """
    Conv1d whose weights are a *mixture of E expert kernels* blended by per-sample
    gating weights w (B, E). This directly implements "gating dynamically changes the
    TCN weights" rather than only blending outputs.

    Shapes
      - experts: E
      - W: (E, C_out, C_in, K)
      - b: (E, C_out)
      - x: (B, C_in, T)
      - w: (B, E)
    """
    def __init__(self, experts: int, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.experts = experts
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Expert kernels & biases
        self.W = nn.Parameter(torch.empty(experts, out_ch, in_ch, kernel_size))
        self.b = nn.Parameter(torch.zeros(experts, out_ch))
        bound = math.sqrt(6.0 / (in_ch * kernel_size + out_ch))
        nn.init.uniform_(self.W, -bound, bound)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C_in, T = x.shape
        E = self.experts
        assert w.shape == (B, E), f"Expected w=(B,E) got {tuple(w.shape)}"

        # Blend expert kernels per-sample: Wb[b] = sum_e w[b,e] * W[e]
        # Vectorize blending, then apply conv per-sample
        # blended_W: (B, C_out, C_in, K), blended_b: (B, C_out)
        blended_W = torch.einsum('be,ecik->bcik', w, self.W)
        blended_b = torch.einsum('be,eo->bo', w, self.b)

        outs = []
        for b in range(B):
            out_b = F.conv1d(
                x[b:b+1], blended_W[b], blended_b[b],
                stride=self.stride, padding=self.padding, dilation=self.dilation
            )
            outs.append(out_b)
        return torch.cat(outs, dim=0)  # (B, C_out, T_out)


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
        w, logits = self.gate(phaseinputs)           # (B, E)
        feats_t = self.tcn(seq_input, w)             # (B, C_last, T)
        feats = feats_t[:, :, -1]
        out = self.head(feats)
        return {
            'pred': out,
            'weights': w,
            'logits': logits,
            'feats': feats,
        }

# Note on efficiency:
# The ExpertConv1d above blends kernels per sample and then performs a per-sample conv.
# This is simple and correct. For large batches, you can micro-batch or move to a grouped-conv
# implementation by expanding inputs/weights block-diagonally. We can add that variant if needed.
