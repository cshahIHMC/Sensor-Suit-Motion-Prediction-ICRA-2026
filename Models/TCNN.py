import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
 
# Chomp1d ensures the output length is the same as the input length after convolution.
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TemporalBlock defines a single TCN block, consisting of two convolutional layers, weight normalization with ReLU activations, dropout, and residual connections.
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
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

        # Sequentially stack the operations in the temporal block.
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # Downsample if the number of input channels is different from the number of output channels.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Initialize the weights of the convolutions.
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape: (batch_size, n_inputs, sequence_length)
                
        out = self.net(x)
        # After conv1: (batch_size, n_outputs, sequence_length + 2*padding - dilation*(kernel_size-1))
        # After chomp1: (batch_size, n_outputs, sequence_length)
        # After conv2: (batch_size, n_outputs, sequence_length + 2*padding - dilation*(kernel_size-1))
        # After chomp2: (batch_size, n_outputs, sequence_length)
        
        res = x if self.downsample is None else self.downsample(x)
        # If downsampling: res shape: (batch_size, n_outputs, sequence_length)


        # Return the result of adding the original input to the output of the block (residual connection) followed by ReLU activation.
        return self.relu(out + res)
        # Final shape: (batch_size, n_outputs, sequence_length)

# TemporalConvNet is a stack of temporal blocks.
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Add TemporalBlock layers with increasing dilation sizes.
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # Stack all the layers to form the temporal convolutional network.
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, num_inputs, sequence_length)
        return self.network(x)
        # Final shape: (batch_size, num_channels[-1], sequence_length)

# TCNModel includes the TCN and a final linear layer to map the output of the TCN to the desired output size.
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        # Define the TCN part of the model.
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # Linear layer to produce the final output.
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y1 = self.tcn(x)
        # y1 shape: (batch_size, num_channels[-1], sequence_length)
        # Use the output from the last time step of the TCN to make the prediction.
        out = self.linear(y1[:, :, -1])
        # out shape: (batch_size, output_size)
 
        return out

class TCNModel_Forecast(nn.Module):
    def __init__(self, input_size, output_size, horizon, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        # Define the TCN part of the model.
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # Linear layer to produce the final output.
        self.horizon = horizon
        self.linear = nn.Linear(num_channels[-1], output_size * horizon)

    def forward(self, x):
        """
        x: [B, C_in, T_in]
        returns: [B, H, C_out]   # forecasts for next H steps
        """
        y = self.tcn(x)               # [B, C_hidden, T_in]
        last = y[:, :, -1]            # [B, C_hidden]
        out = self.proj(last)         # [B, H*C_out]
        return out.view(x.size(0), self.horizon, -1)