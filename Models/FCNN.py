"""
Fully-connected sliding-window baseline predictor (FCNN_SW). Flattens a window of
past IMU/kinematic input features and directly regresses the joint angle/moment
targets for one or more future timesteps.

Author: Chinmay Shah
Institution: Institute for Human and Machine Cognition (IHMC) / University of West Florida (UWF)
"""

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """Feed-forward baseline: flattened input window -> hidden MLP -> (output_dim x prediction_horizon)."""
    def __init__(self, inputs, outputs, numOfLayers, hiddenDimension, input_seq_len=1, predictionHorizon=1, dropoutRate=0.2):
        super(FCNN, self).__init__()
        
        self.input_seq_len = input_seq_len  
        self.prediction_horizon = predictionHorizon
        self.output_dim = outputs
        
        # Initilize the first layer
        self.layers = nn.ModuleList([nn.Linear(inputs * self.input_seq_len, hiddenDimension)])
        # self.layers.append(nn.Dropout(p=dropoutRate))
        
        # Iterate over and add all the hidden layers
        for _ in range(numOfLayers - 1):
            self.layers.append(nn.Linear(hiddenDimension, hiddenDimension))
            
        # Store dropout layer
        self.dropout = nn.Dropout(p=dropoutRate)
        
        # Add the output Layer
        self.layers.append(nn.Linear(hiddenDimension, self.output_dim * self.prediction_horizon))
        
    def forward(self, x):
        
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        
        # No activation on the output layer
        x = self.layers[-1](x)
        
        # Reshape to (batch, horizon, output_dim)
        x = x.view(x.size(0), self.output_dim, self.prediction_horizon)
        x = x.squeeze(-1)  # Remove the horizon dimension if it's 1
        
        return x
        
        