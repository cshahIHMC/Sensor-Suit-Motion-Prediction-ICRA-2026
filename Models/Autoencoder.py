## Author - Chinmay Shah
# Simple Autoencoder setup using FCNN-style encoder and decoder

import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, num_layers, hidden_dim, dropout_rate=0.0):
        super(AutoEncoder, self).__init__()

        # Encoder: input_dim → latent_dim
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.encoder_out = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Decoder: latent_dim → input_dim (or output_dim if reconstructing another domain)
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(latent_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.decoder_out = nn.Linear(hidden_dim, output_dim)  # Change if mapping to another output

    def forward(self, x):
        # Encoding
        for i, layer in enumerate(self.encoder_layers):
            x = torch.relu(layer(x))
            x = self.dropout(x)
        # z = torch.tanh(self.encoder_out(x))  # Latent representation
        z = self.encoder_out(x)
        
        # Decoding
        x = z
        for layer in self.decoder_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        # x_recon = torch.tanh(self.decoder_out(x))  # Output
        x_recon = self.decoder_out(x)

        # return x_recon, z  # Return both reconstruction and latent
        return x_recon