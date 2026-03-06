# ## Author: Chinmay Shah
# # LSTM Motion Prediction Model for Hip/Knee/Ankle Joint Angles and Moments

# import torch
# import torch.nn as nn

# class LSTM(nn.Module):
#     """
#     LSTM-based motion prediction model.
#     Supports multi-step future prediction of joint angles and moments.
#     """
#     def __init__(self, input_size, hidden_size=256, num_layers=3,
#                  output_size=20, pred_horizon=1, dropout=0.2):
#         """
#         Args:
#             input_size (int): Number of input features per timestep (e.g., IMU signals, past joint angles).
#             hidden_size (int): Hidden size of LSTM layers.
#             num_layers (int): Number of stacked LSTM layers.
#             output_size (int): Number of outputs per timestep (joint angles + moments).
#             pred_horizon (int): Number of future timesteps to predict.
#             dropout (float): Dropout probability between LSTM layers.
#         """
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.pred_horizon = pred_horizon
#         self.output_size = output_size

#         # LSTM backbone
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout
#         )

#         # MLP head to map hidden state to output features
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )
#         # Linear to project autoregressive output back to hidden_size
#         self.output_to_hidden = nn.Linear(output_size, hidden_size)
        

#     def forward(self, x, future_steps=None):
#         """
#         Forward pass.

#         Args:
#             x: Input tensor of shape (batch_size, seq_len, input_size)
#             future_steps: Number of future steps to predict (overrides self.pred_horizon if provided)

#         Returns:
#             predictions: Tensor of shape (batch_size, T_pred, output_size)
#         """
#         x = x.transpose(1, 2)  # (B, input_size, T_in) -> (B, T_in, input_size)
#         batch_size, seq_len, _ = x.size()
#         T_pred = self.pred_horizon if future_steps is None else future_steps

#         # Pass input sequence through LSTM
#         lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, T_in, hidden_size)

#         # Use last timestep hidden state as starting point
#         last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)

#         # Autoregressive multi-step prediction
#         predictions = []
#         current_input = last_hidden
#         for t in range(T_pred):
#             out_t = self.fc(current_input)  # (B, output_size)
#             predictions.append(out_t.unsqueeze(1))  # add timestep dimension
#             current_input = self.output_to_hidden(out_t)

#         predictions = torch.cat(predictions, dim=1)  # (B, T_pred, output_size)
#         predictions = predictions.transpose(1, 2)  # (B, output_size, T_pred)
#         predictions = predictions.squeeze(-1)
#         return predictions


## Author: Chinmay Shah
# Direct Multi-Horizon LSTM Motion Prediction Model

import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM-based direct multi-horizon motion predictor.
    Predicts all future timesteps in one forward pass (non-autoregressive).
    """

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_size=20, pred_horizon=1, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_horizon = pred_horizon
        self.output_size = output_size

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Direct multi-horizon prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size * pred_horizon)
        )

    def forward(self, x, future_steps=None):
        """
        Args:
            x: (B, input_size, T_in)
            future_steps: optional override for prediction horizon

        Returns:
            predictions: (B, output_size, T_pred)
        """

        x = x.transpose(1, 2)  # (B, input_size, T_in) -> (B, T_in, input_size)
        B = x.size(0)

        T_pred = self.pred_horizon if future_steps is None else future_steps

        # Encode input sequence
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)

        # Directly predict all future steps
        out = self.fc(last_hidden)  # (B, output_size * T_pred)

        # Reshape to (B, T_pred, output_size)
        out = out.view(B, self.pred_horizon, self.output_size)

        # Match your previous output format: (B, output_size, T_pred)
        out = out.transpose(1, 2)

        if T_pred == 1:
            out = out.squeeze(-1)

        return out