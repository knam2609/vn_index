import torch
import torch.nn as nn
import random

class LSTMModelMultiStep(nn.Module):
    """
    Seq2Seq LSTM for multi-step forecasting.
    """
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        n_forecasts=5,
        teacher_forcing_ratio=0.5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_forecasts = n_forecasts
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder: process past sequence
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # Decoder: predict horizon steps
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # Project decoder outputs to scalar forecasts
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, y=None):
        """
        x: (batch, seq_len, input_size)
        y: (batch, n_forecasts) optional teacher signals
        returns: (batch, n_forecasts, 1)
        """
        # Encode input sequence
        _, (h, c) = self.encoder(x)

        # Initialize decoder input: last timestep of target feature
        decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2)  # (batch,1,1)
        outputs = []

        for t in range(self.n_forecasts):
            # Run decoder for one step
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))  # dec_out: (batch,1,hidden)
            pred = self.fc(dec_out.squeeze(1))                     # (batch,1)
            outputs.append(pred.unsqueeze(1))                      # (batch,1,1)

            # Teacher forcing: choose next input
            if y is not None and random.random() < self.teacher_forcing_ratio:
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(2)
            else:
                decoder_input = pred.unsqueeze(2)

        return torch.cat(outputs, dim=1)  # (batch, n_forecasts, 1)
    
class LSTMModelMultiOutput(nn.Module):
    """
    LSTM model for multi-output time series forecasting.
    Predicts all n_forecasts in one forward pass.
    Output shape: (batch_size, n_forecasts, 1)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_forecasts: int = 5
    ):
        super().__init__()
        self.n_forecasts = n_forecasts

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Final output shape: (batch_size, n_forecasts, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_forecasts * 1)  # output flattened
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, n_forecasts, 1)
        """
        _, (h_n, _) = self.lstm(x)            # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]                 # (batch, hidden_size)
        out = self.fc(last_hidden)            # (batch, n_forecasts * 1)
        out = out.view(-1, self.n_forecasts, 1)  # reshape to (batch, n_forecasts, 1)
        return out