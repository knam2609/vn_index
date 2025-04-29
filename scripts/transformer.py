import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                   # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)      # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                        # non-trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)             # broadcast over batch

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,     # number of features (e.g. 5: Change + cyclical embeddings)
        d_model: int = 64,   # internal embedding dim
        nhead: int = 4,      # number of attention heads
        num_layers: int = 2, # number of TransformerEncoder layers
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        n_forecasts: int = 10
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # final head: for each of the next n_forecasts steps we predict one value
        self.output_head = nn.Linear(d_model, n_forecasts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch_size, seq_len, input_size)
        Returns:
          preds: (batch_size, n_forecasts)
        """
        # 1) project into d_model space
        x = self.input_proj(x)             # → (batch, seq_len, d_model)
        # 2) add positional encoding
        x = self.pos_enc(x)                # → (batch, seq_len, d_model)
        # 3) transformer expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        # 4) encode
        h = self.transformer(x)            # → (seq_len, batch, d_model)
        # 5) take the final time-step embedding
        last = h[-1]                       # → (batch, d_model)
        # 6) map to n_forecasts outputs
        out = self.output_head(last)       # → (batch, n_forecasts)
        return out
