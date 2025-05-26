import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)  # type: ignore # (1, seq_len, d_model)

# -----------------------------
# Transformer with decoder (seq2seq style)
# -----------------------------
class TimeSeriesTransformerMultiStep(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,   # used as d_model
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 n_forecasts: int = 10):
        super().__init__()
        self.d_model = hidden_size
        self.n_forecasts = n_forecasts

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.query_embed = nn.Parameter(torch.zeros(n_forecasts, self.d_model))
        self.output_head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.input_proj(x)             # (B, L, d_model)
        x = self.pos_enc(x)                # (B, L, d_model)
        x = x.permute(1, 0, 2)             # (L, B, d_model)
        memory = self.encoder(x)           # (L, B, d_model)

        tgt = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)  # (T, B, d_model)
        tgt = self.pos_enc(tgt.permute(1, 0, 2)).permute(1, 0, 2)

        out = self.decoder(tgt=tgt, memory=memory)    # (T, B, d_model)
        out = self.output_head(out)                   # (T, B, 1)
        out = out.permute(1, 0, 2)                    # (B, T, 1)
        return out

# -----------------------------
# Transformer with encoder-only (direct multi-output)
# -----------------------------
class TimeSeriesTransformerMultiOutput(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 n_forecasts: int = 10):
        super().__init__()
        self.n_forecasts = n_forecasts
        self.d_model = hidden_size

        self.input_proj = nn.Linear(input_size, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_forecasts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # (B, L, d_model)
        x = self.pos_enc(x)                  # (B, L, d_model)
        x = x.permute(1, 0, 2)               # (L, B, d_model)
        x = self.encoder(x)                  # (L, B, d_model)
        x = x.permute(1, 2, 0)               # (B, d_model, L)
        x = self.global_pool(x).squeeze(-1)  # (B, d_model)
        out = self.output_head(x)            # (B, n_forecasts)
        out = out.unsqueeze(-1)              # (B, n_forecasts, 1)
        return out
