import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) Plain LSTM → multi‐step head
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_forecasts=5):
        super().__init__()
        self.n_forecasts = n_forecasts
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        # now project to n_forecasts instead of 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_forecasts)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch, hidden_size)
        h_last = h_n[-1]                # (batch, hidden_size)
        out    = self.head(h_last)      # (batch, n_forecasts)
        # reshape to (batch, n_forecasts, 1) if you want each as a “1‐step” target
        return out.unsqueeze(-1)        # → (batch, n_forecasts, 1)


# -----------------------------
# 2) Feature‐ & Temporal‐Attention LSTM → multi‐step head
# -----------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v  = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, enc_out, final_h):
        # enc_out: (batch, seq_len, hidden), final_h: (batch, hidden)
        seq_len = enc_out.size(1)
        h_exp   = final_h.unsqueeze(1).repeat(1, seq_len, 1)
        scores  = self.v(torch.tanh(self.W1(enc_out) + self.W2(h_exp)))  # (batch,seq_len,1)
        alpha   = torch.softmax(scores, dim=1)                            # (batch,seq_len,1)
        context = (alpha * enc_out).sum(dim=1)                            # (batch, hidden)
        return context


class FeatureSelfAttnLSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_forecasts=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_forecasts = n_forecasts

        # feature‐level attention
        self.input_attn = nn.Linear(hidden_size + input_size, input_size)
        # stepwise LSTMCell
        self.lstm_cell  = nn.LSTMCell(input_size, hidden_size)
        # temporal attention
        self.temporal_attn = TemporalAttention(hidden_size)
        # multi‐step head
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_forecasts)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        b, seq_len, f = x.size()
        h = torch.zeros(b, self.hidden_size, device=x.device)
        c = torch.zeros(b, self.hidden_size, device=x.device)
        enc_h = []

        for t in range(seq_len):
            xt  = x[:, t, :]                    # (batch, input_size)
            cat = torch.cat([h, xt], dim=1)     # (batch, hidden+input)
            e   = self.input_attn(cat)          # (batch, input)
            a_f = torch.softmax(e, dim=1)       # (batch, input)
            xt_att = a_f * xt                   # feature‐weighted input
            h, c = self.lstm_cell(xt_att, (h, c))
            enc_h.append(h.unsqueeze(1))

        enc_out  = torch.cat(enc_h, dim=1)      # (batch, seq_len, hidden)
        final_h  = h                            # (batch, hidden)
        context  = self.temporal_attn(enc_out, final_h)  # (batch, hidden)
        comb     = torch.cat([final_h, context], dim=1)  # (batch, 2*hidden)
        y        = self.head(comb)              # (batch, n_forecasts)
        return y.unsqueeze(-1)                  # → (batch, n_forecasts, 1)
