import torch
import torch.nn as nn
import random

class LSTMModel(nn.Module):
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


class TemporalAttention(nn.Module):
    """
    Computes attention over encoder outputs given the final hidden state.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v    = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, final_hidden):
        # encoder_outputs: (batch, seq_len, hidden)
        # final_hidden:    (batch, hidden)
        seq_len = encoder_outputs.size(1)
        # repeat final hidden
        final_expanded = final_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # energy scores
        energy = torch.tanh(self.attn(torch.cat([encoder_outputs, final_expanded], dim=2)))
        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(energy).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        # weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


class FeatureSelfAttnLSTMForecast(nn.Module):
    """
    Seq2Seq attention-based LSTM for multi-step forecasting.
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

        # feature-level attention
        self.input_attn = nn.Linear(hidden_size + input_size, input_size)
        # encoder cells
        self.cells = nn.ModuleList([
            nn.LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size
            ) for i in range(num_layers)
        ])
        # temporal attention
        self.temporal_attn = TemporalAttention(hidden_size)
        # decoder LSTM for multi-step
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # projection for decoder outputs
        self.fc_decoder = nn.Linear(hidden_size, 1)

    def forward(self, x, y=None):
        """
        x: (batch, seq_len, input_size)
        y: (batch, n_forecasts) optional teacher signals
        returns: (batch, n_forecasts, 1)
        """
        batch_size, seq_len, _ = x.size()
        # initialize hidden & cell states
        h_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]
        enc_hiddens = []

        # Encoder with feature attention
        for t in range(seq_len):
            xt = x[:, t, :]
            # feature attention
            cat = torch.cat([h_states[-1], xt], dim=1)
            e = self.input_attn(cat)
            a_f = torch.softmax(e, dim=1)
            xt_att = a_f * xt
            # pass through cells
            h0, c0 = self.cells[0](xt_att, (h_states[0], c_states[0]))
            h_states[0], c_states[0] = h0, c0
            for i in range(1, self.num_layers):
                hi, ci = self.cells[i](h_states[i-1], (h_states[i], c_states[i]))
                h_states[i], c_states[i] = hi, ci
            enc_hiddens.append(h_states[-1].unsqueeze(1))

        enc_out = torch.cat(enc_hiddens, dim=1)  # (batch, seq_len, hidden)
        final_h = h_states[-1]                   # (batch, hidden)
        # temporal attention context
        context = self.temporal_attn(enc_out, final_h)

        # prepare decoder initial states
        h_dec = torch.stack(h_states)  # (num_layers, batch, hidden)
        c_dec = torch.stack(c_states)
        decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2)  # (batch,1,1)
        outputs = []

        # decode multi-step
        for t in range(self.n_forecasts):
            dec_out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            dec_h = dec_out.squeeze(1)         # (batch, hidden)
            pred  = self.fc_decoder(dec_h)     # (batch,1)
            outputs.append(pred.unsqueeze(1))  # (batch,1,1)
            # teacher forcing
            if y is not None and random.random() < self.teacher_forcing_ratio:
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(2)
            else:
                decoder_input = pred.unsqueeze(2)

        return torch.cat(outputs, dim=1)       # (batch, n_forecasts, 1)
