import numpy as np
import os
import random
import torch
import torch.nn as nn

# ✅ Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Directional Loss Definition
# -----------------------------
class DirectionalLoss(nn.Module):
    """
    Directional loss: 1 - directional accuracy.
    Returns the fraction of predictions whose sign differs from the true sign.
    """
    def __init__(self):
        super().__init__()
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # print("Loss")
        # print(y_pred.shape)
        # print(y_true.shape)
        # drop any trailing singleton dim => (B, H)
        if y_pred.dim() == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
        if y_true.dim() == 3 and y_true.size(-1) == 1:
            y_true = y_true.squeeze(-1)

        # shapes must match
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes must match for DirectionalLoss: {y_pred.shape} vs {y_true.shape}")

        # print(y_pred.shape)
        # print(y_true.shape)
        # no direction to compare if horizon < 2
        if y_pred.size(1) < 2:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # compute day-to-day changes
        diff_p = y_pred[:, 1:] - y_pred[:, :-1]   # (B, H-1)
        diff_t = y_true[:, 1:] - y_true[:, :-1]   # (B, H-1)

        # directional accuracy
        correct = (torch.sign(diff_p) == torch.sign(diff_t)).float()
        acc = correct.mean()                      # fraction correct

        return 1.0 - acc                           # directional loss

# -----------------------------
# Customized Loss Definition
# -----------------------------    
class CustomizedLoss(nn.Module):
    """
    Combined Huber + directional loss:
      loss = alpha * Huber + beta * DirectionalLoss
    """
    def __init__(self, alpha: float = 1, beta: float = 0.1):
        """
        Args:
            λ: weight on the directional term (between 0 and 1)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber_criterion = nn.HuberLoss()
        self.dir_criterion = DirectionalLoss()

    def forward(self, y_pred, y_true):
        huber_loss = self.huber_criterion(y_pred, y_true)
        dir_loss = self.dir_criterion(y_pred, y_true)
        loss = self.alpha * huber_loss + self.beta * dir_loss
        return loss                            
        
# -----------------------------
# Helper Functions
# -----------------------------
def create_sequences(data, dates, train_seq_len=60, test_seq_len=10, target_col_idx=0):
    X, y, y_dates = [], [], []
    for i in range(0, len(data) - train_seq_len - test_seq_len + 1):
        X.append(data[i:i + train_seq_len])
        y.append(data[i + train_seq_len:i + train_seq_len + test_seq_len, target_col_idx])
        y_dates.append(dates[i + train_seq_len:i + train_seq_len + test_seq_len])
    print(y_dates[-10:])
    print(dates[-1])
    return np.array(X), np.array(y), np.array(y_dates)

def inverse_scale_predictions(predictions, scaler, feature_idx=0):
    """
    Inverse-scale predictions tensor using scaler parameters (MinMaxScaler or StandardScaler)
    in a differentiable way. Suitable for use during training.
    """
    device = predictions.device
    dtype = predictions.dtype

    if hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
        # MinMaxScaler inverse transform
        scale = torch.tensor(scaler.scale_[feature_idx], dtype=dtype, device=device)
        min_ = torch.tensor(scaler.min_[feature_idx], dtype=dtype, device=device)
        inv_predictions = (predictions - min_) / scale

    elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        # StandardScaler inverse transform
        mean = torch.tensor(scaler.mean_[feature_idx], dtype=dtype, device=device)
        scale = torch.tensor(scaler.scale_[feature_idx], dtype=dtype, device=device)
        inv_predictions = predictions * scale + mean

    else:
        raise ValueError("Scaler type not supported. Only MinMaxScaler and StandardScaler are supported.")

    return inv_predictions

def scale_value(value, scaler, feature_idx=0):
    # StandardScaler
    if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        return (value - scaler.mean_[feature_idx]) / scaler.scale_[feature_idx]
    # MinMaxScaler
    elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        return (value - scaler.data_min_[feature_idx]) / (
            scaler.data_max_[feature_idx] - scaler.data_min_[feature_idx]
        )
    else:
        raise ValueError(f"Unsupported scaler type: {type(scaler)}")

def inverse_scale_value(value, scaler, feature_idx=0):
    # StandardScaler
    if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        return (value * scaler.scale_[feature_idx] + scaler.mean_[feature_idx])
    # MinMaxScaler
    elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        return value * (scaler.data_max_[feature_idx] - scaler.data_min_[feature_idx]) + scaler.data_min_[feature_idx]
    else:
        raise ValueError(f"Unsupported scaler type: {type(scaler)}")

def set_seed(seed: int = 42):
    # 1) Python built-ins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch (CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU

    # 4) cuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
