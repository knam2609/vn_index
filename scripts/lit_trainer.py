import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from .helper import inverse_scale_predictions, set_seed, device

# ────────────────────────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# LightningModule
# ────────────────────────────────────────────────────────────────────────────────
class LitForecast(pl.LightningModule):
    def __init__(
        self,
        model_type,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        batch_size: int,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        criterion=nn.HuberLoss(),
        scaler=None,
    ):
        super().__init__()
        # save scalar hyperparameters; ignore others
        self.save_hyperparameters(ignore=['criterion', 'scaler', 'X_train', 'y_train', 'X_val', 'y_val'])

        # assign data attributes (not part of hparams)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

        # core forecasting model
        self.model     = model_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.criterion = criterion
        self.scaler    = scaler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y     = y.unsqueeze(-1)  # [B] -> [B,1]
        loss  = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y     = y.unsqueeze(-1)
        loss  = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate
        )

    def train_dataloader(self):
        ds = TensorDataset(self.X_train, self.y_train)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        if self.X_val is None:
            return None
        ds = TensorDataset(self.X_val, self.y_val)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


# ────────────────────────────────────────────────────────────────────────────────
# Training function
# ────────────────────────────────────────────────────────────────────────────────
def train_model(
    model_type,
    params,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor=None,
    y_test_tensor=None,
    criterion=nn.HuberLoss(),
    scaler=StandardScaler(),
    epochs=50
):
    """
    Returns: best_model, train_loss, y_pred (inverse-scaled), val_loss
    """
    # reproducibility
    set_seed(0)

    # instantiate LightningModule with flattened hyperparams and data
    lit_model = LitForecast(
        model_type    = model_type,
        input_size    = X_train_tensor.shape[2],
        hidden_size   = params['hidden_size'],
        num_layers    = params['num_layers'],
        dropout       = params['dropout'],
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        X_train       = X_train_tensor.to(device),
        y_train       = y_train_tensor.to(device),
        X_val         = None if X_test_tensor is None else X_test_tensor.to(device),
        y_val         = None if y_test_tensor is None else y_test_tensor.to(device),
        criterion     = criterion,
        scaler        = scaler
    )

    # configure callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_loss:.4f}'
    )

    # configure and run Trainer
    trainer = pl.Trainer(
        max_epochs  = epochs,
        accelerator = 'auto',  # GPU if available, else CPU
        devices     = 1,
        callbacks   = [checkpoint_cb]
    )
    trainer.fit(lit_model)

    # fetch final metrics
    train_loss = trainer.callback_metrics['train_loss'].item()
    val_loss   = trainer.callback_metrics.get('val_loss', None)
    val_loss   = val_loss.item() if val_loss is not None else None

    # get predictions on test data
    best_model = lit_model.to(device).eval()
    y_pred     = None
    if X_test_tensor is not None:
        with torch.no_grad():
            y_pred = best_model(X_test_tensor.to(device))

    logger.info(
        f"Final: train_loss={train_loss:.4f}"
        + (f", val_loss={val_loss:.4f}" if val_loss is not None else "")
    )

    return best_model, train_loss, y_pred, val_loss
