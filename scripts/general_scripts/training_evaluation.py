# -----------------------------
# Training & Evaluation
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .helper import  set_seed, device
import logging

lam = 10.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_model(model_type,
                params,
                X_train_tensor,
                y_train_tensor,
                X_val_tensor,
                y_val_tensor,
                criterion=nn.HuberLoss(),
                epochs=50):
    """
    Train the model with mini-batch updates, gradient clipping, and average epoch losses.

    Returns:
        model: trained PyTorch model
        train_loss: average training loss of last epoch
        val_loss: validation loss of last epoch
    """
    # reproducibility
    set_seed(0)

    # instantiate model and optimizer
    model = model_type(
        input_size = X_train_tensor.shape[2],
        hidden_size= params["hidden_size"],
        num_layers = params.get("num_layers", 2),
        dropout    = params["dropout"],
        n_forecasts = params["n_forecasts"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # scheduler to reduce LR on plateau of val loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # prepare data loader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader   = DataLoader(
        train_ds,
        batch_size   = params['batch_size'],
        shuffle      = True,
        pin_memory   = True,
        num_workers  = 4
    )

    val_loss     = None
    train_loss    = None

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        prev_y_true = None

        for batch_X, batch_y in train_loader:
            # move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            # forward pass
            y_pred = model(batch_X)
            # align shapes: [B] -> [B,1]
            y_true = batch_y.unsqueeze(-1)

            # compute loss
            loss = criterion(y_pred, y_true)
            # reweight:
            if prev_y_true is not None:
                weights = 1 + lam * torch.abs((y_true - prev_y_true[-y_true.shape[0]:])).mean()
                loss = weights * loss

            # backward pass
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            prev_y_true = y_true

        # average training loss for epoch
        train_loss = epoch_train_loss / len(train_loader)

        # validate on valiation set
        _, val_loss = evaluate_model(model, X_val_tensor, y_val_tensor, criterion)

        # scheduler step
        scheduler.step(val_loss)

        # logger.info(f"Epoch {epoch:02d} – Train Loss: {train_loss:.4f} – Val Loss: {val_loss:.4f}")

    return model, train_loss, val_loss


def evaluate_model(model, X_test_tensor, y_true_tensor, criterion):
    """
    Evaluate the model on test data.

    Returns:
        y_pred: raw predictions tensor
        val_loss: computed loss on test set
    """
    model.eval()
    with torch.no_grad():
        X = X_test_tensor.to(device)
        y_pred = model(X)
        # assume y_true_tensor is already on device or move it
        y_true = y_true_tensor.to(device).unsqueeze(-1)
        val_loss = criterion(y_pred, y_true).item() #/ len(y_true) if criterion.reduction == 'sum' else criterion(y_pred, y_true).item()
    return y_pred, val_loss


def objective(model_type,
              trial,
              X_train_tensor,
              y_train_tensor,
              X_val_tensor,
              y_val_tensor,
              criterion=nn.HuberLoss(),
              epochs = 50):
    """
    Optuna objective for hyperparameter tuning.
    """
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "n_forecasts": y_val_tensor.shape[1]
    }
    model, train_loss, val_loss = train_model(
        model_type,
        params,
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        criterion,
        epochs
    )
    trial.set_user_attr("model", model)
    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("val_loss", val_loss)
    return val_loss
