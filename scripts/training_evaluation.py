# -----------------------------
# Training & Evaluation
# -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .helper import  set_seed, device
import logging

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
                X_test_tensor=None,
                y_test_tensor=None,
                criterion=nn.HuberLoss(),
                epochs=50):
    """
    Train the model with mini-batch updates, gradient clipping, and average epoch losses.

    Returns:
        model: trained PyTorch model
        train_loss: average training loss of last epoch
        y_pred_tensor: raw predictions on test set (if provided)
        test_loss: loss on test set (if provided)
    """
    # reproducibility
    set_seed(0)

    # instantiate model and optimizer
    model = model_type(
        input_size = X_train_tensor.shape[2],
        hidden_size= params["hidden_size"],
        num_layers = params.get("num_layers", 2),
        dropout    = params["dropout"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # prepare data loader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    loader   = DataLoader(
        train_ds,
        batch_size   = params['batch_size'],
        shuffle      = True,
        pin_memory   = True,
        num_workers  = 4
    )

    y_pred_tensor = None
    test_loss     = None
    train_loss    = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
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
            # backward pass
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # average training loss for epoch
        train_loss = epoch_loss / len(loader)

        # evaluate on test set if provided
        if X_test_tensor is not None and y_test_tensor is not None:
            y_pred_tensor, test_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion)
            logger.info(f"Epoch {epoch:02d} – Train Loss: {train_loss:.4f} – Test Loss: {test_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch:02d} – Train Loss: {train_loss:.4f}")

    return model, train_loss, y_pred_tensor, test_loss


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
        val_loss = criterion(y_pred, y_true).item() / len(y_true) if criterion.reduction == 'sum' else criterion(y_pred, y_true).item()
    return y_pred, val_loss


def objective(model_type,
              trial,
              X_train_tensor,
              y_train_tensor,
              X_test_tensor,
              y_test_tensor,
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
    }
    model, train_loss, y_pred, test_loss = train_model(
        model_type,
        params,
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        criterion,
        epochs
    )
    trial.set_user_attr("model", model)
    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("y_pred_tensor", y_pred)
    return test_loss
