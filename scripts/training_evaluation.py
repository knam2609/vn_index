# -----------------------------
# Training & Evaluation
# -----------------------------
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .helper import inverse_scale_predictions, set_seed, device

def train_model(model_type,
                params,
                X_train_tensor,
                y_train_tensor,
                X_test_tensor=None,
                y_test_tensor=None,
                criterion=nn.HuberLoss(),
                scaler = StandardScaler(),
                epochs=50):
    set_seed(0)
    model = model_type(
        input_size = X_train_tensor.shape[2],
        hidden_size= params["hidden_size"],
        num_layers = params.get("num_layers", 2),
        dropout    = params["dropout"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # build our Dataset that also returns the dates
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    loader   = DataLoader(train_ds, batch_size=1, shuffle=False)

    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0
        i = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            y_pred_tensor= inverse_scale_predictions(model(batch_X), scaler)
            y_true_tensor = inverse_scale_predictions(batch_y, scaler).unsqueeze(-1)
            # print('Train')
            # print(y_pred_tensor.size())
            # print(y_true_tensor.size())
            loss = criterion(y_pred_tensor, y_true_tensor)
            loss.backward()
            optimizer.step()
            total += loss.item()
            i += 1
        train_loss = total / len(loader)
        epoch_losses.append(train_loss)
        # print('Test')
        if X_test_tensor != None and y_test_tensor != None:
            test_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion, scaler)
            print(f"Epoch {_+1} - Train_Loss: {train_loss} - Test_Loss: {test_loss}")
        else:
            print(f"Epoch {_+1} - Train_Loss: {train_loss}")
    return model, np.mean(epoch_losses)

def evaluate_model(model, X_test_tensor, y_true, criterion, scaler):
    val_loss = 0
    l = len(y_true)
    y_pred_tensor= inverse_scale_predictions(model(X_test_tensor), scaler)
    y_true_tensor = inverse_scale_predictions(y_true, scaler).unsqueeze(-1)              
    val_loss = criterion(y_pred_tensor, y_true_tensor).item()
    return val_loss / l


def objective(model_type, trial, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, scaler):
    """
    Optuna objective for hyperparameter tuning.
    """
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
    }
    model, overall_avg_loss = train_model(model_type, params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, scaler, epochs=50)
    trial.set_user_attr("model", model)
    return overall_avg_loss