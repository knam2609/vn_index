# -----------------------------
# Attention Mechanisms
# -----------------------------
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from .helper import create_sequences, inverse_scale_predictions, set_seed, device
from .training_evaluation import train_model, evaluate_model, objective
from .lstm import LSTMModel

# -----------------------------
# Main Pipeline Function
# -----------------------------
def change_model(data, df=None, scaler=StandardScaler(), model_type=LSTMModel, criterion=nn.HuberLoss(), train_seq_len=60, test_seq_len=5, tuning=False,
                        best_params={"hidden_size": 128, "num_layers": 2,
                                     "dropout": 0.3, "learning_rate": 1e-4, "batch_size": 8}):
    """
    Main LSTM pipeline.
      - 'data' is the DataFrame used for scaling and creating sequences.
      - 'df' is an optional DataFrame that contains the actual VN_Index_Close values.
        If df is not provided, 'data' will be used for VN_Index_Close.
        
      IMPORTANT: Only the "Change" column is scaled and used for training.
    """
    # Use the provided df for VN_Index_Close if available; otherwise use data.
    source_df = df if df is not None else data
    if isinstance(data, pd.Series):
        data = data.to_frame()

    set_seed(0)

    train_size = len(data) - train_seq_len - test_seq_len
    scaler.fit(data.iloc[:train_size])
    data_scaled = scaler.transform(data)

    # Create sequences from the scaled target values.
    X, y, y_dates = create_sequences(data_scaled, data.index, train_seq_len, test_seq_len, target_col_idx=0)
    X_train, X_test = X[:-1], X[-1]
    y_train, y_test = y[:-1], y[-1]
    y_dates_train, y_dates_test = y_dates[:-1], y_dates[-1]
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0).to(device)

    # Hyperparameter tuning or use fixed parameters.
    if tuning:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(model_type, trial, X_train_tensor, y_train_tensor, X_tensor, y_test_tensor, criterion, scaler), n_trials=10)
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        print(f"Best Overall Average Training Loss: {study.best_value:.4f}")
        model = study.best_trial.user_attrs["model"]
    else:
        model, overall_avg_loss = train_model(model_type, best_params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, scaler, epochs=1)
        print(f"Overall Average Training Loss: {overall_avg_loss:.4f}")

    predicted_pct_tensor = model(X_test_tensor)
    val_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion, scaler)
    print(f"Final Evaluation Loss on Test Set: {val_loss:.4f}")

    # Inverse-transform predicted and true percent changes.
    predicted_pct = inverse_scale_predictions(predicted_pct_tensor, scaler).detach().cpu().numpy().ravel()
    true_pct = inverse_scale_predictions(y_test_tensor, scaler).detach().cpu().numpy().ravel()

    # Print a table comparing predicted vs. actual percent change.
    changes_df = pd.DataFrame({
        "Date": y_dates_test,
        "Predicted Change": predicted_pct,
        "Actual Change": true_pct
    })
    print("Predicted Change vs Actual Change (Test Set):")
    print(changes_df)

    # Reconstruct VN_Index_Close predictions for each test sequence.
    # For each test sequence, use the actual closing price on the last day of that sequence.
    pred_close = []
    for date, pct in zip(y_dates_test, predicted_pct):
        try:
            idx = source_df.index.get_loc(date)
        except KeyError:
            # If the exact date is not found, use padding to get the previous available date index.
            idx = source_df.index.get_indexer([date], method="pad")[0]
        # Use the previous available row (if idx is 0, we use that same value).
        if not pred_close:
            base_close = source_df.iloc[idx - 1]["VN_Index_Close"]
        else:
            base_close = pred_close[-1]
        new_close = base_close * (1 + pct)
        pred_close.append(new_close)
        
    true_close = source_df.loc[y_dates_test, "VN_Index_Close"].values

    rmse = np.sqrt(mean_squared_error(true_close, pred_close))
    mae = mean_absolute_error(true_close, pred_close)
    r2 = r2_score(true_close, pred_close)
    directional_accuracy = np.mean(np.sign(predicted_pct) == np.sign(true_pct)) if len(true_pct) > 0 else np.nan
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_dates_test, true_close, label="Actual VN-INDEX", marker="o", color="blue")
    plt.plot(y_dates_test, pred_close, label="Predicted VN-INDEX", marker="s", linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("VN-INDEX")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("LSTM Predictions vs. Actual VN-INDEX")
    plt.show()

    results_df = pd.DataFrame({
        "Date": y_dates_test,
        "Actual VN-INDEX": true_close,
        "Predicted VN-INDEX": pred_close
    })
    print("Predicted vs. Actual VN-INDEX (Test Set):")
    print(results_df)

    final_model, _ = train_model(model_type, best_params, X_tensor, y_tensor, criterion, scaler, epochs=50)

    data_for_prediction = np.array(data_scaled[-train_seq_len:])
    return final_model, torch.tensor(data_for_prediction, dtype=torch.float32).to(device), scaler, pred_close

def price_model(data, scaler=StandardScaler(), model_type=LSTMModel, criterion=nn.HuberLoss(), train_seq_len=60, test_seq_len=5, 
                tuning=True, best_params={'hidden_size': 128, 'num_layers': 2,
    'dropout': 0.3, 'learning_rate': 1e-4, 'batch_size': 8}):

    if isinstance(data, pd.Series):
        data = data.to_frame()

    set_seed(0)
    
    train_size = len(data) - train_seq_len - test_seq_len
    scaler.fit(data[:train_size])
    data_scaled = scaler.transform(data)

    X, y, y_dates = create_sequences(data_scaled, data.index, train_seq_len, test_seq_len, target_col_idx=0)
    X_train, X_test = X[:-1], X[-1]
    y_train, y_test = y[:-1], y[-1]
    y_dates_train, y_dates_test = y_dates[:-1], y_dates[-1]

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0).to(device)

    if tuning:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(model_type, trial, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, scaler), n_trials=10)
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
        print(f"Best Overall Average Training Loss: {study.best_value:.4f}")
        model = study.best_trial.user_attrs["model"]
    else:
        model, overall_avg_loss = train_model(model_type, best_params, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, criterion, scaler, epochs=1)
        print(f"Overall Average Training Loss: {overall_avg_loss:.4f}")

    y_pred_tensor = model(X_test_tensor)
    print(y_pred_tensor.shape)
    val_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion, scaler)
    print(f"Final Evaluation Loss on Test Set: {val_loss:.4f}")

    y_pred = inverse_scale_predictions(y_pred_tensor, scaler).detach().cpu().numpy().ravel()
    y_true = inverse_scale_predictions(y_test_tensor, scaler).detach().cpu().numpy().ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if len(y_true) > 1:
        directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    else:
        directional_accuracy = np.nan

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(y_dates_test, y_true, label="Actual VN-INDEX", marker='o', color="blue")
    plt.plot(y_dates_test, y_pred, label="Predicted VN-INDEX", marker='s', linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("VN-INDEX")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("LSTM Predictions vs. Actual VN-INDEX")
    plt.show()

    results_df = pd.DataFrame({
        "Date": y_dates_test,
        "Actual VN-INDEX": y_true,
        "Predicted VN-INDEX": y_pred
    })
    print("Predicted vs. Actual VN-INDEX (Test Set):")
    print(results_df)

    final_model, _ = train_model(model_type, best_params, X_tensor, y_tensor, None, None, criterion, scaler, epochs=1)

    data_for_prediction = np.array(data_scaled[-train_seq_len:])
    return final_model, torch.tensor(data_for_prediction, dtype=torch.float32).to(device), scaler, y_pred