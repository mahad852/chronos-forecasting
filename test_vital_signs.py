import numpy as np
from chronos import ChronosPipeline
import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math 
import scipy.io
from custom_datasets.vital_signs_dataset import VitalSignsDataset
from torch.utils.data import DataLoader

context_len = 600
pred_len = 64

data_path = "/home/x-mali3/datasets/vital_signs" # "/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"

print(f"Data path: {data_path}. Loading vital signs data...")


user_ids = []
for num in range(1, 31):
    if num < 10:
        user_id = f"GDN000{num}"
    else:
        user_id = f"GDN00{num}"
    user_ids.append(user_id)

test_loader = DataLoader(VitalSignsDataset(
    user_ids=user_ids,
    data_attribute="tfm_ecg2",
    scenarios=["Resting"],
    data_path=data_path,
    is_train=False,
    context_len=context_len,
    pred_len=pred_len
), batch_size=1, shuffle=False)

print("Vital signs data loaded.")

# model_path = "scripts/output/run-15/checkpoint-final/ -> mini on 100000 720_1"
model_path = "amazon/chronos-t5-tiny"


pipeline = ChronosPipeline.from_pretrained(
    model_path,
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

print(f"Chronos {model_path} loaded successfully.")

mses = []
maes = []

mse_by_pred_len = {}
rmse_by_pred_len = {}
mae_by_pred_len = {}

total = 0

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] = 0.0
    rmse_by_pred_len[p_len] = 0.0
    mae_by_pred_len[p_len] = 0.0

for i, (x, y) in enumerate(test_loader):
    forecast = pipeline.predict(
        context=x,
        prediction_length=pred_len,
        num_samples=20,
    )

    forecast = np.quantile(forecast.numpy(), 0.5, axis=1)

    mse = mean_squared_error(y, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, forecast)

    mses.append(mse)
    maes.append(mae)

    total += 1

    for p_len in range(1, pred_len + 1):
        mse_by_pred_len[p_len] += mean_squared_error(y[:, :p_len], forecast[:, :p_len])
        mae_by_pred_len[p_len] += mean_absolute_error(y[:, :p_len], forecast[:, :p_len])

    if i % 20 == 0:
        print(f"iteraition: {i} | MSE: {mse} RMSE: {rmse} MAE: {mae}")

    if i % 200 == 0:
        print(f"AGG STATS | Iteration: {i} | MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)}")

print(f"MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)}")

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] /= total
    rmse_by_pred_len[p_len] = np.sqrt(mse_by_pred_len[p_len])
    mae_by_pred_len[p_len] /= total

if not os.path.exists("logs"):
    os.mkdir("logs")

# with open(os.path.join("logs", f"Chronos_Mini_Run8_{context_len}_{pred_len}.csv"), "w") as f:
with open(os.path.join("logs", f"Chronos_Tiny_{context_len}_{pred_len}.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")