import numpy as np
from chronos import ChronosPipeline
import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math 
import scipy.io
from custom_datasets.vital_signs_dataset import VitalSignsDataset
from torch.utils.data import DataLoader
from typing import List, OrderedDict

context_len = 512
pred_len = 64

batch_size = 64
batches = 30000

data_path = "/home/mali2/datasets/vital_signs" # "/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"

def calculate_smape(y_gt, y_pred):
    return np.mean(200 * np.abs(y_pred - y_gt) / (np.abs(y_pred) + np.abs(y_gt) + 1e-8))

def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


print(f"Data path: {data_path}. Loading vital signs data...")

user_ids = []
for num in range(1, 31):
    # if num in [15, 18, 24]:
    #     continue
    if num < 10:
        user_id = f"GDN000{num}"
    else:
        user_id = f"GDN00{num}"
    user_ids.append(user_id)

test_dataset = VitalSignsDataset(
    user_ids=user_ids,
    data_attribute="tfm_icg",
    scenarios=["resting"],
    data_path=data_path,
    is_train=False,
    context_len=context_len,
    pred_len=pred_len
)

indices = sorted(np.random.permutation(len(test_dataset))[: (batches * batch_size)])

print("Vital signs data loaded.")

# model_path = "scripts/output/run-15/checkpoint-final/ -> mini on 100000 720_1"
model_path = "amazon/chronos-t5-tiny"


pipeline = ChronosPipeline.from_pretrained(
    model_path,
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

############################### CODE TO LOAD FED LEARNING WEIGHTS - UNCOMMENT FOR ZEROSHOT ####################################
###############################################################################################################################
###############################################################################################################################

npy_model = np.load("logs/fed_avg_hetro2/round-5-weights.npz")
npy_params = [npy_model[file] for file in npy_model.files]
set_params(pipeline.model, npy_params)

###############################################################################################################################
###############################################################################################################################

print(f"Chronos {model_path} loaded successfully.")

def batch_loader(indices : List[int], dataset: VitalSignsDataset, batch_size: int):
    for start_index in range(0, len(indices), batch_size):
        end_index = min(len(indices), start_index + batch_size)
        batch_indices = indices[start_index:end_index]
        batch_x, batch_y = [], []
        for index in batch_indices:
            x, y = dataset[index]
            batch_x.append(x)
            batch_y.append(y)
        yield torch.tensor(np.array(batch_x)), torch.tensor(np.array(batch_y))

mses = []
maes = []
smapes = []

mse_by_pred_len = {}
rmse_by_pred_len = {}
mae_by_pred_len = {}
smapes_by_pred_len = {}

total = 0

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] = 0.0
    rmse_by_pred_len[p_len] = 0.0
    mae_by_pred_len[p_len] = 0.0
    smapes_by_pred_len[p_len] = 0.0

for i, (x, y) in enumerate(batch_loader(indices, test_dataset, batch_size)):
    forecast = pipeline.predict(
        context=x,
        prediction_length=pred_len,
        num_samples=20,
    )

    forecast = np.quantile(forecast.numpy(), 0.5, axis=1)

    mse = mean_squared_error(y, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, forecast)
    smape = calculate_smape(y.numpy(), forecast)

    mses.append(mse)
    maes.append(mae)
    smapes.append(smape)

    total += 1

    for p_len in range(1, pred_len + 1):
        mse_by_pred_len[p_len] += mean_squared_error(y[:, :p_len], forecast[:, :p_len])
        mae_by_pred_len[p_len] += mean_absolute_error(y[:, :p_len], forecast[:, :p_len])
        smapes_by_pred_len[p_len] += calculate_smape(y[:, :p_len].numpy(), forecast[:, :p_len])

    if i % 20 == 0:
        print(f"iteraition: {i} | MSE: {mse} RMSE: {rmse} MAE: {mae} SMAPE: {smape}")

    if i % 200 == 0:
        print(f"AGG STATS | Iteration: {i} | MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)} SMAPE: {np.average(smapes)}")

print(f"MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)} SMAPE: {np.average(smapes)}")

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] /= total
    rmse_by_pred_len[p_len] = np.sqrt(mse_by_pred_len[p_len])
    mae_by_pred_len[p_len] /= total
    smapes_by_pred_len[p_len] /= total

if not os.path.exists("logs"):
    os.mkdir("logs")

# with open(os.path.join("logs", f"Chronos_Tiny_ZS_{context_len}_{pred_len}.csv"), "w") as f:
# with open(os.path.join("logs", f"Chronos_Tiny_FA_Apnea_{context_len}_{pred_len}.csv"), "w") as f:
# with open(os.path.join("logs", f"Chronos_Tiny_FAH_TiltDown_{context_len}_{pred_len}.csv"), "w") as f:
with open(os.path.join("logs", f"Chronos_Tiny_FAH2_ICG_{context_len}_{pred_len}.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE,SMAPE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]},{smapes_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")