import numpy as np
from chronos import ChronosPipeline
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from custom_datasets.PTBDataset import PTBDataset
from utils.model import restore_state_dict, calculate_smape, set_params, batch_loader
import argparse
import json

context_len = 512
pred_len = 64

batch_size = 64
batches = 50

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="The path where weights and event logs would be stored.")
parser.add_argument("--data_path", help="The path where the dataset is stored.")
parser.add_argument("--model_path", help="The path where the model weights are stored.", default=None)
parser.add_argument("--partition_path", help="The path where the partition json is stored.")

args = parser.parse_args()

data_path = args.data_path
log_path = args.log_path
partition_path = args.partition_path

print(f"Data path: {data_path}. Loading PTB-XL data...")

clients = []

with open(partition_path, "r") as f:
    json_obj = json.load(f)

pids = [k for k in json_obj["test_partition"].keys()]

for pid in pids:
    test_dataset = PTBDataset(
        partition_path=partition_path, 
        ds_path=data_path, 
        is_train=False, 
        pred_len=pred_len, 
        context_len=context_len, 
        partition_id=pid
    )
    clients.append(test_dataset)



indices_per_client = []

for client in clients:
    indices = sorted(np.random.permutation(len(client))[: (batches * batch_size)])
    indices_per_client.append(indices)


print("PTB-XL data loaded.")

model_path = "amazon/chronos-t5-tiny"

pipeline = ChronosPipeline.from_pretrained(
    model_path,
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

if args.model_path:
    npy_model = np.load(args.model_path)
    npy_params = [npy_model[file] for file in npy_model.files]
    npy_params = restore_state_dict(pipeline.model, npy_params)
    set_params(pipeline.model, npy_params)


print(f"Chronos {model_path} loaded successfully.")

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

for indices, client in zip(indices_per_client, clients):
    for i, (x, y) in enumerate(batch_loader(indices, client, batch_size)):
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

with open(args.log_path, "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE,SMAPE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]},{smapes_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")