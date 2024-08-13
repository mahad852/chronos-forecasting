from chronos import ChronosPipeline
from scripts.training.train import train_vital_signs, load_model
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from custom_datasets.vital_signs_dataset import VitalSignsDataset
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.client import ClientApp

from typing import List
from flwr.common import Metrics

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from custom_fl.CustomFedAvg import CustomFedAvg

from flwr.simulation import start_simulation

import time

context_len = 512
pred_len = 64
model_path = "amazon/chronos-t5-tiny"

data_path = "/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"
batch_size = 64
max_steps = 100

num_rounds = 5

client_ds = [
    VitalSignsDataset(
        user_ids=["GDN0001", "GDN0002", "GDN0003", "GDN0004", "GDN0005", "GDN0006", "GDN0007", "GDN0008", "GDN0009", "GDN0010"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

    VitalSignsDataset(
        user_ids=["GDN0011", "GDN0012", "GDN0013", "GDN0014", "GDN0015", "GDN0016", "GDN0017", "GDN0018", "GDN0019", "GDN0020"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

    VitalSignsDataset(
        user_ids=["GDN0021", "GDN0022", "GDN0023", "GDN0024", "GDN0025", "GDN0026", "GDN0027", "GDN0028", "GDN0029", "GDN0030"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

]

print("Client Validation lengths...")
for i, client in enumerate(client_ds):
    print(f"Client0{i+1} Val Length: {len(client)}. Train Length (aprox.): {len(client) * 1.50}")

print("=========================================\n\n")

loader = DataLoader(client_ds[0], batch_size=batch_size, shuffle=False)

def batch_loader(indices : List[int], dataset: VitalSignsDataset, batch_size:int):
    indices = sorted(indices)
    for start_index in range(0, len(indices), batch_size):
        end_index = min(len(indices), start_index + batch_size)
        batch_indices = indices[start_index:end_index]
        batch_x, batch_y = [], []
        for index in batch_indices:
            x, y = dataset[index]
            batch_x.append(x)
            batch_y.append(y)
        yield torch.tensor(batch_x), torch.tensor(batch_y)


num_batches = 10
indices = np.random.permutation(len(client_ds[0]))[: (num_batches * batch_size)]

for x, y in batch_loader(indices, client_ds[0], batch_size):
    print(x.shape, y.shape)

# print(loader[0], loader[-1], len(loader))