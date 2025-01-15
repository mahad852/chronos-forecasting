from chronos import ChronosPipeline
from custom_datasets.vital_signs_dataset import VitalSignsDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Tuple, Callable
import torch
from utils.metrics import calculate_smape
from collections import OrderedDict
from torch.optim import SGD, Optimizer

from flwr.common import Metrics

from utils.general import log_event

import os

def set_params(model: torch.nn.Module, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model: torch.nn.Module):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def batch_loader(indices: List[int], dataset: VitalSignsDataset, batch_size: int):
    for start_index in range(0, len(indices), batch_size):
        end_index = min(len(indices), start_index + batch_size)
        batch_indices = indices[start_index:end_index]
        batch_x, batch_y = [], []
        for index in batch_indices:
            x, y = dataset[index]
            batch_x.append(x)
            batch_y.append(y)
        yield torch.tensor(np.array(batch_x)), torch.tensor(np.array(batch_y))

def test(pipeline: ChronosPipeline, dataset: VitalSignsDataset, indices: List[int], pred_len: int, val_batch_size: int):
    mses = []
    rmses = []
    maes = []
    smapes = []

    for _, (x, y) in enumerate(batch_loader(indices, dataset, val_batch_size)):
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
        rmses.append(rmse)
        maes.append(mae)
        smapes.append(smape)

    return np.average(mses), np.average(rmses), np.average(maes), np.average(smapes)

def gen_weighted_avergage_fn(log_path: str) -> Callable:
    # Define metric aggregation function
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        log_event(os.path.join(log_path, "events.txt"), "STARTING weighted averaging for evaluation metrics.")

        # Multiply accuracy of each client by number of examples used
        maes = [num_examples * m["mae"] for num_examples, m in metrics]
        mses = [num_examples * m["mse"] for num_examples, m in metrics]
        rmses = [num_examples * m["rmse"] for num_examples, m in metrics]
        smapes = [num_examples * m["smape"] for num_examples, m in metrics]

        examples = [num_examples for num_examples, _ in metrics]

        with open(os.path.join(log_path, "eval_stats.txt"), "a") as f:
            f.write(f"MSE: {sum(mses)/sum(examples)} | RMSE: {sum(rmses)/sum(examples)} | MAE: {sum(maes)/sum(examples)} | SMAPE: {sum(smapes)/sum(examples)} \n\n")

        log_event(os.path.join(log_path, "events.txt"), "COMPLETED weighted averaging for evaluation metrics.")

        # Aggregate and return custom metric (weighted average)
        return {"mae": sum(maes) / sum(examples), "mse": sum(mses) / sum(examples), "smape" : sum(smapes)/sum(examples)}
    
    return weighted_average


class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, grads, step_size, momentum, weight_decay, server_cv, client_cv):
        super().__init__(
            grads, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )
        
        self.server_cv = server_cv
        self.client_cv = client_cv

        print("***" * 50)
        print(f"Server_cv len: {len(server_cv)}; client_cv len: {len(client_cv)}")

    def step(self, closure=None):
        """Implement the custom step function fo SCAFFOLD."""
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        super().step(closure=closure)
        
        for group in self.param_groups:
            print("params len:", len(group["params"]))
            for par, s_cv, c_cv in zip(group["params"], self.server_cv, self.client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])