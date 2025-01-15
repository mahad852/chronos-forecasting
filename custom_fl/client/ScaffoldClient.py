from chronos import ChronosPipeline
from scripts.training.train import train_vital_signs, load_model

import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

import numpy as np

from custom_datasets.vital_signs_dataset import VitalSignsDataset

from flwr.common import Context

from typing import List, Dict

from utils.model import test, get_params, set_params, ScaffoldOptimizer
from utils.general import log_event
import os

model_path = "amazon/chronos-t5-tiny"

class FlowerClient(NumPyClient):
    """Flower client implementing scaffold."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        train_data_path: str, 
        valdataset: VitalSignsDataset,
        cid: int,
        val_batches: int, 
        val_batch_size: int,
        max_steps: int,
        context_len: int,
        pred_len: int,
        log_path: str,

        learning_rate: float,
        momentum: float,
        weight_decay: float,
        save_dir: str = "",
    ) -> None:

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.train_data_path = train_data_path
        self.valdataset = valdataset
        self.val_indices = sorted(np.random.permutation(len(valdataset))[: (val_batches * val_batch_size)])

        self.val_batch_size = val_batch_size

        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        self.model = load_model(model_id=model_path)

        self.client_id = cid
        self.context_len = context_len
        self.pred_len = pred_len

        self.max_steps = max_steps
        self.log_path = log_path

        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(torch.zeros(param.shape))
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        set_params(self.model, parameters)

        self.client_cv = []
        for param in self.model.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.client_id}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.client_id}.pt")

        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]

        optimizer = ScaffoldOptimizer(self.model.parameters(), self.learning_rate, self.momentum, self.weight_decay, server_cv, self.client_cv)

        self.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=self.context_len, prediction_length=self.pred_len, max_steps=self.max_steps, optimizer=(optimizer, None), learning_rate=self.learning_rate)
        
        x = parameters
        y_i = get_params(self.model)
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.max_steps))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.client_id}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            self.max_steps,
            {},
        )

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        log_event(f"STARTING eval for client: {self.client_id}")

        set_params(self.pipeline.model, parameters)
        # do local evaluation (call same function as centralised setting)
        mse, rmse, mae, smape = test(self.pipeline, self.valdataset, self.val_indices, self.pred_len, self.val_batch_size)
        
        with open(os.path.join(self.log_path, "eval_stats.txt"), "a") as f:
            f.write(f"Client: {self.client_id}; MSE: {mse} | RMSE: {rmse} | MAE: {mae} | SMAPE: {smape}\n")

        log_event(f"COMPLETED eval for client: {self.client_id}")

        # send statistics back to the server
        return float(rmse), len(self.val_indices), {"mae": mae, "mse": mse, "rmse": rmse, "smape": smape}


def get_scaffold_client_fn(
    client_ds, 
    val_batches: int, 
    val_batch_size: int, 
    max_steps_for_clients: List[int], 
    context_len: int, 
    pred_len: int, 
    log_path: str,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    learning_rate: float = 1e-3
):
    def client_fn(context: Context):
        """Returns a FlowerClient containing its data partition."""

        partition_id = int(context.node_config["partition-id"])

        return FlowerClient(train_data_path=f"vital_signs_arrow/client0{partition_id + 1}.arrow", 
                            valdataset=client_ds[partition_id],
                            cid=partition_id + 1,
                            val_batches=val_batches,
                            val_batch_size=val_batch_size,
                            max_steps=max_steps_for_clients[partition_id],
                            context_len=context_len,
                            pred_len=pred_len,
                            log_path=log_path,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            learning_rate=learning_rate).to_client()
    
    return client_fn