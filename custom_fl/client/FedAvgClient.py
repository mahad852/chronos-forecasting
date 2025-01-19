from chronos import ChronosPipeline
from scripts.training.train import train_vital_signs, load_model

import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

import numpy as np

from custom_datasets.vital_signs_dataset import VitalSignsDataset

from flwr.common import Context

from typing import List, Dict

from utils.model import test, get_params, set_params
from utils.general import log_event
import os

model_path = "amazon/chronos-t5-tiny"

class FlowerClient(NumPyClient):
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
        log_path: str
    ) -> None:
        
        super().__init__()

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
        self.events_path = os.path.join(self.log_path, "events.txt")

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""
        
        log_event(self.events_path, f"STARTING training for client: {self.client_id}")

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)

        # do local training (call same function as centralised setting)
        self.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=self.context_len, prediction_length=self.pred_len, max_steps=self.max_steps)

        log_event(self.events_path, f"COMPLETED training for client: {self.client_id}")

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), self.max_steps, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""
        
        log_event(self.events_path, f"STARTING eval for client: {self.client_id}")

        set_params(self.pipeline.model, parameters)
        # do local evaluation (call same function as centralised setting)
        mse, rmse, mae, smape = test(self.pipeline, self.valdataset, self.val_indices, self.pred_len, self.val_batch_size)
        
        with open(os.path.join(self.log_path, "eval_stats.txt"), "a") as f:
            f.write(f"Client: {self.client_id}; MSE: {mse} | RMSE: {rmse} | MAE: {mae} | SMAPE: {smape}\n")

        log_event(self.events_path, f"COMPLETED eval for client: {self.client_id}")

        # send statistics back to the server
        return float(rmse), len(self.val_indices), {"mae": mae, "mse": mse, "rmse": rmse, "smape": smape}
    

def get_fedavg_client_fn(client_ds, train_root: str, val_batches: int, val_batch_size: int, max_steps_for_clients: List[int], context_len: int, pred_len: int, log_path: str):
    def client_fn(context: Context):
        """Returns a FlowerClient containing its data partition."""

        partition_id = int(context.node_config["partition-id"])

        return FlowerClient(train_data_path=f"{train_root}/client0{partition_id + 1}.arrow", 
                            valdataset=client_ds[partition_id],
                            cid=partition_id + 1,
                            val_batches=val_batches,
                            val_batch_size=val_batch_size,
                            max_steps=max_steps_for_clients[partition_id],
                            context_len=context_len,
                            pred_len=pred_len,
                            log_path=log_path).to_client()
    
    return client_fn