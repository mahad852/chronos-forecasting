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

import os

context_len = 512
pred_len = 64
model_path = "amazon/chronos-t5-tiny"

data_path = "/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"

val_batch_size = 64
val_batches = 2000

max_steps_for_clients = [4000, 4000, 400]

num_rounds = 5

log_path = "logs/fed_avg_hetro2"

def calculate_smape(y_gt, y_pred):
    return np.mean(200 * np.abs(y_pred - y_gt) / (np.abs(y_pred) + np.abs(y_gt) + 1e-8))

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists(log_path):
    os.mkdir(log_path)

event_file = os.path.join(log_path, "events.txt")
open(event_file, "w").close()

client_ds = [
    VitalSignsDataset(
        user_ids=["GDN0001", "GDN0002", "GDN0003", "GDN0004", "GDN0005", "GDN0006", "GDN0007", "GDN0008", "GDN0009", "GDN0010", "GDN0011", "GDN0012", "GDN0013", "GDN0014", "GDN0015"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

    VitalSignsDataset(
        user_ids=["GDN0016", "GDN0017", "GDN0018", "GDN0019", "GDN0020", "GDN0021", "GDN0022", "GDN0023", "GDN0024", "GDN0025", "GDN0026", "GDN0027", "GDN0028", "GDN0029", "GDN0030"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

    VitalSignsDataset(
        user_ids=["GDN0001", "GDN0002", "GDN0003", "GDN0004", "GDN0005", "GDN0006", "GDN0007", "GDN0008", "GDN0009", "GDN0010", "GDN0011", "GDN0012", "GDN0013", "GDN0014", "GDN0015", 
                  "GDN0016", "GDN0017", "GDN0018", "GDN0019", "GDN0020", "GDN0021", "GDN0022", "GDN0023", "GDN0024", "GDN0025", "GDN0026", "GDN0027", "GDN0028", "GDN0029", "GDN0030"],
        data_attribute="tfm_icg",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=False,
    ),

]

def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def batch_loader(indices : List[int], dataset: VitalSignsDataset, batch_size:int):
    for start_index in range(0, len(indices), batch_size):
        end_index = min(len(indices), start_index + batch_size)
        batch_indices = indices[start_index:end_index]
        batch_x, batch_y = [], []
        for index in batch_indices:
            x, y = dataset[index]
            batch_x.append(x)
            batch_y.append(y)
        yield torch.tensor(np.array(batch_x)), torch.tensor(np.array(batch_y))

def test(pipeline: ChronosPipeline, dataset: VitalSignsDataset, indices):
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

def log_event(event_str: str):
    with open(event_file, "a") as f:
        f.write(f"{event_str}\n")

class FlowerClient(NumPyClient):
    def __init__(self, train_data_path: str, valdataset: VitalSignsDataset, cid: int) -> None:
        super().__init__()

        self.train_data_path = train_data_path
        self.valdataset = valdataset
        self.val_indices = sorted(np.random.permutation(len(valdataset))[: (val_batches * val_batch_size)])

        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        self.model = load_model(model_id=model_path)

        self.client_id = cid

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""
        
        log_event(f"STARTING training for client: {self.client_id}")

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)

        # do local training (call same function as centralised setting)
        self.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=context_len, prediction_length=pred_len, max_steps=max_steps_for_clients[self.client_id - 1])

        log_event(f"COMPLETED training for client: {self.client_id}")

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), max_steps_for_clients[self.client_id - 1], {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""
        
        log_event(f"STARTING eval for client: {self.client_id}")

        set_params(self.pipeline.model, parameters)
        # do local evaluation (call same function as centralised setting)
        mse, rmse, mae, smape = test(self.pipeline, self.valdataset, self.val_indices)
        
        with open(os.path.join(log_path, "eval_stats.txt"), "a") as f:
            f.write(f"Client: {self.client_id}; MSE: {mse} | RMSE: {rmse} | MAE: {mae} | SMAPE: {smape}\n")

        log_event(f"COMPLETED eval for client: {self.client_id}")

        # send statistics back to the server
        return float(rmse), len(self.val_indices), {"mae": mae, "mse": mse, "rmse": rmse, "smape": smape}
    

def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""

    partition_id = int(context.node_config["partition-id"])

    return FlowerClient(train_data_path=f"vital_signs_arrow/client0{partition_id + 1}.arrow", 
                        valdataset=client_ds[partition_id],
                        cid=partition_id + 1).to_client()

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    log_event("STARTING weighted averaging for evaluation metrics.")

    # Multiply accuracy of each client by number of examples used
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    mses = [num_examples * m["mse"] for num_examples, m in metrics]
    rmses = [num_examples * m["rmse"] for num_examples, m in metrics]
    smapes = [num_examples * m["smape"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    with open(os.path.join(log_path, "eval_stats.txt"), "a") as f:
        f.write(f"MSE: {sum(mses)/sum(examples)} | RMSE: {sum(rmses)/sum(examples)} | MAE: {sum(maes)/sum(examples)} | SMAPE: {sum(smapes)/sum(examples)} \n\n")

    log_event("COMPLETED weighted averaging for evaluation metrics.")

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples), "mse": sum(mses) / sum(examples), "smape" : sum(smapes)/sum(examples)}


model = load_model(model_id=model_path)
ndarrays = get_params(model)
global_model_init = ndarrays_to_parameters(ndarrays)
strategy = CustomFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
    initial_parameters=global_model_init,  # initialised global model,
    min_fit_clients=len(client_ds),
    min_evaluate_clients=len(client_ds),
    min_available_clients=len(client_ds),
    log_path=log_path,
    event_file=event_file
)

# each client gets 1xCPU (this is the default if no resources are specified)
my_client_resources = {'num_cpus': 1, 'num_gpus': 2/len(client_ds)}


start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=len(client_ds), # Total number of clients available
    config=ServerConfig(num_rounds=num_rounds), # Specify number of FL rounds
    strategy=strategy, # A Flower strategy
    ray_init_args = {'num_cpus': 1, 'num_gpus': 2},
    client_resources=my_client_resources
)
