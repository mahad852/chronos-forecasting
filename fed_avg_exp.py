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
from flwr.server.strategy import FedAvg

from flwr.simulation import run_simulation

context_len = 600
pred_len = 60
model_path = "amazon/chronos-t5-tiny"

data_path = "/home/x-mali3/datasets/vital_signs" # "/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"
batch_size = 16
max_steps = 10

num_rounds = 5

client_ds = [
    VitalSignsDataset(
        user_ids=["GDN0001", "GDN0002", "GDN0003", "GDN0004", "GDN0005", "GDN0006", "GDN0007", "GDN0008", "GDN0009", "GDN0010"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=True,
    ),

    VitalSignsDataset(
        user_ids=["GDN0011", "GDN0012", "GDN0013", "GDN0014", "GDN0015", "GDN0016", "GDN0017", "GDN0018", "GDN0019", "GDN0020"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=True,
    ),

    VitalSignsDataset(
        user_ids=["GDN0021", "GDN0022", "GDN0023", "GDN0024", "GDN0025", "GDN0026", "GDN0027", "GDN0028", "GDN0029", "GDN0030"],
        data_attribute="tfm_ecg2",
        scenarios=["resting"],
        context_len=context_len, pred_len=pred_len, 
        data_path=data_path, 
        is_train=True,
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


def test(pipeline: ChronosPipeline, val_loader):
    mses = []
    rmses = []
    maes = []

    for _, (x, y) in enumerate(val_loader):
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
        rmses.append(rmse)
        maes.append(mae)

    return np.average(mses), np.average(rmses), np.average(maes)

class FlowerClient(NumPyClient):
    def __init__(self, train_data_path, valloader) -> None:
        super().__init__()

        self.train_data_path = train_data_path
        self.valloader = valloader
        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        self.model = load_model(model_id=model_path)

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)

        # do local training (call same function as centralised setting)
        self.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=context_len, prediction_length=pred_len, max_steps=5000)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), max_steps, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.pipeline.model, parameters)
        # do local evaluation (call same function as centralised setting)
        mse, rmse, mae = test(self.pipeline, self.valloader)
        # send statistics back to the server
        return float(rmse), len(self.valloader), {"mae": mae, "mse": mse}
    

def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""

    partition_id = int(context.node_config["partition-id"])

    return FlowerClient(train_data_path=f"vital_signs_arrow/client0{partition_id + 1}.arrow", 
                        valloader=DataLoader(client_ds[partition_id], batch_size=batch_size, shuffle=False)).to_client()

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    mses = [num_examples * m["mse"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples), "mse": sum(mses) / sum(examples)}


# Concstruct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)

def server_fn(context: Context):
    # instantiate the model
    model = load_model(model_id=model_path)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=0.1,  # 10% clients sampled each round to do fit()
        fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
        evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
        initial_parameters=global_model_init,  # initialised global model
    )

    # Construct ServerConfig
    config = ServerConfig(num_rounds=num_rounds)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(strategy=strategy, config=config)

# Create your ServerApp
server_app = ServerApp(server_fn=server_fn)


run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=3
)


# pipeline = ChronosPipeline.from_pretrained(
#     model_path,
#     device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
#     torch_dtype=torch.bfloat16,
# )

# model = load_model(model_id=model_path)
# model = train_vital_signs(training_data_paths=["vital_signs_arrow/client01.arrow"], model=model, context_length=context_len, prediction_length=pred_len, output_dir="weights/client01/", max_steps=10)

# params = get_params(model)
# set_params(model, get_params(model))
# set_params(pipeline.model, params)

# print("Param setting and getting successful")

# print("Testing pipeline prediction")


# print(f"Data path: {data_path}. Loading vital signs data...")


# user_ids = []
# for num in range(1, 31):
#     if num < 10:
#         user_id = f"GDN000{num}"
#     else:
#         user_id = f"GDN00{num}"
#     user_ids.append(user_id)

# test_loader = DataLoader(VitalSignsDataset(
#     user_ids=user_ids,
#     data_attribute="tfm_ecg2",
#     scenarios=["Resting"],
#     data_path=data_path,
#     is_train=False,
#     context_len=context_len,
#     pred_len=pred_len
# ), batch_size=1, shuffle=False)

# print("Vital signs data loaded.")

# for _, (x, y) in enumerate(test_loader):
#     forecast = pipeline.predict(
#         context=x,
#         prediction_length=pred_len,
#         num_samples=20,
#     )

#     forecast = np.quantile(forecast.numpy(), 0.5, axis=1)

#     mse = mean_squared_error(y, forecast)

#     print("MSE:", mse)
