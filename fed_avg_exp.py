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


context_len = 600
pred_len = 60
model_path = "amazon/chronos-t5-tiny"

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

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.pipeline.model, parameters)

        # do local training (call same function as centralised setting)
        self.pipeline.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=context_len, prediction_length=pred_len, max_steps=5000)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.pipeline.model), 5000, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.pipeline.model, parameters)
        # do local evaluation (call same function as centralised setting)
        loss, accuracy = test(self.pipeline, self.valloader)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}

pipeline = ChronosPipeline.from_pretrained(
    model_path,
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

pipeline.model = load_model(model_id=model_path)

pipeline.model.config.prediction_length = pred_len
pipeline.model.config.context_length = context_len

model = train_vital_signs(training_data_paths=["vital_signs_arrow/client01.arrow"], model=pipeline.model, context_length=context_len, prediction_length=pred_len, output_dir="weights/client01/", max_steps=10)

get_params(model)
set_params(model, get_params(model))

print("Param setting and getting successful")

print("Testing pipeline prediction")

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

for _, (x, y) in enumerate(test_loader):
    forecast = pipeline.predict(
        context=x,
        prediction_length=pred_len,
        num_samples=20,
    )

    forecast = np.quantile(forecast.numpy(), 0.5, axis=1)

    mse = mean_squared_error(y, forecast)

    print("MSE:", mse)
