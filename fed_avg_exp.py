from chronos import ChronosPipeline
from scripts.training.train import train_vital_signs
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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


def test(model, val_loader):
    mses = []
    rmses = []
    maes = []

    for _, (x, y) in enumerate(val_loader):
        forecast = model.predict(
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
        self.model = ChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    def fit(self, parameters, config):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""

        # copy parameters sent by the server into client's local model
        set_params(self.model, parameters)

        # do local training (call same function as centralised setting)
        self.model = train_vital_signs(training_data_paths=[self.train_data_path], model=self.model, context_length=context_len, prediction_length=pred_len, max_steps=5000)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model), 5000, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        set_params(self.model, parameters)
        # do local evaluation (call same function as centralised setting)
        loss, accuracy = test(self.model, self.valloader)
        # send statistics back to the server
        return float(loss), len(self.valloader), {"accuracy": accuracy}

pipeline = ChronosPipeline.from_pretrained(
    model_path,
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

model = train_vital_signs(training_data_paths=["vital_signs_arrow/client01.arrow"], model=pipeline.model, context_length=context_len, prediction_length=pred_len, output_dir="weights/client01/", max_steps=10)

print(get_params(model))

set_params(model, get_params(model))

print("Param setting and getting successful")