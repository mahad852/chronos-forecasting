from chronos import ChronosPipeline
from scripts.training.train import train_vital_signs
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient

def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]



model = train_vital_signs(training_data_paths=["vital_signs_arrow/client01.arrow"], model_id="amazon/chronos-t5-tiny", context_length=600, prediction_length=60, output_dir="weights/client01/", max_steps=5000)

print(get_params(model))

set_params(model, get_params(model))

print("Param setting and getting successful")