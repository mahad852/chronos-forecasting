import numpy as np
from chronos import ChronosPipeline
import os
import torch

dataset_path = "/home/mali2/datasets/ecg/MIT-BIH_lagllama_384_64_forecast.npz"

sample_key = "a103-48_384_64_"
sample_true_key = f"{sample_key}true"

context_len = 384
pred_len = 64

def get_context(ds, true_key):
    return ds[true_key][:context_len]

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

dataset = np.load(dataset_path)
context = np.array([get_context(dataset, sample_true_key)])


forecast = pipeline.predict(
    context=torch.tensor(np.array(context)),
    prediction_length=pred_len,
    num_samples=20,
)
forecast = np.quantile(forecast.numpy(), 0.5, axis=1)[0]

if not os.path.exists("forecasts"):
    os.mkdir("forecasts")

np.save("forecasts/a103-48_384_64_chronos", forecast)