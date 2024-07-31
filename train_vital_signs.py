from scripts.training.train import train_vital_signs
import os

if not os.path.exists("weights/client01"):
    os.makedirs("weights/client01")

train_vital_signs(training_data_paths=["vital_signs_arrow/client01.arrow"], model_id="amazon/chronos-t5-mini", context_length=600, prediction_length=60, output_dir="weights/client01/")