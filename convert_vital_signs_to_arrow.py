from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from datasets.vital_signs_dataset import VitalSignsDataset

import os


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )

def create_vital_signs_dataset(dataset: VitalSignsDataset, file_path: str):
    time_series = []

    for user_data_path in dataset.user_data_paths:
        data = dataset.get_data(user_data_path)
        time_series.append(data)
    
    convert_to_arrow(file_path, time_series=time_series)
        

if __name__ == "__main__":
    if not os.path.exists("./vital_signs_arrow"):
        os.mkdir("./vital_signs_arrow")

    data_path="/home/x-mali3/datasets/vital_signs" #"/home/mali2/datasets/vital_signs", 
    context_len = 600
    pred_len = 60

    clients = [
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

    client_ids = ["client01", "client02", "client03"]

    for i, client in enumerate(clients):
        id = client_ids[i]
        create_vital_signs_dataset(client, os.path.join("vital_signs_arrow", f"{id}.arrow"))