from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter


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

def get_ecg_mit_dataset(dpath = "/home/mali2/datasets/ecg/MIT-BIH.npz"):
    f = np.load(dpath)
    data = np.zeros((f[f.files[0]].shape[0], len(f.files)))

    for i, file in enumerate(f.files):
        data[:, i] = f[file]
    
    train_b1, train_b2 = 0, int(data.shape[0] * 0.30)
    
    return data[train_b1: train_b2, :]


if __name__ == "__main__":
    time_series = []
    data = get_ecg_mit_dataset() #"/Users/ma649596/Downloads/MIT-BIH.npz")
    
    for i in range(data.shape[1]):
        time_series.append(data[:, i])

    # Convert to GluonTS arrow format
    convert_to_arrow("./ecg_mit.arrow", time_series=time_series)
