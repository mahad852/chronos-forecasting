from scripts.training.train import load_model

from custom_datasets.PTBDataset import PTBDataset

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

import os

import argparse

from custom_fl.strategy.FedAvgStrategy import FedAvgStrategy
from custom_fl.strategy.ScaffoldStrategy import ScaffoldStrategy

from custom_fl.client.FedAvgClient import get_fedavg_client_fn
from custom_fl.client.ScaffoldClient import get_scaffold_client_fn
from custom_fl.client.LocalClient import get_local_client_fn

from custom_fl.server.ScaffoldServer import ScaffoldServer
from flwr.server import Server
from flwr.server.client_manager import SimpleClientManager

from utils.model import gen_weighted_avergage_fn, get_params, restore_state_dict, set_params
from utils.general import find_round_offset

from convert_vital_signs_to_arrow import create_ptb_dataset
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", help="The strategy to use. pass one of 'scaffold' or 'fedavg'")
parser.add_argument("--log_path", help="The path where weights and event logs would be stored.")
parser.add_argument("--data_path", help="The path where the dataset is stored.")
parser.add_argument("--partition_path", help="The path where the partition json is stored.")
parser.add_argument("--cv_dir", help="directory to save client cvs for scaffold", default="")
parser.add_argument("--num_rounds", help="number of communication rounds", type=int, default=10)

args = parser.parse_args()

if args.strategy not in ["fedavg", "scaffold", "local"]:
    raise NotImplementedError(f"{args.strategy} is not support. Please use the --help flag to see valid strategy options.")

context_len = 512
pred_len = 64
model_path = "amazon/chronos-t5-tiny"

data_path = args.data_path #"/home/mali2/datasets/vital_signs" # "/Users/ma649596/Downloads/vital_signs_data/data"
partition_path = args.partition_path

val_batch_size = 64
val_batches = 50

num_rounds = args.num_rounds

log_path = args.log_path #"logs/fed_avg_hetro2"

round_offset = find_round_offset(log_path)

if not os.path.exists(log_path):
    os.mkdir(log_path)

if round_offset == 0 and args.strategy == "scaffold" and args.cv_dir != "":
    if os.path.exists(args.cv_dir):
        for file in os.listdir(args.cv_dir):
            os.remove(os.path.join(args.cv_dir, file))
        os.removedirs(args.cv_dir)
    os.makedirs(args.cv_dir)

event_file = os.path.join(log_path, "events.txt")

if round_offset == 0:
    open(event_file, "w").close()


with open(partition_path, "r") as f:
    partition_ids = json.load(f)["partition"].keys()

client_ds = []
max_steps_for_clients = []

for i, pid in enumerate(partition_ids):
    train_ds = PTBDataset(
        partition_path=partition_path, 
        ds_path=data_path, 
        is_train=True, 
        pred_len=pred_len, 
        context_len=context_len, 
        partition_id=pid
    )

    test_ds = PTBDataset(
        partition_path=partition_path, 
        ds_path=data_path, 
        is_train=False, 
        pred_len=pred_len, 
        context_len=context_len, 
        partition_id=pid
    )

    # create_ptb_dataset(train_ds, os.path.join("ptb_arrow", f"client0{i + 1}.arrow"))
    client_ds.append(test_ds)

    max_steps_for_clients.append(len(train_ds))

client_fn_getter = None
strategy_class = None
server = None

if args.strategy == "fedavg":
    client_fn_getter = get_fedavg_client_fn
    strategy_class = FedAvgStrategy
elif args.strategy == "scaffold":
    client_fn_getter = get_scaffold_client_fn
    strategy_class = ScaffoldStrategy
elif args.strategy == "local":
    client_fn_getter = get_local_client_fn
    strategy_class = FedAvgStrategy

client_fn = client_fn_getter(client_ds=client_ds,
                             train_root="ptb_arrow",
                             val_batches=val_batches, val_batch_size=val_batch_size, 
                             max_steps_for_clients=max_steps_for_clients, 
                             context_len=context_len, pred_len= pred_len, 
                             log_path=log_path, save_dir=args.cv_dir)

model = load_model(model_id=model_path)

if round_offset > 0:
    npy_model = np.load(os.path.join(log_path, f"round-{round_offset}-weights.npz"))
    npy_params = [npy_model[file] for file in npy_model.files]
    npy_params = restore_state_dict(model, npy_params)
    set_params(model, npy_params)

ndarrays = get_params(model)
global_model_init = ndarrays_to_parameters(ndarrays)

strategy = strategy_class(
    evaluate_metrics_aggregation_fn=gen_weighted_avergage_fn(log_path),  # callback defined earlier
    initial_parameters=global_model_init,  # initialised global model,
    min_fit_clients=len(client_ds),
    min_evaluate_clients=len(client_ds),
    min_available_clients=len(client_ds),
    log_path=log_path,
    event_file=event_file,
    round_offset=round_offset
)

if args.strategy == "scaffold":
    server = ScaffoldServer(client_manager=SimpleClientManager(), strategy=strategy, model=model, model_name=model_path, log_path=log_path)
    # server = Server(client_manager=SimpleClientManager(), strategy=strategy)

# each client gets 1xCPU (this is the default if no resources are specified)
my_client_resources = {'num_cpus': 1, 'num_gpus': 2/len(client_ds)}


start_simulation(
    server=server,
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=len(client_ds), # Total number of clients available
    config=ServerConfig(num_rounds=num_rounds), # Specify number of FL rounds
    strategy=strategy, # A Flower strategy
    ray_init_args = {'num_cpus': 1, 'num_gpus': 2},
    client_resources=my_client_resources
)
