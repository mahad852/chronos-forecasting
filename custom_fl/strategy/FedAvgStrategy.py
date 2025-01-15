from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar, parameters_to_ndarrays

from typing import Callable, Optional, List, Tuple, Union, Dict

import numpy as np

import os


class FedAvgStrategy(FedAvg):
    def __init__(self, *, fraction_fit: float = 1, fraction_evaluate: float = 1, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, evaluate_fn: Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None] | None = None, on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, accept_failures: bool = True, initial_parameters: Parameters | None = None, fit_metrics_aggregation_fn: MetricsAggregationFn | None = None, evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None, inplace: bool = True, log_path: str = "logs/", event_file: str = "logs/events.txt") -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn, on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn, accept_failures=accept_failures, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.log_path = log_path
        self.event_file = event_file
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        with open(self.event_file, "a") as f:
            f.write("STARTING aggregation of weights.\n")

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(os.path.join(self.log_path, f"round-{server_round}-weights.npz"), *aggregated_ndarrays)

        with open(self.event_file, "a") as f:
            f.write("COMPLETED aggregation of weights.\n")

        return aggregated_parameters, aggregated_metrics
