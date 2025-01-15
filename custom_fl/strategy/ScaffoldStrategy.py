"""FedNova and SCAFFOLD strategies."""

from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    MetricsAggregationFn
)
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple, Union, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

import os

class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""
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
        """Aggregate fit results using weighted average."""

        with open(self.event_file, "a") as f:
            f.write("STARTING aggregation of weights.\n")

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]
        # Zip parameters and num_examples
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        # Aggregate parameters
        parameters_aggregated = aggregate(weights_results)

        print(f"Saving round {server_round} aggregated_ndarrays...")
        np.savez(os.path.join(self.log_path, f"round-{server_round}-weights.npz"), *parameters_aggregated)

        # Zip client_cv_updates and num_examples
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        with open(self.event_file, "a") as f:
            f.write("COMPLETED aggregation of weights.\n")

        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated,
        )