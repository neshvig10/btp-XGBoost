import flwr as fl
from flwr.server.strategy import FedXgbBagging

# FL experimental settings
pool_size = 2
num_rounds = 5
num_clients_per_round = 2
num_evaluate_clients = 2

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (RMSE) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    rmse_aggregated = sum([metrics["RMSE"] * num for num, metrics in eval_metrics]) / total_num
    r2_aggregated = sum([metrics["R-squared"] * num for num, metrics in eval_metrics]) / total_num
    
    metrics_aggregated = {"RMSE": rmse_aggregated,"R2" : r2_aggregated}
    return metrics_aggregated

# Define strategy
strategy = FedXgbBagging(
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=num_evaluate_clients,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
)

# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
