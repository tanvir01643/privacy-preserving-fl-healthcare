import numpy as np


# Compute the federated weighted average of client model weights proportional to client sample sizes.
def fedavg(weight_lists, sizes):
    total_size = sum(sizes)
    if total_size <= 0:
        raise ValueError("Total size of client datasets must be greater than zero.")

    num_params = len(weight_lists[0])
    aggregated = []

    for param_idx in range(num_params):
        weighted_sum = np.zeros_like(weight_lists[0][param_idx], dtype=np.float64)
        for client_idx, client_weights in enumerate(weight_lists):
            weight = sizes[client_idx] / total_size
            weighted_sum += weight * client_weights[param_idx]
        aggregated.append(weighted_sum.astype(weight_lists[0][param_idx].dtype))

    return aggregated
