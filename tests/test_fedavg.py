import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from src.fedavg import fedavg
from src.federated import run_federated
from src.model import get_weights, make_model
from src.preprocess import apply_imputer


# Verify that FedAvg equals an exact hand-computed weighted average on a tiny example.
def test_fedavg_hand_computed_weighted_average():
    # Client 0: size 100, weights [array([1.0, 2.0]), array([[3.0], [4.0]])]
    # Client 1: size 300, weights [array([5.0, 6.0]), array([[7.0], [8.0]])]
    # Total size = 400. Client 0 weight = 0.25, Client 1 weight = 0.75.
    # Parameter 0: 0.25 * [1.0, 2.0] + 0.75 * [5.0, 6.0] = [4.0, 5.0]
    # Parameter 1: 0.25 * [[3.0], [4.0]] + 0.75 * [[7.0], [8.0]] = [[6.0], [7.0]]
    w0 = [np.array([1.0, 2.0], dtype=np.float32), np.array([[3.0], [4.0]], dtype=np.float32)]
    w1 = [np.array([5.0, 6.0], dtype=np.float32), np.array([[7.0], [8.0]], dtype=np.float32)]
    sizes = [100, 300]

    aggregated = fedavg([w0, w1], sizes)

    assert np.allclose(aggregated[0], np.array([4.0, 5.0], dtype=np.float32)), (
        f"Expected [4.0, 5.0], got {aggregated[0]}"
    )
    assert np.allclose(aggregated[1], np.array([[6.0], [7.0]], dtype=np.float32)), (
        f"Expected [[6.0], [7.0]], got {aggregated[1]}"
    )


# Verify that when client dataset sizes are equal, FedAvg matches the arithmetic mean.
def test_fedavg_equal_sizes_equals_plain_mean():
    rng = np.random.RandomState(42)
    w0 = [rng.randn(4, 5).astype(np.float32), rng.randn(3).astype(np.float32)]
    w1 = [rng.randn(4, 5).astype(np.float32), rng.randn(3).astype(np.float32)]
    w2 = [rng.randn(4, 5).astype(np.float32), rng.randn(3).astype(np.float32)]
    sizes = [150, 150, 150]

    aggregated = fedavg([w0, w1, w2], sizes)

    expected_0 = (w0[0] + w1[0] + w2[0]) / 3.0
    expected_1 = (w0[1] + w1[1] + w2[1]) / 3.0

    assert np.allclose(aggregated[0], expected_0), "FedAvg does not match arithmetic mean for equal sizes"
    assert np.allclose(aggregated[1], expected_1), "FedAvg does not match arithmetic mean for equal sizes"


# Verify that aggregating identical weights across clients returns identical weights regardless of sizes.
def test_fedavg_identical_weights_preservation():
    model = make_model(seed=42)
    weights = get_weights(model)

    weight_lists = [
        [w.copy() for w in weights],
        [w.copy() for w in weights],
        [w.copy() for w in weights],
    ]
    sizes = [242, 98, 160]

    aggregated = fedavg(weight_lists, sizes)

    for orig, res in zip(weights, aggregated):
        assert np.allclose(orig, res), "FedAvg failed to preserve identical weights"


# Verify that FedAvg preserves the shapes and data types of all parameter arrays in the model.
def test_fedavg_keeps_array_shapes():
    model1 = make_model(seed=1)
    model2 = make_model(seed=2)
    model3 = make_model(seed=3)
    model4 = make_model(seed=4)

    w1 = get_weights(model1)
    w2 = get_weights(model2)
    w3 = get_weights(model3)
    w4 = get_weights(model4)
    sizes = [242, 235, 98, 160]

    aggregated = fedavg([w1, w2, w3, w4], sizes)

    assert len(aggregated) == len(w1), f"Expected {len(w1)} weight tensors, got {len(aggregated)}"
    for idx, (orig, agg) in enumerate(zip(w1, aggregated)):
        assert agg.shape == orig.shape, f"Shape mismatch at tensor {idx}: {agg.shape} vs {orig.shape}"
        assert agg.dtype == orig.dtype, f"Dtype mismatch at tensor {idx}: {agg.dtype} vs {orig.dtype}"


# Verify that run_federated executes correctly, fits preprocessors on train only, and produces valid results.
def test_federated_preprocessors_no_leakage():
    results, artifacts = run_federated(seed=42, epochs_total=2, weight_decay=0.01, local_epochs=2, return_artifacts=True)
    splits = artifacts["splits"]
    hospitals = list(splits.keys())

    assert len(results) == 5, f"Expected 5 result rows, got {len(results)}"
    for h in hospitals:
        imputer = artifacts["imputers"][h]
        scaler = artifacts["scalers"][h]
        X_train = splits[h][0]

        X_train_imp = apply_imputer(imputer, X_train)
        expected_mean = X_train_imp.mean(axis=0).values

        assert np.allclose(scaler.mean_, expected_mean), (
            f"Hospital {h}: scaler mean does not match its own train data mean"
        )


if __name__ == "__main__":
    print("Running test_fedavg_hand_computed_weighted_average...")
    test_fedavg_hand_computed_weighted_average()
    print("  [PASS]")

    print("Running test_fedavg_equal_sizes_equals_plain_mean...")
    test_fedavg_equal_sizes_equals_plain_mean()
    print("  [PASS]")

    print("Running test_fedavg_identical_weights_preservation...")
    test_fedavg_identical_weights_preservation()
    print("  [PASS]")

    print("Running test_fedavg_keeps_array_shapes...")
    test_fedavg_keeps_array_shapes()
    print("  [PASS]")

    print("Running test_federated_preprocessors_no_leakage...")
    test_federated_preprocessors_no_leakage()
    print("  [PASS]")

    print("\nAll assertions in test_fedavg.py passed successfully!")
