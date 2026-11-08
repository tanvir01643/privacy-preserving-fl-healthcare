import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.experiments import run_isolated_mlp
from src.model import HeartMLP, get_weights, make_model, set_weights
from src.preprocess import apply_imputer
from src.train import predict_scores, train_model


# Verify the model architecture has exactly 289 trainable parameters
# and outputs raw logits of shape (n, 1) for input features of shape (n, 8).
def test_model_architecture_and_output_shape():
    model = make_model(42)
    assert isinstance(model, HeartMLP), "make_model did not return a HeartMLP instance"

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 289, f"Expected exactly 289 parameters, got {total_params}"

    for n in [1, 5, 20]:
        x = torch.randn(n, 8)
        out = model(x)
        assert out.shape == (n, 1), f"Expected output shape ({n}, 1), got {out.shape}"


# Verify that extracting weights from a model and setting them into a new model
# results in identical prediction outputs.
def test_get_and_set_weights_identical_predictions():
    model1 = make_model(123)
    weights = get_weights(model1)

    model2 = make_model(999)  # Different initial weights
    set_weights(model2, weights)

    x_np = np.random.RandomState(42).randn(10, 8).astype(np.float32)
    x_tensor = torch.tensor(x_np)

    model1.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model1(x_tensor)
        out2 = model2(x_tensor)

    assert torch.allclose(out1, out2), "Model outputs differ after set_weights"

    scores1 = predict_scores(model1, x_np)
    scores2 = predict_scores(model2, x_np)
    assert np.allclose(scores1, scores2), "predict_scores differ after set_weights"


# Verify that calling make_model with the same seed twice produces identical initial weights.
def test_make_model_seed_reproducibility():
    model_a = make_model(42)
    model_b = make_model(42)

    weights_a = get_weights(model_a)
    weights_b = get_weights(model_b)

    assert len(weights_a) == len(weights_b), "Weights lists have different lengths"
    for i, (wa, wb) in enumerate(zip(weights_a, weights_b)):
        assert np.array_equal(wa, wb), f"Weight tensor {i} differs between two make_model(42) calls"


# Verify that training the model on a small toy dataset decreases the training loss
# from the first epoch to the final epoch.
def test_training_decreases_loss():
    rng = np.random.RandomState(42)
    X_toy = rng.randn(40, 8).astype(np.float32)
    y_toy = ((X_toy[:, 0] + X_toy[:, 1] + X_toy[:, 2]) > 0).astype(np.float32)

    model = make_model(42)
    losses = train_model(model, X_toy, y_toy, epochs=50, lr=0.05, batch_size=10, seed=42)

    assert len(losses) == 50, f"Expected 50 epoch losses, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"Expected last epoch loss ({losses[-1]:.4f}) to be lower than first epoch loss ({losses[0]:.4f})"
    )


# Verify that in run_isolated_mlp each hospital's scaler is fitted only on that hospital's
# own training data without cross-hospital or test-set leakage.
def test_isolated_mlp_scalers_no_leakage():
    results, artifacts = run_isolated_mlp(seed=42, return_artifacts=True)
    splits = artifacts["splits"]
    hospitals = list(splits.keys())

    assert len(hospitals) == 4, f"Expected 4 hospitals, got {len(hospitals)}"

    for h in hospitals:
        X_train = splits[h][0]
        imputer = artifacts["imputers"][h]
        scaler = artifacts["scalers"][h]

        X_train_imp = apply_imputer(imputer, X_train)
        expected_train_mean = X_train_imp.mean(axis=0).values

        # 1. scaler.mean_ must match that hospital's own imputed train mean
        assert np.allclose(scaler.mean_, expected_train_mean), (
            f"Hospital {h}: scaler.mean_ does not match its imputed train mean"
        )

        # 2. scaler.mean_ must not match other hospitals' train means (no cross-contamination)
        for other in hospitals:
            if other == h:
                continue
            other_X_train = splits[other][0]
            other_imp = apply_imputer(artifacts["imputers"][other], other_X_train)
            other_mean = other_imp.mean(axis=0).values
            assert not np.allclose(scaler.mean_, other_mean), (
                f"Hospital {h}: scaler.mean_ suspiciously matches hospital {other}'s train mean"
            )


if __name__ == "__main__":
    print("Running test_model_architecture_and_output_shape...")
    test_model_architecture_and_output_shape()
    print("  [PASS]")

    print("Running test_get_and_set_weights_identical_predictions...")
    test_get_and_set_weights_identical_predictions()
    print("  [PASS]")

    print("Running test_make_model_seed_reproducibility...")
    test_make_model_seed_reproducibility()
    print("  [PASS]")

    print("Running test_training_decreases_loss...")
    test_training_decreases_loss()
    print("  [PASS]")

    print("Running test_isolated_mlp_scalers_no_leakage...")
    test_isolated_mlp_scalers_no_leakage()
    print("  [PASS]")

    print("\nAll assertions in test_mlp.py passed successfully!")
