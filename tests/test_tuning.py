import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.preprocess import apply_imputer
from src.tuning import choose_best_hyperparams, select_hyperparams


# Verify that src/tuning.py does not contain any references to test sets or test split indices.
def test_no_test_leakage_in_code():
    code_path = project_root / "src" / "tuning.py"
    code = code_path.read_text(encoding="utf-8")
    assert "X_test" not in code, "src/tuning.py contains reference to 'X_test'"
    assert "y_test" not in code, "src/tuning.py contains reference to 'y_test'"
    assert "splits[h][1]" not in code, "src/tuning.py contains reference to 'splits[h][1]'"
    assert "splits[h][3]" not in code, "src/tuning.py contains reference to 'splits[h][3]'"


# Verify that selected hyperparameters come from the evaluated CV grid and match selection rule.
def test_hyperparams_selected_from_grid():
    (best_epochs, best_wd), artifacts = select_hyperparams(seed=42, return_artifacts=True)
    grid_df = artifacts["grid_df"]

    # Verify grid structure and non-empty evaluation
    assert len(grid_df) == 15, f"Expected 15 hyperparameter combinations, got {len(grid_df)}"
    assert {"epochs", "weight_decay", "mean_val_auc", "std_val_auc"}.issubset(grid_df.columns)

    # Verify that the chosen hyperparameter combination exists in the grid
    match = grid_df[(grid_df["epochs"] == best_epochs) & (grid_df["weight_decay"] == best_wd)]
    assert len(match) == 1, f"Chosen ({best_epochs}, {best_wd}) not found uniquely in grid"

    # Verify that the saved seed-specific CSV exists and matches grid_df
    grid_csv_path = Path("results/tuning_grid_seed42.csv")
    assert grid_csv_path.exists(), f"Expected saved grid at {grid_csv_path}"
    saved_df = pd.read_csv(grid_csv_path)
    assert len(saved_df) == len(grid_df), "Saved grid CSV row count mismatch"

    # Verify choose_best_hyperparams logic
    expected_epochs, expected_wd = choose_best_hyperparams(grid_df.to_dict("records"))
    assert (best_epochs, best_wd) == (expected_epochs, expected_wd), "Chosen hyperparameters mismatch"
    assert (best_epochs, best_wd) == (10, 0.01), f"Expected (10, 0.01) for seed 42, got ({best_epochs}, {best_wd})"


# Verify that each fold's imputer and scaler statistics are computed exclusively on fold train data.
def test_fold_statistics_from_fold_train_only():
    # If select_hyperparams hasn't run yet, run it
    if not hasattr(select_hyperparams, "artifacts"):
        _, artifacts = select_hyperparams(seed=42, return_artifacts=True)
    else:
        artifacts = select_hyperparams.artifacts

    fold_imputers = artifacts["fold_imputers"]
    fold_scalers = artifacts["fold_scalers"]
    fold_train_dfs = artifacts["fold_train_dfs"]
    X_train_pooled = artifacts["X_train_pooled"]

    assert len(fold_train_dfs) == 5, f"Expected 5 folds, got {len(fold_train_dfs)}"

    for i in range(5):
        fold_X_tr = fold_train_dfs[i]
        imputer = fold_imputers[i]
        scaler = fold_scalers[i]

        # 1. Imputer statistics must match fold train medians exactly
        expected_median = fold_X_tr.median().values
        assert np.allclose(imputer.statistics_, expected_median), (
            f"Fold {i}: imputer statistics do not match fold train medians"
        )

        # 2. Scaler statistics must match imputed fold train mean and variance
        fold_X_tr_imp = apply_imputer(imputer, fold_X_tr)
        expected_mean = fold_X_tr_imp.mean(axis=0).values
        expected_var = fold_X_tr_imp.var(axis=0, ddof=0).values

        assert np.allclose(scaler.mean_, expected_mean), (
            f"Fold {i}: scaler.mean_ does not match fold train imputed mean"
        )
        assert np.allclose(scaler.var_, expected_var), (
            f"Fold {i}: scaler.var_ does not match fold train imputed variance"
        )

        # 3. Scaler mean should not match the overall pooled train median/mean
        pooled_mean = X_train_pooled.mean(axis=0).values
        assert not np.allclose(scaler.mean_, pooled_mean), (
            f"Fold {i}: scaler.mean_ suspiciously identical to pooled train mean"
        )


if __name__ == "__main__":
    print("Running test_no_test_leakage_in_code...")
    test_no_test_leakage_in_code()
    print("  [PASS]")

    print("Running test_hyperparams_selected_from_grid...")
    test_hyperparams_selected_from_grid()
    print("  [PASS]")

    print("Running test_fold_statistics_from_fold_train_only...")
    test_fold_statistics_from_fold_train_only()
    print("  [PASS]")

    print("\nAll assertions in test_tuning.py passed successfully!")
