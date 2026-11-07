import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.centralized import run_centralized, save_results


# Test that centralized baseline produces valid metrics, correct CSV format,
# and verifies no data leakage by confirming imputer and scaler were fit only on pooled train data.
def test_centralized_baseline():
    csv_path = Path("results/centralized.csv")

    # Run centralized baseline and retrieve results along with fitted artifacts
    results, artifacts = run_centralized(seed=42, return_artifacts=True)

    # Ensure CSV file is saved
    save_results(results, str(csv_path))

    # 1. Assert that the CSV exists and has 5 rows (4 hospitals + ALL)
    assert csv_path.exists(), f"Expected results CSV at {csv_path}"
    df = pd.read_csv(csv_path)

    assert len(df) == 5, f"Expected exactly 5 rows in CSV, got {len(df)}"

    expected_hospitals = ["Cleveland", "Hungary", "Switzerland", "VA", "ALL"]
    assert list(df["hospital"]) == expected_hospitals, (
        f"Expected hospitals {expected_hospitals}, got {list(df['hospital'])}"
    )

    expected_columns = ["model", "hospital", "accuracy", "precision", "recall", "roc_auc", "n_test"]
    assert list(df.columns) == expected_columns, (
        f"Expected CSV columns {expected_columns}, got {list(df.columns)}"
    )
    assert (df["model"] == "centralized").all(), "Expected all rows to have model='centralized'"

    # 2. Assert that all metric values are between 0 and 1 (roc_auc may be NaN)
    for idx, row in df.iterrows():
        hospital = row["hospital"]
        for metric in ["accuracy", "precision", "recall"]:
            val = row[metric]
            assert 0.0 <= val <= 1.0, f"{hospital} {metric} = {val} is not between 0 and 1"

        roc_auc_val = row["roc_auc"]
        assert np.isnan(roc_auc_val) or (0.0 <= roc_auc_val <= 1.0), (
            f"{hospital} roc_auc = {roc_auc_val} is not between 0 and 1 and not NaN"
        )

        assert row["n_test"] > 0, f"{hospital} n_test must be positive"

    # Also verify that calling run_centralized() directly returns the 5 dicts with identical values
    direct_results = run_centralized(seed=42)
    assert len(direct_results) == 5, f"Expected 5 result dicts, got {len(direct_results)}"
    for d, row in zip(direct_results, df.to_dict(orient="records")):
        assert d["hospital"] == row["hospital"]
        assert np.isclose(d["accuracy"], row["accuracy"])
        assert np.isclose(d["precision"], row["precision"])
        assert np.isclose(d["recall"], row["recall"])
        if np.isnan(d["roc_auc"]):
            assert np.isnan(row["roc_auc"])
        else:
            assert np.isclose(d["roc_auc"], row["roc_auc"])
        assert d["n_test"] == row["n_test"]

    # 3. Assert imputer and scaler were fit only on pooled TRAIN:
    # Check that scaler's mean_ equals the mean of the imputed pooled train features,
    # not the mean of train+test.
    scaler = artifacts["scaler"]
    imputer = artifacts["imputer"]
    X_train_imp = artifacts["X_train_imp"]
    X_train_pooled = artifacts["X_train_pooled"]
    splits = artifacts["splits"]

    # Calculate imputed pooled train mean
    train_mean = X_train_imp.mean(axis=0)

    # Scaler mean must match the imputed pooled train mean
    assert np.allclose(scaler.mean_, train_mean), (
        f"Scaler mean_ does not equal imputed pooled train mean! Max diff: {np.max(np.abs(scaler.mean_ - train_mean))}"
    )

    # Calculate train+test imputed features and their mean
    hospitals = list(splits.keys())
    X_test_pooled = pd.concat([splits[h][1] for h in hospitals], axis=0, ignore_index=True)
    X_test_imp = imputer.transform(X_test_pooled)
    X_all_imp = np.vstack([X_train_imp, X_test_imp])
    all_mean = X_all_imp.mean(axis=0)

    # Scaler mean must NOT match train+test mean (guards against data leakage)
    assert not np.allclose(scaler.mean_, all_mean), (
        "Scaler mean_ equals train+test mean: data leakage detected!"
    )

    # Imputer statistics must match pooled train medians
    train_medians = X_train_pooled.median(axis=0).values
    assert np.allclose(imputer.statistics_, train_medians), (
        "Imputer statistics_ does not equal pooled train medians"
    )

    print("All assertions in test_centralized.py passed successfully!")
    print(f"Verified 5 rows in {csv_path}, metrics within bounds [0, 1], and no data leakage.")


if __name__ == "__main__":
    test_centralized_baseline()
