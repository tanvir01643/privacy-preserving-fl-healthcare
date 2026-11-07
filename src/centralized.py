import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import load_clean
from src.evaluate import evaluate_model
from src.split import split_all


# Compute the majority-class baseline accuracy for a given label series.
# This represents the benchmark accuracy achieved by always predicting the most frequent class.
def compute_majority_baseline(y):
    counts = pd.Series(y).value_counts()
    return float(counts.max() / len(y)) if len(y) > 0 else 0.0


# Train a centralized Logistic Regression model on pooled train data from all 4 hospitals
# and evaluate performance on each hospital's test set and the pooled test set.
def run_centralized(seed=42, return_artifacts=False):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    # a) Concatenate the 4 hospitals' X_train and y_train into one pooled train set
    X_train_pooled = pd.concat([splits[h][0] for h in hospitals], axis=0, ignore_index=True)
    y_train_pooled = pd.concat([splits[h][2] for h in hospitals], axis=0, ignore_index=True)

    # b) Fit ONE median imputer on pooled train only, then ONE StandardScaler on imputed pooled train only
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_pooled)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    # c) Train LogisticRegression on pooled train
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train_pooled)

    # Store fitted artifacts for inspection and testing
    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "X_train_pooled": X_train_pooled,
        "y_train_pooled": y_train_pooled,
        "X_train_imp": X_train_imp,
        "splits": splits,
    }
    run_centralized.artifacts = artifacts

    # d) Evaluate the SAME model on each hospital's own test set separately (transform only)
    results = []
    for name in hospitals:
        X_test = splits[name][1]
        y_test = splits[name][3]

        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)

        y_pred = model.predict(X_test_scaled)
        y_score = model.predict_proba(X_test_scaled)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_score)
        results.append({
            "model": "centralized",
            "hospital": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "n_test": len(y_test),
        })

    # Evaluate on the pooled test set as well
    X_test_pooled = pd.concat([splits[h][1] for h in hospitals], axis=0, ignore_index=True)
    y_test_pooled = pd.concat([splits[h][3] for h in hospitals], axis=0, ignore_index=True)

    X_test_pooled_imp = imputer.transform(X_test_pooled)
    X_test_pooled_scaled = scaler.transform(X_test_pooled_imp)

    y_pred_pooled = model.predict(X_test_pooled_scaled)
    y_score_pooled = model.predict_proba(X_test_pooled_scaled)[:, 1]

    metrics_pooled = evaluate_model(y_test_pooled, y_pred_pooled, y_score_pooled)
    results.append({
        "model": "centralized",
        "hospital": "ALL",
        "accuracy": metrics_pooled["accuracy"],
        "precision": metrics_pooled["precision"],
        "recall": metrics_pooled["recall"],
        "roc_auc": metrics_pooled["roc_auc"],
        "n_test": len(y_test_pooled),
    })

    if return_artifacts:
        return results, artifacts
    return results


# Save benchmark results to a CSV file to persist experimental outputs.
def save_results(results, output_path="results/centralized.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


# Format evaluation results alongside majority-class baselines into a clear table for display.
def format_results_table(results, splits):
    table_rows = []
    for r in results:
        h = r["hospital"]
        if h == "ALL":
            y_test = pd.concat([splits[name][3] for name in splits], axis=0, ignore_index=True)
        else:
            y_test = splits[h][3]
        maj_acc = compute_majority_baseline(y_test)
        table_rows.append({
            "Hospital": r["hospital"],
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Majority Baseline": f"{maj_acc:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "ROC AUC": f"{r['roc_auc']:.4f}" if not np.isnan(r["roc_auc"]) else "NaN",
            "n_test": r["n_test"],
        })
    return pd.DataFrame(table_rows)


if __name__ == "__main__":
    results, artifacts = run_centralized(seed=42, return_artifacts=True)
    save_results(results, "results/centralized.csv")
    table_df = format_results_table(results, artifacts["splits"])
    print("\nCentralized Logistic Regression Baseline Results:")
    print(table_df.to_string(index=False))
    print(f"\nSaved results to results/centralized.csv")
