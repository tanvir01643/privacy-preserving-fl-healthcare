import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data import load_clean
from src.evaluate import evaluate_model
from src.model import make_model
from src.preprocess import apply_imputer, apply_scaler, fit_imputer, fit_scaler
from src.split import split_all
from src.train import predict_scores, train_model


# Compute the majority-class baseline accuracy for a given label sequence.
def compute_majority_baseline(y):
    counts = pd.Series(y).value_counts()
    return float(counts.max() / len(y)) if len(y) > 0 else 0.0


# Train a single centralized MLP on pooled train sets from all 4 hospitals
# and evaluate on each hospital's test set and the pooled test set ("ALL").
def run_centralized_mlp(seed=42, epochs=100, lr=0.01, batch_size=32, return_artifacts=False):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    # Pool the 4 train sets
    X_train_pooled = pd.concat([splits[h][0] for h in hospitals], axis=0, ignore_index=True)
    y_train_pooled = pd.concat([splits[h][2] for h in hospitals], axis=0, ignore_index=True)

    # Fit ONE imputer and ONE scaler on pooled train only
    imputer = fit_imputer(X_train_pooled)
    X_train_imp = apply_imputer(imputer, X_train_pooled)

    scaler = fit_scaler(X_train_imp)
    X_train_scaled = apply_scaler(scaler, X_train_imp)

    # Train ONE MLP
    model = make_model(seed=seed)
    train_model(
        model,
        X_train_scaled,
        y_train_pooled,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    results = []

    # Evaluate on each hospital's test set
    for h in hospitals:
        X_test = splits[h][1]
        y_test = splits[h][3]

        X_test_imp = apply_imputer(imputer, X_test)
        X_test_scaled = apply_scaler(scaler, X_test_imp)

        scores = predict_scores(model, X_test_scaled)
        preds = (scores >= 0.5).astype(int)

        metrics = evaluate_model(y_test, preds, scores)
        results.append({
            "model": "centralized_mlp",
            "hospital": h,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "n_test": len(y_test),
        })

    # Evaluate on the pooled test set ("ALL")
    X_test_pooled = pd.concat([splits[h][1] for h in hospitals], axis=0, ignore_index=True)
    y_test_pooled = pd.concat([splits[h][3] for h in hospitals], axis=0, ignore_index=True)

    X_test_pooled_imp = apply_imputer(imputer, X_test_pooled)
    X_test_pooled_scaled = apply_scaler(scaler, X_test_pooled_imp)

    scores_pooled = predict_scores(model, X_test_pooled_scaled)
    preds_pooled = (scores_pooled >= 0.5).astype(int)

    metrics_pooled = evaluate_model(y_test_pooled, preds_pooled, scores_pooled)
    results.append({
        "model": "centralized_mlp",
        "hospital": "ALL",
        "accuracy": metrics_pooled["accuracy"],
        "precision": metrics_pooled["precision"],
        "recall": metrics_pooled["recall"],
        "roc_auc": metrics_pooled["roc_auc"],
        "n_test": len(y_test_pooled),
    })

    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "splits": splits,
        "X_train_imp": X_train_imp,
        "X_train_pooled": X_train_pooled,
        "y_train_pooled": y_train_pooled,
    }
    run_centralized_mlp.artifacts = artifacts

    if return_artifacts:
        return results, artifacts
    return results


# Train a separate isolated MLP for each hospital on its own train set only
# and evaluate each on its own test set, with ALL being concatenated predictions.
def run_isolated_mlp(seed=42, epochs=100, lr=0.01, batch_size=32, return_artifacts=False):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    results = []
    artifacts = {
        "splits": splits,
        "scalers": {},
        "imputers": {},
        "models": {},
        "hospitals": {},
    }

    all_y_true = []
    all_preds = []
    all_scores = []

    for h in hospitals:
        X_train, X_test, y_train, y_test = splits[h]

        # Fit own imputer and scaler on its own train only
        imputer = fit_imputer(X_train)
        X_train_imp = apply_imputer(imputer, X_train)

        scaler = fit_scaler(X_train_imp)
        X_train_scaled = apply_scaler(scaler, X_train_imp)

        # Train own new MLP
        model = make_model(seed=seed)
        train_model(
            model,
            X_train_scaled,
            y_train,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

        # Evaluate on its own test set
        X_test_imp = apply_imputer(imputer, X_test)
        X_test_scaled = apply_scaler(scaler, X_test_imp)

        scores = predict_scores(model, X_test_scaled)
        preds = (scores >= 0.5).astype(int)

        metrics = evaluate_model(y_test, preds, scores)
        results.append({
            "model": "isolated_mlp",
            "hospital": h,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "n_test": len(y_test),
        })

        all_y_true.append(np.asarray(y_test))
        all_preds.append(preds)
        all_scores.append(scores)

        hospital_artifact = {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "X_train_imp": X_train_imp,
            "X_train": X_train,
            "y_train": y_train,
        }
        artifacts["scalers"][h] = scaler
        artifacts["imputers"][h] = imputer
        artifacts["models"][h] = model
        artifacts["hospitals"][h] = hospital_artifact
        artifacts[h] = hospital_artifact

    # The "ALL" row = metrics computed on the concatenated predictions
    # of every hospital's own model on its own test set.
    concat_y_true = np.concatenate(all_y_true)
    concat_preds = np.concatenate(all_preds)
    concat_scores = np.concatenate(all_scores)

    metrics_all = evaluate_model(concat_y_true, concat_preds, concat_scores)
    results.append({
        "model": "isolated_mlp",
        "hospital": "ALL",
        "accuracy": metrics_all["accuracy"],
        "precision": metrics_all["precision"],
        "recall": metrics_all["recall"],
        "roc_auc": metrics_all["roc_auc"],
        "n_test": len(concat_y_true),
    })

    run_isolated_mlp.artifacts = artifacts

    if return_artifacts:
        return results, artifacts
    return results


# Save experiment evaluation results to a CSV file.
def save_results(results, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


# Format evaluation results alongside majority-class baselines into a comparison table.
def format_comparison_table(cent_results, iso_results, splits):
    # Calculate majority-class baseline accuracy for each hospital and ALL
    baselines = {}
    for h in splits:
        baselines[h] = compute_majority_baseline(splits[h][3])
    y_test_all = pd.concat([splits[h][3] for h in splits], axis=0, ignore_index=True)
    baselines["ALL"] = compute_majority_baseline(y_test_all)

    rows = []
    combined = list(cent_results) + list(iso_results)
    for r in combined:
        h = r["hospital"]
        maj = baselines.get(h, 0.0)
        roc_str = f"{r['roc_auc']:.4f}" if not np.isnan(r["roc_auc"]) else "NaN"
        rows.append({
            "Hospital": h,
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Majority Baseline": f"{maj:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "ROC AUC": roc_str,
            "n_test": r["n_test"],
        })

    df_table = pd.DataFrame(rows)
    hospital_order = ["Cleveland", "Hungary", "Switzerland", "VA", "ALL"]
    df_table["order"] = df_table["Hospital"].map(lambda x: hospital_order.index(x) if x in hospital_order else 99)
    df_table = df_table.sort_values(by=["order", "Model"]).drop(columns=["order"])
    return df_table


if __name__ == "__main__":
    print("Running Centralized MLP experiment...")
    cent_results, cent_artifacts = run_centralized_mlp(seed=42, return_artifacts=True)
    save_results(cent_results, "results/centralized_mlp.csv")
    print("Saved results to results/centralized_mlp.csv")

    print("\nRunning Isolated MLP experiment...")
    iso_results, iso_artifacts = run_isolated_mlp(seed=42, return_artifacts=True)
    save_results(iso_results, "results/isolated_mlp.csv")
    print("Saved results to results/isolated_mlp.csv")

    print("\nComparison Table (Centralized MLP vs Isolated MLP):")
    table = format_comparison_table(cent_results, iso_results, cent_artifacts["splits"])
    print(table.to_string(index=False))
