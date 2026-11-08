import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import load_clean
from src.evaluate import evaluate_model
from src.model import make_model
from src.preprocess import apply_imputer, apply_scaler, fit_imputer, fit_scaler
from src.split import split_all
from src.train import predict_scores, train_model


# Load hospital datasets, split with fixed seed, and fit pooled imputer and scaler on train rows only.
def prepare_centralized_data(seed=42):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    X_train_pooled = pd.concat([splits[h][0] for h in hospitals], axis=0, ignore_index=True)
    y_train_pooled = pd.concat([splits[h][2] for h in hospitals], axis=0, ignore_index=True)

    imputer = fit_imputer(X_train_pooled)
    X_train_imp = apply_imputer(imputer, X_train_pooled)

    scaler = fit_scaler(X_train_imp)
    X_train_scaled = apply_scaler(scaler, X_train_imp)

    return hospitals, splits, imputer, scaler, X_train_scaled, y_train_pooled


# Evaluate train and test accuracy and ROC-AUC for each hospital and pooled ALL.
def evaluate_train_test_metrics(predict_fn, hospitals, splits, imputer, scaler, X_train_scaled, y_train_pooled):
    rows = []
    names = list(hospitals.keys()) + ["ALL"]

    for name in names:
        if name == "ALL":
            X_tr = X_train_scaled
            y_tr = y_train_pooled

            X_te_raw = pd.concat([splits[h][1] for h in hospitals], axis=0, ignore_index=True)
            y_te = pd.concat([splits[h][3] for h in hospitals], axis=0, ignore_index=True)
            X_te = apply_scaler(scaler, apply_imputer(imputer, X_te_raw))
        else:
            X_tr_raw = splits[name][0]
            y_tr = splits[name][2]
            X_tr = apply_scaler(scaler, apply_imputer(imputer, X_tr_raw))

            X_te_raw = splits[name][1]
            y_te = splits[name][3]
            X_te = apply_scaler(scaler, apply_imputer(imputer, X_te_raw))

        tr_scores = predict_fn(X_tr)
        tr_preds = (tr_scores >= 0.5).astype(int)
        tr_metrics = evaluate_model(y_tr, tr_preds, tr_scores)

        te_scores = predict_fn(X_te)
        te_preds = (te_scores >= 0.5).astype(int)
        te_metrics = evaluate_model(y_te, te_preds, te_scores)

        rows.append({
            "hospital": name,
            "train_acc": tr_metrics["accuracy"],
            "train_auc": tr_metrics["roc_auc"],
            "test_acc": te_metrics["accuracy"],
            "test_auc": te_metrics["roc_auc"],
        })

    return rows


# Compute predicted probability statistics on test rows for each hospital and pooled ALL.
def evaluate_test_probabilities(predict_fn, hospitals, splits, imputer, scaler):
    rows = []
    names = list(hospitals.keys()) + ["ALL"]

    for name in names:
        if name == "ALL":
            X_te_raw = pd.concat([splits[h][1] for h in hospitals], axis=0, ignore_index=True)
        else:
            X_te_raw = splits[name][1]

        X_te = apply_scaler(scaler, apply_imputer(imputer, X_te_raw))
        scores = np.asarray(predict_fn(X_te), dtype=float)

        rows.append({
            "hospital": name,
            "frac_above_05": float(np.mean(scores > 0.5)),
            "min_prob": float(np.min(scores)),
            "mean_prob": float(np.mean(scores)),
            "max_prob": float(np.max(scores)),
        })

    return rows


# Print a formatted table of train and test accuracy and ROC-AUC metrics.
def print_performance_table(title, rows):
    print(f"\n{title}")
    header = f"{'Hospital':<15} {'Train Acc':<12} {'Train AUC':<12} {'Test Acc':<12} {'Test AUC':<12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['hospital']:<15} "
            f"{r['train_acc']:<12.4f} "
            f"{r['train_auc']:<12.4f} "
            f"{r['test_acc']:<12.4f} "
            f"{r['test_auc']:<12.4f}"
        )


# Print a formatted table of test predicted probability distribution statistics.
def print_probability_table(title, rows):
    print(f"\n{title}")
    header = f"{'Hospital':<15} {'Frac > 0.5':<12} {'Min Prob':<12} {'Mean Prob':<12} {'Max Prob':<12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['hospital']:<15} "
            f"{r['frac_above_05']:<12.4f} "
            f"{r['min_prob']:<12.4f} "
            f"{r['mean_prob']:<12.4f} "
            f"{r['max_prob']:<12.4f}"
        )


# Run diagnostic evaluation for Centralized MLP and baseline Logistic Regression.
def main():
    seed = 42
    hospitals, splits, imputer, scaler, X_train_scaled, y_train_pooled = prepare_centralized_data(seed=seed)

    # 1. Train Centralized MLP
    mlp_model = make_model(seed=seed)
    losses = train_model(
        mlp_model,
        X_train_scaled,
        y_train_pooled,
        epochs=100,
        lr=0.01,
        batch_size=32,
        seed=seed,
    )

    print("1. Centralized MLP Training Loss:")
    target_epochs = [1, 10, 25, 50, 75, 100]
    for ep in target_epochs:
        print(f"   Epoch {ep:3d}: {losses[ep - 1]:.4f}")

    # 2. Centralized MLP Train vs Test Metrics
    mlp_predict = lambda X: predict_scores(mlp_model, X)
    mlp_perf_rows = evaluate_train_test_metrics(
        mlp_predict, hospitals, splits, imputer, scaler, X_train_scaled, y_train_pooled
    )
    print_performance_table("2. Centralized MLP: Train vs Test Accuracy and ROC-AUC", mlp_perf_rows)

    # 3. Centralized MLP Test Predicted Probabilities
    mlp_prob_rows = evaluate_test_probabilities(mlp_predict, hospitals, splits, imputer, scaler)
    print_probability_table("3. Centralized MLP: Test Predicted Probability Distribution", mlp_prob_rows)

    # 4. Baseline Logistic Regression for comparison
    lr_model = LogisticRegression(max_iter=1000, random_state=seed)
    lr_model.fit(X_train_scaled, y_train_pooled)

    lr_predict = lambda X: lr_model.predict_proba(X)[:, 1]
    lr_perf_rows = evaluate_train_test_metrics(
        lr_predict, hospitals, splits, imputer, scaler, X_train_scaled, y_train_pooled
    )
    print_performance_table("4a. Logistic Regression (Comparison): Train vs Test Accuracy and ROC-AUC", lr_perf_rows)

    lr_prob_rows = evaluate_test_probabilities(lr_predict, hospitals, splits, imputer, scaler)
    print_probability_table("4b. Logistic Regression (Comparison): Test Predicted Probability Distribution", lr_prob_rows)


if __name__ == "__main__":
    main()
