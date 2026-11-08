import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.evaluate import evaluate_model
from src.experiments import format_comparison_table, run_centralized_mlp, run_isolated_mlp, save_results
from src.preprocess import apply_imputer, apply_scaler
from src.train import predict_scores
from src.tuning import select_hyperparams


# Compute training Accuracy and ROC-AUC on pooled data ("ALL") for Centralized and Isolated models.
def compute_train_metrics_all(cent_artifacts, iso_artifacts):
    # 1. Centralized MLP train metrics on pooled training data
    cent_model = cent_artifacts["model"]
    cent_imputer = cent_artifacts["imputer"]
    cent_scaler = cent_artifacts["scaler"]
    X_tr_pooled = cent_artifacts["X_train_pooled"]
    y_tr_pooled = cent_artifacts["y_train_pooled"]

    X_tr_scaled = apply_scaler(cent_scaler, apply_imputer(cent_imputer, X_tr_pooled))
    cent_tr_scores = predict_scores(cent_model, X_tr_scaled)
    cent_tr_preds = (cent_tr_scores >= 0.5).astype(int)
    cent_train_metrics = evaluate_model(y_tr_pooled, cent_tr_preds, cent_tr_scores)

    # 2. Isolated MLP train metrics: concatenate predictions of each hospital's model on its own train set
    splits = iso_artifacts["splits"]
    all_y_train = []
    all_preds_train = []
    all_scores_train = []

    for h in splits:
        X_tr = splits[h][0]
        y_tr = splits[h][2]
        h_imputer = iso_artifacts["imputers"][h]
        h_scaler = iso_artifacts["scalers"][h]
        h_model = iso_artifacts["models"][h]

        h_X_tr_scaled = apply_scaler(h_scaler, apply_imputer(h_imputer, X_tr))
        scores = predict_scores(h_model, h_X_tr_scaled)
        preds = (scores >= 0.5).astype(int)

        all_y_train.append(np.asarray(y_tr))
        all_preds_train.append(preds)
        all_scores_train.append(scores)

    concat_y_tr = np.concatenate(all_y_train)
    concat_preds_tr = np.concatenate(all_preds_train)
    concat_scores_tr = np.concatenate(all_scores_train)
    iso_train_metrics = evaluate_model(concat_y_tr, concat_preds_tr, concat_scores_tr)

    return cent_train_metrics, iso_train_metrics


# Run hyperparameter selection, train both models with tuned values, and report evaluation metrics and gaps.
def main():
    seed = 42

    # Step 1: Select optimal hyperparameters on pooled train data only
    print("Selecting optimal MLP hyperparameters via 5-fold cross-validation...")
    chosen_epochs, chosen_wd = select_hyperparams(seed=seed)

    # Step 2: Run Centralized MLP with tuned hyperparameters
    print(f"\nRunning Centralized MLP (epochs={chosen_epochs}, weight_decay={chosen_wd})...")
    cent_results, cent_artifacts = run_centralized_mlp(
        seed=seed,
        epochs=chosen_epochs,
        weight_decay=chosen_wd,
        return_artifacts=True,
    )
    save_results(cent_results, "results/centralized_mlp_tuned.csv")
    print("Saved results to results/centralized_mlp_tuned.csv")

    # Step 3: Run Isolated MLP with tuned hyperparameters
    print(f"\nRunning Isolated MLP (epochs={chosen_epochs}, weight_decay={chosen_wd})...")
    iso_results, iso_artifacts = run_isolated_mlp(
        seed=seed,
        epochs=chosen_epochs,
        weight_decay=chosen_wd,
        return_artifacts=True,
    )
    save_results(iso_results, "results/isolated_mlp_tuned.csv")
    print("Saved results to results/isolated_mlp_tuned.csv")

    # Step 4: Print chosen values and comparison table
    print("\n" + "=" * 80)
    print(f"CHOSEN HYPERPARAMETERS: epochs = {chosen_epochs}, weight_decay = {chosen_wd}")
    print("=" * 80)

    print("\nComparison Table (Centralized MLP vs Isolated MLP with Tuned Hyperparameters):")
    table = format_comparison_table(cent_results, iso_results, cent_artifacts["splits"])
    print(table.to_string(index=False))

    # Step 5: Print Train vs Test metrics and gap for "ALL"
    cent_train_metrics, iso_train_metrics = compute_train_metrics_all(cent_artifacts, iso_artifacts)
    cent_test_all = next(r for r in cent_results if r["hospital"] == "ALL")
    iso_test_all = next(r for r in iso_results if r["hospital"] == "ALL")

    gap_data = [
        {
            "Model": "centralized_mlp",
            "Train Acc": f"{cent_train_metrics['accuracy']:.4f}",
            "Test Acc": f"{cent_test_all['accuracy']:.4f}",
            "Acc Gap": f"{cent_train_metrics['accuracy'] - cent_test_all['accuracy']:+.4f}",
            "Train ROC-AUC": f"{cent_train_metrics['roc_auc']:.4f}",
            "Test ROC-AUC": f"{cent_test_all['roc_auc']:.4f}",
            "AUC Gap": f"{cent_train_metrics['roc_auc'] - cent_test_all['roc_auc']:+.4f}",
        },
        {
            "Model": "isolated_mlp",
            "Train Acc": f"{iso_train_metrics['accuracy']:.4f}",
            "Test Acc": f"{iso_test_all['accuracy']:.4f}",
            "Acc Gap": f"{iso_train_metrics['accuracy'] - iso_test_all['accuracy']:+.4f}",
            "Train ROC-AUC": f"{iso_train_metrics['roc_auc']:.4f}",
            "Test ROC-AUC": f"{iso_test_all['roc_auc']:.4f}",
            "AUC Gap": f"{iso_train_metrics['roc_auc'] - iso_test_all['roc_auc']:+.4f}",
        },
    ]

    print("\n" + "=" * 80)
    print("Train vs Test Gap on Pooled Data ('ALL'):")
    print("-" * 80)
    gap_df = pd.DataFrame(gap_data)
    print(gap_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
