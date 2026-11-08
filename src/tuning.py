import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_clean
from src.model import make_model
from src.preprocess import apply_imputer, apply_scaler, fit_imputer, fit_scaler
from src.split import split_all
from src.train import predict_scores, train_model


# Select the optimal hyperparameter pair based on validation ROC-AUC,
# preferring fewer epochs and larger weight decay when performance is within 0.005.
def choose_best_hyperparams(results):
    max_auc = max(r["mean_val_auc"] for r in results)
    candidates = [
        r for r in results
        if (max_auc - r["mean_val_auc"]) <= 0.005 + 1e-9
    ]
    candidates.sort(key=lambda r: (r["epochs"], -r["weight_decay"], -r["mean_val_auc"]))
    return int(candidates[0]["epochs"]), float(candidates[0]["weight_decay"])


# Run 5-fold cross-validation on pooled training data only to select optimal MLP hyperparameters.
def select_hyperparams(seed=42, return_artifacts=False):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    # Pool training sets only (splits[h][0] = X_train, splits[h][2] = y_train)
    X_train_pooled = pd.concat([splits[h][0] for h in hospitals], axis=0, ignore_index=True)
    y_train_pooled = pd.concat([splits[h][2] for h in hospitals], axis=0, ignore_index=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train_pooled, y_train_pooled))

    epochs_list = [10, 20, 30, 50, 100]
    wd_list = [0.0, 0.001, 0.01]

    # Precompute fold preprocessors and scaled features on fold training part only
    fold_data = []
    fold_imputers = []
    fold_scalers = []
    fold_train_dfs = []

    for train_idx, val_idx in folds:
        fold_X_tr = X_train_pooled.iloc[train_idx]
        fold_y_tr = y_train_pooled.iloc[train_idx]
        fold_X_val = X_train_pooled.iloc[val_idx]
        fold_y_val = y_train_pooled.iloc[val_idx]

        imputer = fit_imputer(fold_X_tr)
        fold_X_tr_imp = apply_imputer(imputer, fold_X_tr)
        fold_X_val_imp = apply_imputer(imputer, fold_X_val)

        scaler = fit_scaler(fold_X_tr_imp)
        fold_X_tr_scaled = apply_scaler(scaler, fold_X_tr_imp)
        fold_X_val_scaled = apply_scaler(scaler, fold_X_val_imp)

        fold_imputers.append(imputer)
        fold_scalers.append(scaler)
        fold_train_dfs.append(fold_X_tr)

        fold_data.append((fold_X_tr_scaled, fold_y_tr, fold_X_val_scaled, fold_y_val))

    results = []
    for epochs in epochs_list:
        for wd in wd_list:
            fold_val_aucs = []
            for fold_X_tr_scaled, fold_y_tr, fold_X_val_scaled, fold_y_val in fold_data:
                model = make_model(seed=seed)
                train_model(
                    model,
                    fold_X_tr_scaled,
                    fold_y_tr,
                    epochs=epochs,
                    lr=0.01,
                    batch_size=32,
                    seed=seed,
                    weight_decay=wd,
                )
                val_scores = predict_scores(model, fold_X_val_scaled)
                auc = float(roc_auc_score(fold_y_val, val_scores))
                fold_val_aucs.append(auc)

            mean_auc = float(np.mean(fold_val_aucs))
            std_auc = float(np.std(fold_val_aucs))
            results.append({
                "epochs": epochs,
                "weight_decay": wd,
                "mean_val_auc": mean_auc,
                "std_val_auc": std_auc,
            })

    grid_df = pd.DataFrame(results)

    # Save full grid to results/tuning_grid.csv
    results_path = Path("results/tuning_grid.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(results_path, index=False)

    best_epochs, best_wd = choose_best_hyperparams(results)

    # Print the full grid as a table and the chosen pair
    print("=" * 65)
    print("Hyperparameter Tuning Grid (5-Fold CV on Pooled Train Data):")
    print("=" * 65)
    print(grid_df.to_string(index=False))
    print("-" * 65)
    print(f"Chosen Hyperparameters: epochs={best_epochs}, weight_decay={best_wd}")
    print("=" * 65)

    artifacts = {
        "fold_imputers": fold_imputers,
        "fold_scalers": fold_scalers,
        "fold_train_dfs": fold_train_dfs,
        "X_train_pooled": X_train_pooled,
        "y_train_pooled": y_train_pooled,
        "grid_df": grid_df,
        "chosen": (best_epochs, best_wd),
    }
    select_hyperparams.artifacts = artifacts

    if return_artifacts:
        return (best_epochs, best_wd), artifacts
    return best_epochs, best_wd


if __name__ == "__main__":
    select_hyperparams(seed=42)
