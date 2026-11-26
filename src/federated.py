import math
import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import load_clean
from src.evaluate import evaluate_model
from src.fedavg import fedavg
from src.model import get_weights, make_model, set_weights
from src.preprocess import apply_imputer, apply_scaler, fit_imputer, fit_scaler
from src.split import split_all
from src.train import predict_scores, train_model
from src.tuning import select_hyperparams


# Run plain Federated Averaging (FedAvg) across 4 hospital clients and evaluate final performance.
def run_federated(seed=42, epochs_total=10, weight_decay=0.01, local_epochs=2, lr=0.01, batch_size=32, return_artifacts=False):
    hospitals = load_clean()
    splits = split_all(hospitals, seed=seed)

    # 1. Each client fits its own imputer and scaler on its own train only (client data never leaves the client)
    client_data = {}
    imputers = {}
    scalers = {}

    for h in hospitals:
        X_train, X_test, y_train, y_test = splits[h]

        imputer = fit_imputer(X_train)
        X_train_imp = apply_imputer(imputer, X_train)
        X_test_imp = apply_imputer(imputer, X_test)

        scaler = fit_scaler(X_train_imp)
        X_train_scaled = apply_scaler(scaler, X_train_imp)
        X_test_scaled = apply_scaler(scaler, X_test_imp)

        imputers[h] = imputer
        scalers[h] = scaler
        client_data[h] = {
            "X_train_scaled": X_train_scaled,
            "y_train": y_train,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    # 2. Initialize the global model
    global_model = make_model(seed=seed)
    global_weights = get_weights(global_model)

    # 3. Federated training rounds
    rounds = math.ceil(epochs_total / local_epochs)
    round_logs = []

    for r in range(1, rounds + 1):
        client_weights_list = []
        client_sizes = []
        client_train_losses = []

        # Local training on each client
        for client_idx, h in enumerate(hospitals):
            # Client creates local model, loads global weights
            local_model = make_model(seed=seed)
            set_weights(local_model, global_weights)

            # Deterministic seed per round and client
            client_seed = seed + r * 1000 + client_idx

            losses = train_model(
                local_model,
                client_data[h]["X_train_scaled"],
                client_data[h]["y_train"],
                epochs=local_epochs,
                lr=lr,
                batch_size=batch_size,
                seed=client_seed,
                weight_decay=weight_decay,
            )

            # Final epoch loss for this client's local training round
            client_loss = float(losses[-1]) if losses else 0.0
            client_train_losses.append(client_loss)
            client_weights_list.append(get_weights(local_model))
            client_sizes.append(client_data[h]["n_train"])

        # Server aggregates client weights via FedAvg
        global_weights = fedavg(client_weights_list, client_sizes)
        set_weights(global_model, global_weights)

        # Weighted mean train loss across all clients
        total_train_size = sum(client_sizes)
        weighted_train_loss = float(
            sum(sz * loss for sz, loss in zip(client_sizes, client_train_losses)) / total_train_size
        )

        # Round evaluation on concatenated test set ("ALL") for display/monitoring only
        all_y_test_round = []
        all_scores_round = []
        for h in hospitals:
            scores = predict_scores(global_model, client_data[h]["X_test_scaled"])
            all_y_test_round.append(np.asarray(client_data[h]["y_test"]))
            all_scores_round.append(scores)

        concat_y_test_round = np.concatenate(all_y_test_round)
        concat_scores_round = np.concatenate(all_scores_round)
        round_test_auc = float(roc_auc_score(concat_y_test_round, concat_scores_round))

        round_logs.append({
            "round": r,
            "weighted_mean_train_loss": weighted_train_loss,
            "test_roc_auc_all": round_test_auc,
        })

    # Save per-round logging to results/federated_rounds.csv
    rounds_df = pd.DataFrame(round_logs)
    rounds_path = Path("results/federated_rounds.csv")
    rounds_path.parent.mkdir(parents=True, exist_ok=True)
    rounds_df.to_csv(rounds_path, index=False)

    # 4. Final evaluation of the converged global model on each hospital's test set
    results = []
    all_y_true = []
    all_preds = []
    all_scores = []

    for h in hospitals:
        y_test = client_data[h]["y_test"]
        scores = predict_scores(global_model, client_data[h]["X_test_scaled"])
        preds = (scores >= 0.5).astype(int)

        metrics = evaluate_model(y_test, preds, scores)
        results.append({
            "model": "federated_plain",
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

    # "ALL" row: metrics on concatenated predictions
    concat_y_true = np.concatenate(all_y_true)
    concat_preds = np.concatenate(all_preds)
    concat_scores = np.concatenate(all_scores)
    metrics_all = evaluate_model(concat_y_true, concat_preds, concat_scores)

    results.append({
        "model": "federated_plain",
        "hospital": "ALL",
        "accuracy": metrics_all["accuracy"],
        "precision": metrics_all["precision"],
        "recall": metrics_all["recall"],
        "roc_auc": metrics_all["roc_auc"],
        "n_test": len(concat_y_true),
    })

    # Save final results to results/federated_plain.csv
    results_path = Path("results/federated_plain.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    artifacts = {
        "global_model": global_model,
        "global_weights": global_weights,
        "imputers": imputers,
        "scalers": scalers,
        "splits": splits,
        "round_logs": round_logs,
        "rounds_df": rounds_df,
        "results_df": results_df,
    }
    run_federated.artifacts = artifacts

    if return_artifacts:
        return results, artifacts
    return results


if __name__ == "__main__":
    print("Selecting optimal hyperparameters with seed=42...")
    best_epochs, best_wd = select_hyperparams(seed=42)
    print(f"Using tuned hyperparameters: epochs={best_epochs}, weight_decay={best_wd}")

    print("\nRunning plain Federated Learning (FedAvg)...")
    results = run_federated(seed=42, epochs_total=best_epochs, weight_decay=best_wd, local_epochs=2)
    print("\nResults saved to results/federated_plain.csv and results/federated_rounds.csv:")
    print(pd.DataFrame(results).to_string(index=False))
