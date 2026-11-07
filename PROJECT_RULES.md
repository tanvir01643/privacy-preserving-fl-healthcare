RULES (read this file at the start of every task):
1. Project: Privacy-Preserving Cross-Hospital Disease Prediction using Federated Learning (UCI Heart Disease, 4 hospitals = 4 clients).
2. Language: simple Python 3.11. Use only pandas, numpy, scikit-learn, PyTorch (small MLP 8->16->8->1), phe (Paillier), streamlit. Ask me before adding any other library.
3. Code goes in small functions in src/ (data, split, model, train, evaluate, fedavg, encryption). No giant files. Tests go in tests/.
4. Never commit, never push, never create branches. I do Git myself.
5. Never read, print, move or delete anything outside this project folder. Never touch venv/, data/, .env, .gitignore unless I ask.
6. Never hard-code API keys or secrets.
7. Split train/test BEFORE any imputation or scaling. Fit imputers/scalers on train data only. Same random seed (42) everywhere.
8. Every experiment writes its result to results/*.csv (columns: model, hospital, accuracy, precision, recall, roc_auc).
9. Write a short comment in simple English above each function saying what it does and why.
10. When a task is done: list files created/changed, how to run, expected output, and anything you were unsure about. Then stop and wait.
