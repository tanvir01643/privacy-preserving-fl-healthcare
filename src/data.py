import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

FILES = {
    "Cleveland":   "processed.cleveland.data",
    "Hungary":     "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA":          "processed.va.data",
}

# প্রায় খালি column, চার হাসপাতাল থেকেই বাদ
DROP_COLS = ["chol", "fbs", "slope", "ca", "thal"]


def load_raw(data_dir="data"):
    """চার হাসপাতালের ফাইল পড়ে dict বানায়। ? কে NaN ধরে।"""
    hospitals = {}
    for name, fname in FILES.items():
        hospitals[name] = pd.read_csv(
            f"{data_dir}/{fname}", header=None, names=COLS, na_values="?"
        )
    return hospitals


def clean_hospital(d):
    """একটা হাসপাতালের table পরিষ্কার করে। আসল table বদলায় না।"""
    c = d.copy()
    c["trestbps"] = c["trestbps"].replace(0, float("nan"))  # রক্তচাপ 0 অসম্ভব
    c = c.drop(columns=DROP_COLS)
    c["target"] = (c["num"] > 0).astype(int)  # 0 = রোগ নেই, 1 = আছে
    return c.drop(columns=["num"])


def load_clean(data_dir="data"):
    """চার হাসপাতালের পরিষ্কার table dict আকারে দেয়।"""
    return {name: clean_hospital(d) for name, d in load_raw(data_dir).items()}
