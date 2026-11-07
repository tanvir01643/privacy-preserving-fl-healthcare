from sklearn.model_selection import train_test_split


# Split a single hospital DataFrame into train and test sets
# using stratified sampling to preserve target class balance.
def split_hospital(df, test_size=0.2, seed=42):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


# Split all hospitals' DataFrames independently into train and test sets
# to prevent any cross-hospital data leakage.
def split_all(hospitals, test_size=0.2, seed=42):
    return {
        name: split_hospital(df, test_size=test_size, seed=seed)
        for name, df in hospitals.items()
    }
