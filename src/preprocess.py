import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Fit a median imputer on training features only to avoid data leakage.
def fit_imputer(X_train):
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


# Apply the fitted imputer and return a DataFrame with original column names and index.
def apply_imputer(imputer, X):
    transformed = imputer.transform(X)
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)


# Fit a standard scaler on imputed training features only to avoid data leakage.
def fit_scaler(X_imputed):
    scaler = StandardScaler()
    scaler.fit(X_imputed)
    return scaler


# Apply the fitted scaler to features, preserving DataFrame structure if input is a DataFrame.
def apply_scaler(scaler, X):
    transformed = scaler.transform(X)
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(transformed, columns=X.columns, index=X.index)
    return transformed

