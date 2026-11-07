import pandas as pd
from sklearn.impute import SimpleImputer


# Fit a median imputer on training features only to avoid data leakage.
def fit_imputer(X_train):
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


# Apply the fitted imputer and return a DataFrame with original column names and index.
def apply_imputer(imputer, X):
    transformed = imputer.transform(X)
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)
