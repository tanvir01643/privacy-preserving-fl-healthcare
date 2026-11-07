import sys
from pathlib import Path

# Add project root to sys.path so src imports work when running directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from src.data import load_clean
from src.split import split_all, split_hospital
from src.preprocess import fit_imputer, apply_imputer


# Verify train/test splitting and missing-value imputation for all 4 hospitals.
def test_splits_and_imputation():
    hospitals = load_clean()
    assert len(hospitals) == 4, f"Expected 4 hospitals, got {len(hospitals)}"

    splits = split_all(hospitals, test_size=0.2, seed=42)

    for name, df in hospitals.items():
        print(f"Testing hospital: {name}")
        X_train, X_test, y_train, y_test = splits[name]

        # Verify split_hospital produces the identical split directly
        sh_X_train, sh_X_test, sh_y_train, sh_y_test = split_hospital(df, test_size=0.2, seed=42)
        assert X_train.equals(sh_X_train), f"{name}: split_hospital X_train differs from split_all"
        assert X_test.equals(sh_X_test), f"{name}: split_hospital X_test differs from split_all"
        assert y_train.equals(sh_y_train), f"{name}: split_hospital y_train differs from split_all"
        assert y_test.equals(sh_y_test), f"{name}: split_hospital y_test differs from split_all"

        # 1. No row index appears in both train and test
        overlap = set(X_train.index).intersection(set(X_test.index))
        assert len(overlap) == 0, f"{name}: row index overlap found: {overlap}"
        assert set(y_train.index).isdisjoint(set(y_test.index)), f"{name}: y index overlap found"
        assert (X_train.index == y_train.index).all(), f"{name}: X_train and y_train index mismatch"
        assert (X_test.index == y_test.index).all(), f"{name}: X_test and y_test index mismatch"

        # 2. Train + test rows = original rows
        assert len(X_train) + len(X_test) == len(df), (
            f"{name}: train rows ({len(X_train)}) + test rows ({len(X_test)}) != original ({len(df)})"
        )
        assert len(y_train) + len(y_test) == len(df), (
            f"{name}: y_train rows ({len(y_train)}) + y_test rows ({len(y_test)}) != original ({len(df)})"
        )

        # 3. Target ratio in train and test differs by less than 0.05
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        ratio_diff = abs(train_ratio - test_ratio)
        assert ratio_diff < 0.05, (
            f"{name}: target ratio difference {ratio_diff:.4f} is >= 0.05 (train={train_ratio:.4f}, test={test_ratio:.4f})"
        )

        # 4. After imputing, there is no NaN in train or test
        imputer = fit_imputer(X_train)
        X_train_imp = apply_imputer(imputer, X_train)
        X_test_imp = apply_imputer(imputer, X_test)

        # Check DataFrame columns and index preservation
        assert list(X_train_imp.columns) == list(X_train.columns), f"{name}: train columns changed"
        assert list(X_test_imp.columns) == list(X_test.columns), f"{name}: test columns changed"
        assert (X_train_imp.index == X_train.index).all(), f"{name}: train indices changed"
        assert (X_test_imp.index == X_test.index).all(), f"{name}: test indices changed"

        assert X_train_imp.isna().sum().sum() == 0, f"{name}: NaNs remain in X_train after imputation"
        assert X_test_imp.isna().sum().sum() == 0, f"{name}: NaNs remain in X_test after imputation"

        # 5. Imputer's medians equal train medians, not full-data medians (no data leakage)
        train_medians = X_train.median().values
        full_medians = df.drop(columns=["target"]).median().values

        assert np.allclose(imputer.statistics_, train_medians), (
            f"{name}: imputer statistics do not equal train medians"
        )
        assert not np.allclose(imputer.statistics_, full_medians), (
            f"{name}: imputer statistics equal full-data medians (data leakage detected)"
        )

        print(f"  [PASS] {name}: rows={len(df)} (train={len(X_train)}, test={len(X_test)}), "
              f"ratio_diff={ratio_diff:.4f}, no NaNs, train medians verified.")

    print("\nAll assertions passed successfully for all 4 hospitals!")


if __name__ == "__main__":
    test_splits_and_imputation()
