
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hr_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Exclude target, identifiers, constant fields, and selected
# demographic fields from this educational demonstration.
DROP_COLUMNS = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "Age",
    "Gender",
    "MaritalStatus",
    "EducationField",
]


def main():
    # --------------------------------------------------
    # 1. Load and validate the dataset
    # --------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place hr_data.csv in the project root folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "Attrition" not in df.columns:
        raise ValueError(
            "The CSV must contain an 'Attrition' target column."
        )

    df = df.dropna(subset=["Attrition"]).copy()

    y = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain both Attrition classes: Yes and No."
        )

    # --------------------------------------------------
    # 2. Prepare model features
    # --------------------------------------------------
    feature_drop = [
        column for column in DROP_COLUMNS
        if column in df.columns
    ]

    X = df.drop(
        columns=feature_drop,
        errors="ignore"
    ).copy()

    # Remove constant columns.
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if X.empty:
        raise ValueError("No usable feature columns remain.")

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No numeric or categorical features were found.")

    # --------------------------------------------------
    # 3. Build preprocessing pipelines
    # --------------------------------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    # --------------------------------------------------
    # 4. Group identical profiles before splitting
    # --------------------------------------------------
    # EmployeeNumber is excluded from X above.
    # Identical feature profiles are assigned to one group,
    # so the same profile cannot appear in both split sets.
    #
    # IMPORTANT: this reduces profile leakage but does not
    # make resampled records independent employee observations.

    profile_groups = pd.util.hash_pandas_object(
        X.astype(str),
        index=False,
    )

    group_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        group_split.split(
            X,
            y,
            groups=profile_groups,
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The grouped split does not contain both target classes "
            "in training and testing. Use a suitable independent dataset "
            "or revise the evaluation strategy."
        )

    train_groups = set(profile_groups.iloc[train_idx])
    test_groups = set(profile_groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            "Data leakage detected: identical profiles occur in both sets."
        )

    print("\nDATA SPLIT SUMMARY")
    print("-" * 40)
    print(f"Total rows: {len(X):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Features used: {len(X.columns)}")
    print(f"Training groups: {len(train_groups):,}")
    print(f"Testing groups: {len(test_groups):,}")
    print(f"Overlapping profile groups: {len(overlap)}")
    print(
        "\nNote: the dataset contains resampled profiles. "
        "Results are educational and not evidence of performance "
        "on independent real-world employees."
    )

    # --------------------------------------------------
    # 5. Define candidate models
    # --------------------------------------------------
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # --------------------------------------------------
    # 6. Train and evaluate candidate models
    # --------------------------------------------------
    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Stayed", "Left"],
                output_dict=True,
                zero_division=0,
            ),
        }

        results[name] = metrics
        fitted[name] = pipeline

    # This retains your existing model-selection approach.
    # For rigorous model selection, use a separate validation set
    # and reserve the test set for one final evaluation.
    best_name = max(
        results,
        key=lambda name: results[name]["roc_auc"],
    )
    best_pipeline = fitted[best_name]

    # --------------------------------------------------
    # 7. Calculate permutation feature importance
    # --------------------------------------------------
    importance_result = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance_result.importances_mean,
        "ImportanceStd": importance_result.importances_std,
    }).sort_values(
        "Importance",
        ascending=False,
    )

    importance_df.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTOP 10 FEATURES BY PERMUTATION IMPORTANCE")
    print("-" * 50)
    print(importance_df.head(10).to_string(index=False))

    # --------------------------------------------------
    # 8. Save the selected model and metadata
    # --------------------------------------------------
    joblib.dump(
        best_pipeline,
        MODEL_DIR / "attrition_model.joblib",
    )

    joblib.dump(
        list(X.columns),
        MODEL_DIR / "feature_columns.joblib",
    )

    metrics_output = {
        "selected_model": best_name,
        "target": "Attrition (Yes=1, No=0)",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_strategy": "GroupShuffleSplit by identical feature profile",
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_columns": list(X.columns),
        "excluded_fields": feature_drop,
        "results": results,
        "dataset_limitations": (
            "The dataset contains 1,470 original records and "
            "14,530 resampled profiles with assigned identifiers. "
            "It does not represent 16,000 independent employee "
            "observations. Grouped evaluation reduces identical-profile "
            "leakage but cannot establish real-world generalization."
        ),
        "intended_use": (
            "Educational demonstration only. Not for real employment "
            "decisions or individual employee assessment."
        ),
    }

    with open(
        MODEL_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_output,
            file,
            indent=2,
        )

    # --------------------------------------------------
    # 9. Print evaluation summary
    # --------------------------------------------------
    print("\nTRAINING COMPLETED")
    print("-" * 40)
    print(f"Rows used: {len(df):,}")
    print(f"Features used: {len(X.columns)}")
    print(f"Selected model by test ROC-AUC: {best_name}")

    print("\nMODEL COMPARISON")
    for name, metrics in results.items():
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.3f}, "
            f"Balanced Accuracy={metrics['balanced_accuracy']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )

    print(f"\nSaved model files in: {MODEL_DIR}")
    print(
        "\nReminder: these metrics are for an educational, "
        "resampled dataset and should not be interpreted as "
        "real-world employee attrition performance."
    )


if __name__ == "__main__":
    main()
