from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hr_data.csv"
ORIGINAL_DATA_PATH = ROOT / "hr_data_original_backup.csv"
MODEL_PATH = ROOT / "models" / "attrition_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.joblib"
OUTPUT_DIR = ROOT / "reports" / "lime"
RANDOM_STATE = 42
MISSING_CATEGORY = "__CAREERPATH_MISSING__"


def load_reference_data():
    source = ORIGINAL_DATA_PATH if ORIGINAL_DATA_PATH.exists() else DATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"No compatible dataset found at {source}.")
    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    return frame, source


def load_model_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "The trained pipeline or feature order is missing. "
            "Run `python train_model.py` first."
        )
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    if not hasattr(model, "named_steps") or not {"preprocess", "model"}.issubset(model.named_steps):
        raise ValueError(
            "The saved model must contain the original 'preprocess' and 'model' steps."
        )
    return model, feature_columns


def _class_name(value):
    if value == 1:
        return "Left"
    if value == 0:
        return "Stayed"
    return str(value)


def _encoded_data(
    reference,
    feature_columns,
    categorical_columns,
    categories,
    numeric_fill_values,
):
    encoded = pd.DataFrame(index=reference.index)
    for column in feature_columns:
        if column in categorical_columns:
            category_codes = {
                value: index for index, value in enumerate(categories[column])
            }
            missing_code = category_codes.get(MISSING_CATEGORY, 0)
            encoded[column] = reference[column].map(
                lambda value: category_codes.get(
                    MISSING_CATEGORY if pd.isna(value) else str(value),
                    missing_code,
                )
            ).astype(float)
        else:
            numeric = pd.to_numeric(reference[column], errors="coerce")
            encoded[column] = numeric.fillna(numeric_fill_values[column])
    return encoded.to_numpy(dtype=float)


def _make_lime_input(model, reference_data, feature_columns):
    missing = [column for column in feature_columns if column not in reference_data]
    if missing:
        raise ValueError(f"Reference data is missing model features: {missing}")

    reference = reference_data.reindex(columns=feature_columns).copy()
    preprocessor = model.named_steps["preprocess"]
    numeric_columns = [
        column for column in feature_columns
        if column in preprocessor.transformers_[0][2]
    ]
    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    categories = {}
    for column in categorical_columns:
        values = reference[column].dropna().astype(str).unique().tolist()
        if MISSING_CATEGORY in values:
            raise ValueError(
                f"Feature {column!r} contains a reserved missing-value token."
            )
        values = sorted(values)
        if reference[column].isna().any():
            values.append(MISSING_CATEGORY)
        if not values:
            values = [MISSING_CATEGORY]
        categories[column] = values

    numeric_imputer = (
        preprocessor.named_transformers_["num"].named_steps["imputer"]
    )
    numeric_fill_values = dict(zip(numeric_columns, numeric_imputer.statistics_))
    encoded_reference = _encoded_data(
        reference,
        feature_columns,
        categorical_columns,
        categories,
        numeric_fill_values,
    )
    category_indices = [
        feature_columns.index(column) for column in categorical_columns
    ]
    categorical_names = {
        feature_columns.index(column): categories[column]
        for column in categorical_columns
    }

    def predict_probabilities(encoded_rows):
        encoded_rows = np.asarray(encoded_rows)
        if encoded_rows.ndim == 1:
            encoded_rows = encoded_rows.reshape(1, -1)
        decoded = pd.DataFrame(index=range(len(encoded_rows)))
        for index, column in enumerate(feature_columns):
            if column in categorical_columns:
                options = categories[column]
                codes = np.rint(encoded_rows[:, index]).astype(int)
                codes = np.clip(codes, 0, len(options) - 1)
                decoded[column] = [
                    np.nan if options[code] == MISSING_CATEGORY else options[code]
                    for code in codes
                ]
            else:
                decoded[column] = encoded_rows[:, index]
        return model.predict_proba(decoded.reindex(columns=feature_columns))

    return (
        reference,
        encoded_reference,
        category_indices,
        categorical_names,
        predict_probabilities,
        numeric_fill_values,
    )


def explain_profile_lime(
    model,
    reference_data,
    profile,
    feature_columns,
    output_dir=OUTPUT_DIR,
    profile_id=None,
    num_features=12,
    num_samples=3000,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lime.lime_tabular import LimeTabularExplainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        reference,
        encoded_reference,
        categorical_indices,
        categorical_names,
        predict_probabilities,
        numeric_fill_values,
    ) = _make_lime_input(model, reference_data, feature_columns)
    row = profile.reindex(columns=feature_columns)
    if row.empty:
        raise ValueError("An employee profile is required for LIME.")
    encoded_profile = _encoded_data(
        row,
        feature_columns,
        [
            feature_columns[index]
            for index in categorical_indices
        ],
        {
            column: categorical_names[feature_columns.index(column)]
            for column in [
                feature_columns[index] for index in categorical_indices
            ]
        },
        numeric_fill_values,
    )[0]

    estimator = model.named_steps["model"]
    probabilities = predict_probabilities(encoded_profile)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = estimator.classes_[predicted_index]
    class_names = [_class_name(value) for value in estimator.classes_]
    explainer = LimeTabularExplainer(
        training_data=encoded_reference,
        feature_names=list(feature_columns),
        categorical_features=categorical_indices,
        categorical_names=categorical_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        encoded_profile,
        predict_probabilities,
        labels=(predicted_index,),
        num_features=min(num_features, len(feature_columns)),
        num_samples=num_samples,
    )

    terms = explanation.as_list(label=predicted_index)
    contribution_frame = pd.DataFrame(
        terms,
        columns=["Feature condition", "Contribution to predicted class"],
    )
    contribution_frame["Direction"] = np.where(
        contribution_frame["Contribution to predicted class"] >= 0,
        "Supports prediction",
        "Opposes prediction",
    )
    safe_id = str(profile_id if profile_id is not None else "profile").replace(
        "/", "_"
    ).replace("\\", "_")
    csv_path = output_dir / f"profile_{safe_id}_lime.csv"
    contribution_frame.to_csv(csv_path, index=False)

    ordered = contribution_frame.sort_values(
        "Contribution to predicted class",
        ascending=True,
    )
    colors = [
        "#38bdf8" if value >= 0 else "#f97316"
        for value in ordered["Contribution to predicted class"]
    ]
    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(11, 7))
    figure.set_facecolor("#0b1120")
    axis.set_facecolor("#111827")
    axis.barh(
        ordered["Feature condition"],
        ordered["Contribution to predicted class"],
        color=colors,
    )
    axis.axvline(0, color="#cbd5e1", linewidth=0.8)
    axis.set_title(f"LIME Local Explanation — Predicted {class_names[predicted_index]}")
    axis.set_xlabel("Contribution to the locally approximated predicted-class probability")
    axis.tick_params(colors="#e5e7eb")
    axis.xaxis.label.set_color("#e5e7eb")
    axis.yaxis.label.set_color("#e5e7eb")
    axis.title.set_color("#f8fafc")
    figure.tight_layout()
    plot_path = output_dir / f"profile_{safe_id}_lime.png"
    figure.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="#0b1120",
    )
    plt.close(figure)

    html_path = output_dir / f"profile_{safe_id}_lime.html"
    explanation.save_to_file(str(html_path))
    attrition_index = np.flatnonzero(estimator.classes_ == 1)
    attrition_probability = (
        float(probabilities[int(attrition_index[0])])
        if len(attrition_index) == 1 else None
    )
    result = {
        "profile_id": str(profile_id) if profile_id is not None else None,
        "predicted_class": str(predicted_class),
        "predicted_class_name": class_names[predicted_index],
        "predicted_class_probability": float(probabilities[predicted_index]),
        "attrition_probability": attrition_probability,
        "class_probabilities": {
            class_names[index]: float(probabilities[index])
            for index in range(len(class_names))
        },
        "contributions": contribution_frame,
        "plot": str(plot_path),
        "html": str(html_path),
        "csv": str(csv_path),
        "background_rows": len(reference),
    }
    json_path = output_dir / f"profile_{safe_id}_lime.json"
    payload = {
        key: value
        for key, value in result.items()
        if key != "contributions"
    }
    payload["contributions"] = contribution_frame.to_dict("records")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    result["json"] = str(json_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LIME explanation for one employee profile."
    )
    parser.add_argument(
        "--employee-number",
        type=str,
        help="EmployeeNumber to explain; defaults to the first available record.",
    )
    args = parser.parse_args()

    model, feature_columns = load_model_artifacts()
    reference, source = load_reference_data()
    missing = [column for column in feature_columns if column not in reference]
    if missing:
        raise ValueError(f"{source.name} is missing model features: {missing}")
    if args.employee_number is not None and "EmployeeNumber" in reference:
        selected = reference.loc[
            reference["EmployeeNumber"].astype(str) == args.employee_number
        ]
        if selected.empty:
            raise ValueError(
                f"EmployeeNumber {args.employee_number!r} was not found in {source.name}."
            )
        profile = selected.iloc[[0]].reindex(columns=feature_columns)
        profile_id = args.employee_number
    else:
        profile = reference.reindex(columns=feature_columns).iloc[[0]]
        profile_id = None

    result = explain_profile_lime(
        model,
        reference.reindex(columns=feature_columns),
        profile,
        feature_columns,
        profile_id=profile_id,
    )
    print(f"LIME explanation saved to: {OUTPUT_DIR}")
    print(f"Predicted class: {result['predicted_class_name']}")
    print(f"Attrition probability: {result['attrition_probability']}")
    print(f"Explanation plot: {result['plot']}")


if __name__ == "__main__":
    main()
