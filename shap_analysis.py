from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hr_data.csv"
ORIGINAL_DATA_PATH = ROOT / "hr_data_original_backup.csv"
MODEL_PATH = ROOT / "models" / "attrition_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.joblib"
OUTPUT_DIR = ROOT / "reports" / "shap"
POSITIVE_CLASS = 1
RANDOM_STATE = 42


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


def _positive_class_index(estimator):
    classes = np.asarray(estimator.classes_)
    indexes = np.flatnonzero(classes == POSITIVE_CLASS)
    if len(indexes) != 1:
        raise ValueError(
            f"Expected attrition class {POSITIVE_CLASS!r} in model.classes_; "
            f"found {classes.tolist()}."
        )
    return int(indexes[0])


def _dense(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def _class_shap_values(values, sample_count, feature_count, class_index):
    if isinstance(values, list):
        if class_index >= len(values):
            raise ValueError("SHAP returned fewer class outputs than the model.")
        values = values[class_index]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[:2] == (sample_count, feature_count):
            values = values[:, :, class_index]
        elif values.shape[1:] == (sample_count, feature_count):
            values = values[class_index]
        else:
            raise ValueError(f"Unrecognized SHAP output dimensions: {values.shape}.")
    elif values.ndim == 2:
        if values.shape == (feature_count, sample_count):
            values = values.T
        elif values.shape != (sample_count, feature_count):
            raise ValueError(f"Unrecognized SHAP output dimensions: {values.shape}.")
    else:
        raise ValueError(f"Unrecognized SHAP output dimensions: {values.shape}.")
    return values


def _positive_base_value(expected_value, class_index):
    values = np.asarray(expected_value)
    if values.ndim == 0:
        return float(values)
    flattened = values.reshape(-1)
    if class_index >= len(flattened):
        raise ValueError("SHAP expected values do not include the attrition class.")
    return float(flattened[class_index])


def _display_feature_names(preprocessor):
    return [
        name.split("__", 1)[-1]
        for name in preprocessor.get_feature_names_out().tolist()
    ]


def _transformed(model, frame):
    transformed = model.named_steps["preprocess"].transform(frame)
    return _dense(transformed)


def _make_explainer(model):
    import shap

    estimator = model.named_steps["model"]
    if not isinstance(estimator, RandomForestClassifier):
        raise ValueError(
            "The saved estimator is not a RandomForestClassifier; "
            "Tree SHAP probability explanations are configured for the existing Random Forest."
        )
    _positive_class_index(estimator)
    # Tree SHAP's raw output for this sklearn classifier is per-class probability.
    return shap.TreeExplainer(
        estimator,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )


def calculate_global_shap(
    model,
    reference_data,
    feature_columns,
    output_dir=OUTPUT_DIR,
    max_samples=500,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [column for column in feature_columns if column not in reference_data]
    if missing:
        raise ValueError(f"Reference data is missing model features: {missing}")

    unique = reference_data.reindex(columns=feature_columns).drop_duplicates()
    if unique.empty:
        raise ValueError("No complete profile rows are available for SHAP.")
    if len(unique) > max_samples:
        explain_data = unique.sample(n=max_samples, random_state=RANDOM_STATE)
    else:
        explain_data = unique

    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    class_index = _positive_class_index(estimator)
    transformed_data = _transformed(model, explain_data)
    feature_names = _display_feature_names(preprocessor)
    if transformed_data.shape[1] != len(feature_names):
        raise ValueError("Transformed features do not match preprocessing feature names.")

    explainer = _make_explainer(model)
    raw_values = explainer.shap_values(transformed_data, check_additivity=True)
    values = _class_shap_values(
        raw_values,
        len(explain_data),
        len(feature_names),
        class_index,
    )
    expected_value = _positive_base_value(explainer.expected_value, class_index)
    model_probabilities = estimator.predict_proba(transformed_data)[:, class_index]
    if not np.allclose(
        expected_value + values.sum(axis=1),
        model_probabilities,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "SHAP contributions did not add up to the Random Forest's attrition probability."
        )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean absolute SHAP value": np.abs(values).mean(axis=0),
        "Mean SHAP value": values.mean(axis=0),
    }).sort_values("Mean absolute SHAP value", ascending=False)
    importance_path = output_dir / "global_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    plt.style.use("dark_background")
    shap.summary_plot(
        values,
        transformed_data,
        feature_names=feature_names,
        max_display=min(20, len(feature_names)),
        show=False,
        plot_type="dot",
    )
    summary_path = output_dir / "shap_summary_beeswarm.png"
    summary_figure = plt.gcf()
    summary_figure.set_size_inches(12, 8)
    for axis in summary_figure.axes:
        axis.set_facecolor("#111827")
        axis.tick_params(colors="#e5e7eb")
        axis.xaxis.label.set_color("#e5e7eb")
        axis.yaxis.label.set_color("#e5e7eb")
        axis.title.set_color("#f8fafc")
    summary_figure.axes[0].set_xlabel(
        "SHAP value (impact on attrition probability)"
    )
    summary_figure.savefig(
        summary_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="#0b1120",
    )
    plt.close(summary_figure)

    importance_figure, importance_axis = plt.subplots(figsize=(10, 7))
    importance_axis.barh(
        importance.head(20)["Feature"][::-1],
        importance.head(20)["Mean absolute SHAP value"][::-1],
        color="#38bdf8",
    )
    importance_axis.set_title("Global SHAP Feature Importance")
    importance_axis.set_xlabel("Mean absolute contribution to attrition probability")
    importance_figure.tight_layout()
    importance_path_png = output_dir / "shap_global_feature_importance.png"
    importance_figure.savefig(
        importance_path_png,
        dpi=180,
        bbox_inches="tight",
        facecolor="#0b1120",
    )
    plt.close(importance_figure)

    return {
        "importance": importance,
        "summary_plot": str(summary_path),
        "importance_plot": str(importance_path_png),
        "explained_rows": len(explain_data),
        "feature_names": feature_names,
        "shap_values": values,
        "transformed_data": transformed_data,
        "expected_value": expected_value,
        "class_index": class_index,
    }


def explain_profile_shap(
    model,
    profile,
    feature_columns,
    output_dir=OUTPUT_DIR,
    profile_id=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = profile.reindex(columns=feature_columns)
    if row.empty:
        raise ValueError("An employee profile is required for SHAP.")
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    class_index = _positive_class_index(estimator)
    transformed_profile = _transformed(model, row)
    feature_names = _display_feature_names(preprocessor)
    explainer = _make_explainer(model)
    raw_values = explainer.shap_values(transformed_profile, check_additivity=True)
    values = _class_shap_values(
        raw_values,
        1,
        len(feature_names),
        class_index,
    )[0]
    base_value = _positive_base_value(explainer.expected_value, class_index)
    probability = float(model.predict_proba(row)[0, class_index])
    estimator_probability = float(
        estimator.predict_proba(transformed_profile)[0, class_index]
    )
    if not np.isclose(probability, estimator_probability, rtol=1e-7, atol=1e-7):
        raise RuntimeError(
            "The saved pipeline and its Random Forest produced different probabilities."
        )
    if not np.isclose(base_value + values.sum(), probability, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "SHAP contributions did not add up to the selected profile's attrition probability."
        )
    prediction_index = int(np.argmax(model.predict_proba(row)[0]))
    predicted_class = estimator.classes_[prediction_index]
    safe_id = str(profile_id if profile_id is not None else "profile").replace(
        "/", "_"
    ).replace("\\", "_")

    contributions = pd.DataFrame({
        "Feature": feature_names,
        "Transformed profile value": transformed_profile[0],
        "SHAP contribution to attrition probability": values,
        "Direction": np.where(values >= 0, "Increases attrition probability", "Decreases attrition probability"),
    })
    contributions["Magnitude"] = contributions[
        "SHAP contribution to attrition probability"
    ].abs()
    contributions = contributions.sort_values("Magnitude", ascending=False)
    csv_path = output_dir / f"profile_{safe_id}_shap.csv"
    contributions.drop(columns="Magnitude").to_csv(csv_path, index=False)

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=transformed_profile[0],
        feature_names=feature_names,
    )
    plt.style.use("dark_background")
    shap.plots.waterfall(
        explanation,
        max_display=min(15, len(feature_names)),
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plot_path = output_dir / f"profile_{safe_id}_shap_waterfall.png"
    figure.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="#0b1120",
    )
    plt.close(figure)

    result = {
        "profile_id": str(profile_id) if profile_id is not None else None,
        "positive_class": str(estimator.classes_[class_index]),
        "predicted_class": str(predicted_class),
        "attrition_probability": probability,
        "base_value": base_value,
        "contributions": contributions.drop(columns="Magnitude"),
        "plot": str(plot_path),
        "csv": str(csv_path),
    }
    json_path = output_dir / f"profile_{safe_id}_shap.json"
    payload = {
        key: value
        for key, value in result.items()
        if key != "contributions"
    }
    payload["contributions"] = contributions.drop(columns="Magnitude").to_dict("records")
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    result["json"] = str(json_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP global and individual attrition explanations."
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
    rows = reference.reindex(columns=feature_columns)
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
        profile = rows.iloc[[0]]
        profile_id = None

    global_result = calculate_global_shap(
        model,
        rows,
        feature_columns,
    )
    local_result = explain_profile_shap(
        model,
        profile,
        feature_columns,
        profile_id=profile_id,
    )
    print(f"Global SHAP plots and results saved to: {OUTPUT_DIR}")
    print(f"Explained profiles: {global_result['explained_rows']}")
    print(f"Attrition probability for selected profile: {local_result['attrition_probability']:.4f}")
    print(f"Individual explanation: {local_result['plot']}")


if __name__ == "__main__":
    main()
