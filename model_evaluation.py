
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METRICS_FILE = MODEL_DIR / "model_metrics.json"
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            "model_metrics.json not found. "
            "Run python train_model.py first."
        )

    with open(METRICS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix, model_name):
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Stayed", "Left"])
    ax.set_yticklabels(["Stayed", "Left"])

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{model_name} - Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=13
            )

    fig.colorbar(image, ax=ax)
    filename = model_name.lower().replace(" ", "_")
    save_chart(f"{filename}_confusion_matrix.png")


def main():
    data = load_metrics()
    results = data["results"]

    print("\nCAREERPATH ANALYTICS - MODEL EVALUATION")
    print("=" * 55)

    print("Selected model:", data["selected_model"])
    print("Training rows:", data["training_rows"])
    print("Testing rows:", data["testing_rows"])
    print("Split strategy:", data["split_strategy"])

    # 1. Prepare comparison table
    records = []

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    for model_name, metrics in results.items():
        record = {"Model": model_name}

        for metric in metric_names:
            record[metric.replace("_", " ").title()] = metrics[metric]

        records.append(record)

    comparison = pd.DataFrame(records)

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # 2. Accuracy comparison
    plt.figure(figsize=(7, 5))

    plt.bar(
        comparison["Model"],
        comparison["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    save_chart("accuracy_comparison.png")

    # 3. Compare precision, recall and F1
    metrics_to_plot = [
        "Precision",
        "Recall",
        "F1"
    ]

    comparison.set_index("Model")[metrics_to_plot].plot(
        kind="bar",
        figsize=(9, 6)
    )

    plt.title("Precision, Recall and F1-score Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

    save_chart("classification_metrics_comparison.png")

    # 4. ROC-AUC comparison
    plt.figure(figsize=(7, 5))

    plt.bar(
        comparison["Model"],
        comparison["Roc Auc"]
    )

    plt.title("ROC-AUC Comparison")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1)

    save_chart("roc_auc_comparison.png")

    # 5. Confusion matrices
    for model_name, metrics in results.items():
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            model_name
        )

    # 6. Classification reports
    for model_name, metrics in results.items():
        report = pd.DataFrame(
            metrics["classification_report"]
        ).T

        filename = (
            model_name.lower().replace(" ", "_")
            + "_classification_report.csv"
        )

        report.to_csv(OUTPUT_DIR / filename)

    # 7. Print interpretation
    best_model = data["selected_model"]
    print("\nMODEL SELECTION")
    print("Selected model:", best_model)
    print(
        "Selection criterion: highest ROC-AUC "
        "in the saved evaluation results."
    )

    print("\nIMPORTANT INTERPRETATION")
    print(
        "Do not select a model using accuracy alone. "
        "Compare precision, recall, F1-score and balanced accuracy."
    )

    print("\nLIMITATION")
    print(data.get("dataset_limitations", "See project documentation."))

    print("\nMODEL EVALUATION COMPLETED")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
