
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA = ROOT / "hr_data_original_backup.csv"
WORKING_DATA = ROOT / "hr_data.csv"

OUTPUT_DIR = ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    # Prefer original independent records for descriptive EDA.
    if ORIGINAL_DATA.exists():
        path = ORIGINAL_DATA
        print("Using original dataset for EDA.")
    elif WORKING_DATA.exists():
        path = WORKING_DATA
        print("WARNING: Using expanded working dataset.")
    else:
        raise FileNotFoundError("No HR dataset found.")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df, path.name


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df, source_name = load_dataset()

    print("\nCAREERPATH ANALYTICS - EDA REPORT")
    print("=" * 50)

    print(f"Dataset source: {source_name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 1. Basic information
    print("\nDATASET INFORMATION")
    print(df.info())

    # 2. Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS")
    print(df.describe(include="all").T)

    df.describe(include="all").T.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv"
    )

    # 3. Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    print("\nMISSING VALUES")
    print(missing[missing > 0])

    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    # 4. Duplicate records
    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    # 5. Attrition distribution
    if "Attrition" in df.columns:
        attrition_counts = df["Attrition"].value_counts()

        print("\nATTRITION DISTRIBUTION")
        print(attrition_counts)

        plt.figure(figsize=(7, 5))
        attrition_counts.plot(kind="bar")
        plt.title("Employee Attrition Distribution")
        plt.xlabel("Attrition")
        plt.ylabel("Number of Employees")
        plt.xticks(rotation=0)
        save_chart("attrition_distribution.png")

        attrition_percent = (
            df["Attrition"].value_counts(normalize=True) * 100
        )

        attrition_percent.to_csv(
            OUTPUT_DIR / "attrition_percentages.csv"
        )

    # 6. Department-wise employee distribution
    if "Department" in df.columns:
        department_counts = df["Department"].value_counts()

        print("\nDEPARTMENT DISTRIBUTION")
        print(department_counts)

        plt.figure(figsize=(8, 5))
        department_counts.plot(kind="bar")
        plt.title("Employee Distribution by Department")
        plt.xlabel("Department")
        plt.ylabel("Number of Employees")
        plt.xticks(rotation=30, ha="right")
        save_chart("department_distribution.png")

    # 7. Department-wise attrition rate
    if {"Department", "Attrition"}.issubset(df.columns):
        dept_attrition = (
            df.assign(
                AttritionFlag=df["Attrition"].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
            )
            .groupby("Department")["AttritionFlag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        print("\nDEPARTMENT-WISE ATTRITION RATE (%)")
        print(dept_attrition.round(2))

        dept_attrition.to_csv(
            OUTPUT_DIR / "department_attrition_rate.csv"
        )

        plt.figure(figsize=(8, 5))
        dept_attrition.plot(kind="bar")
        plt.title("Attrition Rate by Department")
        plt.xlabel("Department")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=30, ha="right")
        save_chart("department_attrition.png")

    # 8. Job role distribution
    if "JobRole" in df.columns:
        role_counts = df["JobRole"].value_counts()

        plt.figure(figsize=(10, 6))
        role_counts.plot(kind="bar")
        plt.title("Employee Distribution by Job Role")
        plt.xlabel("Job Role")
        plt.ylabel("Number of Employees")
        plt.xticks(rotation=45, ha="right")
        save_chart("job_role_distribution.png")

        print("\nJOB ROLE DISTRIBUTION")
        print(role_counts)

    # 9. Overtime and attrition
    if {"OverTime", "Attrition"}.issubset(df.columns):
        overtime_attrition = (
            df.assign(
                AttritionFlag=df["Attrition"].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
            )
            .groupby("OverTime")["AttritionFlag"]
            .mean()
            .mul(100)
        )

        print("\nATTRITION RATE BY OVERTIME (%)")
        print(overtime_attrition.round(2))

        plt.figure(figsize=(7, 5))
        overtime_attrition.plot(kind="bar")
        plt.title("Attrition Rate by Overtime")
        plt.xlabel("Overtime")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=0)
        save_chart("overtime_attrition.png")

    # 10. Job satisfaction and attrition
    if {"JobSatisfaction", "Attrition"}.issubset(df.columns):
        satisfaction_attrition = (
            df.assign(
                AttritionFlag=df["Attrition"].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
            )
            .groupby("JobSatisfaction")["AttritionFlag"]
            .mean()
            .mul(100)
        )

        print("\nATTRITION RATE BY JOB SATISFACTION (%)")
        print(satisfaction_attrition.round(2))

        plt.figure(figsize=(7, 5))
        satisfaction_attrition.plot(kind="bar")
        plt.title("Attrition Rate by Job Satisfaction")
        plt.xlabel("Job Satisfaction Level")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=0)
        save_chart("satisfaction_attrition.png")

    # 11. Monthly income by job level
    if {"JobLevel", "MonthlyIncome"}.issubset(df.columns):
        income_by_level = df.groupby("JobLevel")[
            "MonthlyIncome"
        ].mean()

        print("\nAVERAGE MONTHLY INCOME BY JOB LEVEL")
        print(income_by_level.round(2))

        income_by_level.to_csv(
            OUTPUT_DIR / "income_by_job_level.csv"
        )

        plt.figure(figsize=(7, 5))
        income_by_level.plot(kind="bar")
        plt.title("Average Monthly Income by Job Level")
        plt.xlabel("Job Level")
        plt.ylabel("Average Monthly Income")
        plt.xticks(rotation=0)
        save_chart("income_by_job_level.png")

    # 12. Correlation heatmap using Matplotlib
    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr()

    if not correlation.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))
        ax.set_xticklabels(correlation.columns, rotation=90)
        ax.set_yticklabels(correlation.columns)

        ax.set_title("Numerical Feature Correlation Matrix")
        fig.colorbar(image, ax=ax, label="Correlation")
        save_chart("correlation_matrix.png")

        correlation.to_csv(
            OUTPUT_DIR / "correlation_matrix.csv"
        )

    # 13. Save a summary
    summary = {
        "dataset_source": source_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "attrition_distribution": (
            df["Attrition"].value_counts().to_dict()
            if "Attrition" in df.columns else {}
        ),
    }

    with open(OUTPUT_DIR / "eda_summary.json", "w") as file:
        json.dump(summary, file, indent=4, default=str)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print(f"All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
