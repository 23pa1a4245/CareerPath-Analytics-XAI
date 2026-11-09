
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "hr_data.csv"
BACKUP_FILE = ROOT / "hr_data_original_backup.csv"

def main():
    df = pd.read_csv(DATA_FILE)

    print("\nCAREERPATH ANALYTICS — DATA QUALITY REPORT")
    print("-" * 45)
    print(f"Current rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "EmployeeNumber" in df.columns:
        print(
            "Unique employee IDs:",
            df["EmployeeNumber"].nunique()
        )

        profile_cols = [
            col for col in df.columns
            if col != "EmployeeNumber"
        ]
        repeated = df.duplicated(
            subset=profile_cols
        ).sum()

        print("Repeated profiles excluding ID:", repeated)

    print("\nMissing values:", int(df.isna().sum().sum()))

    if "Attrition" in df.columns:
        print("\nAttrition distribution:")
        print(df["Attrition"].value_counts(dropna=False))

    if BACKUP_FILE.exists():
        original = pd.read_csv(BACKUP_FILE)
        print(f"\nOriginal backup rows: {len(original):,}")
        print(
            "Additional rows compared with original:",
            max(0, len(df) - len(original))
        )

    print(
        "\nCaveat: resampled records are not independent "
        "employee observations."
    )

if __name__ == "__main__":
    main()
