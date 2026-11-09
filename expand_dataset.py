
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "hr_data_original_backup.csv"
OUTPUT = ROOT / "hr_data_expanded.csv"
TARGET_ROWS = 16000

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Please create hr_data_original_backup.csv first."
        )

    df = pd.read_csv(SOURCE)
    df.columns = df.columns.str.strip()

    if len(df) >= TARGET_ROWS:
        print(f"Dataset already contains {len(df):,} rows.")
        return

    if "Attrition" not in df.columns or "Department" not in df.columns:
        raise ValueError("The source dataset does not have expected HR columns.")

    # Retain every original row and sample additional rows
    # from the existing dataset. Added rows are simulated duplicates.
    additional_count = TARGET_ROWS - len(df)
    extra = df.sample(
        n=additional_count,
        replace=True,
        random_state=42
    ).copy()

    # Give added rows unique identifiers where the column exists.
    if "EmployeeNumber" in df.columns:
        existing_ids = pd.to_numeric(
            df["EmployeeNumber"], errors="coerce"
        ).dropna()

        start_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
        extra["EmployeeNumber"] = range(
            start_id, start_id + additional_count
        )

    expanded = pd.concat([df, extra], ignore_index=True)
    expanded.to_csv(OUTPUT, index=False)

    print("Dataset expansion completed.")
    print(f"Original rows: {len(df):,}")
    print(f"Added simulated rows: {additional_count:,}")
    print(f"Total rows: {len(expanded):,}")
    print(f"Columns: {len(expanded.columns)}")
    print(f"Saved to: {OUTPUT}")
    print(
        "IMPORTANT: Added records are resampled copies, "
        "not independent observations of real employees."
    )

if __name__ == "__main__":
    main()
