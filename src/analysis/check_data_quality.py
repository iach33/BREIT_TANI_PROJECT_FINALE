import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from config import settings

def check_quality(file_path):
    print(f"=== Checking Modeling Readiness: {file_path.name} ===")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")
    
    # 1. Missing Values
    print("\n--- Missing Values ---")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if not missing_cols.empty:
        print(f"Columns with missing values: {len(missing_cols)}")
        print(missing_cols.head(10))
        
        # Critical check: Rows with ALL features missing?
        # Assuming features start after N_HC
        feat_cols = [c for c in df.columns if c != 'N_HC']
        rows_all_missing = df[feat_cols].isna().all(axis=1).sum()
        print(f"Rows with ALL features missing: {rows_all_missing}")
    else:
        print("No missing values found!")

    # 2. Infinite Values
    print("\n--- Infinite Values ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols]).sum().sum()
    if inf_count > 0:
        print(f"Total infinite values found: {inf_count}")
        inf_cols = df[num_cols].columns[np.isinf(df[num_cols]).any()].tolist()
        print(f"Columns with inf: {inf_cols}")
    else:
        print("No infinite values found.")

    # 3. Target Balance
    print("\n--- Target Balance (deficit) ---")
    if 'deficit' in df.columns:
        counts = df['deficit'].value_counts(normalize=True) * 100
        print(counts)
        if counts.min() < 5:
            print("WARNING: Highly imbalanced target (<5% positive class)")
    else:
        print("Target 'deficit' not found!")

    # 4. Data Types
    print("\n--- Non-Numeric Columns ---")
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"Non-numeric columns (need encoding?): {non_num}")
    else:
        print("All columns are numeric.")

    # 5. Constant Columns
    print("\n--- Constant Columns ---")
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        print(f"Constant columns (no information): {const_cols}")
    else:
        print("No constant columns found.")

if __name__ == "__main__":
    check_quality(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")
