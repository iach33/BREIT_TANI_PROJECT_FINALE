import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from config import settings

def clean_patient_features(input_path, output_path):
    print(f"=== Cleaning Patient Features: {input_path.name} ===")
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Original Shape: {df.shape}")
    
    # 1. Drop Constant Columns
    print("\n--- Dropping Constant Columns ---")
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    # Keep N_HC even if constant (unlikely)
    const_cols = [c for c in const_cols if c != 'N_HC']
    
    if const_cols:
        print(f"Dropping {len(const_cols)} constant columns: {const_cols}")
        df.drop(columns=const_cols, inplace=True)
    else:
        print("No constant columns to drop.")
        
    # 2. Impute Missing Values
    print("\n--- Imputing Missing Values ---")
    # Strategy:
    # - Flags/Counts: Fill with 0 (assumption: missing means event didn't happen or count is 0)
    # - Slopes: Fill with 0 (no change) or median? 0 is safer for "stable".
    # - Z-scores/Stats: Fill with median of the population (robust to outliers).
    
    # Identify column types
    cols = df.columns.tolist()
    if 'N_HC' in cols: cols.remove('N_HC')
    if 'deficit' in cols: cols.remove('deficit') # Don't impute target if missing (shouldn't be)
    
    for col in cols:
        if df[col].isna().sum() > 0:
            if 'flg_' in col or 'n_' in col or 'Cantidad_' in col or 'intensidad_' in col:
                # Flags and counts -> 0
                df[col] = df[col].fillna(0)
            elif 'slope_' in col:
                # Slopes -> 0 (assumption of stability if insufficient data)
                df[col] = df[col].fillna(0)
            else:
                # Continuous stats (mean, min, max, std, Z-scores) -> Median
                # Ensure numeric first (coerce errors like 'N' to NaN)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    # 3. Final Check
    print(f"Final Shape: {df.shape}")
    
    # Save
    print(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_patient_features(
        settings.PROCESSED_DATA_DIR / "tani_patient_features.csv",
        settings.PROCESSED_DATA_DIR / "tani_model_ready.csv"
    )
