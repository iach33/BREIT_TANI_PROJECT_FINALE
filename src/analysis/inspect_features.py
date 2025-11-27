import pandas as pd
import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from config import settings

def inspect_dataset(file_path, name):
    print(f"\n=== Inspecting {name} ===")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Shape: {df.shape}")
    
    # Columns to check
    new_features = [
        'flg_anemia', 'flg_prematuro', 'flg_bajo_peso_nacer', 'flg_macrosomia',
        'flg_desnutricion_cronica', 'flg_desnutricion_aguda', 'flg_sobrepeso', 'flg_obesidad',
        'intensidad_consejeria',
        # Aggregated Window Features
        'flg_anemia_window', 'flg_prematuro', 'flg_bajo_peso_nacer', 'flg_macrosomia',
        'flg_desnutricion_cronica_window', 'flg_desnutricion_aguda_window', 
        'flg_sobrepeso_window', 'flg_obesidad_window', 'intensidad_consejeria_window_sum',
        # First Year Features
        'n_controles_primer_anio', 'flg_desnutricion_primer_anio', 'flg_anemia_primer_anio'
    ]
    
    # Check if columns exist
    existing_cols = [c for c in new_features if c in df.columns]
    
    if not existing_cols:
        print("No new features found in this dataset.")
        return

    print("\n--- Feature Statistics ---")
    for col in existing_cols:
        print(f"\nFeature: {col}")
        print(f"Missing: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")
        print("Value Counts:")
        print(df[col].value_counts(dropna=False).head())

    # Check Tam_hb specifically if flg_anemia is present
    if 'flg_anemia' in existing_cols and 'Tam_hb' in df.columns:
        print("\n--- Tam_hb vs flg_anemia ---")
        print(f"Tam_hb Missing: {df['Tam_hb'].isna().sum()}")
        print(df[['Tam_hb', 'flg_anemia']].head(10))

def main():
    inspect_dataset(settings.PROCESSED_DATA_DIR / "tani_analytical_dataset.csv", "Longitudinal Dataset")
    inspect_dataset(settings.PROCESSED_DATA_DIR / "tani_patient_features.csv", "Patient Features")

if __name__ == "__main__":
    main()
