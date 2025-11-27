import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from config import settings
from data import preprocessing
from features import build_features
from visualization import visualize

def main():
    print("Loading data...")
    try:
        # Load New Data (2017-2025)
        print("Loading New Data (2017-2025)...")
        df_nut_new = pd.read_excel(settings.DATA_FILE_NEW, sheet_name="DESNUTRICION")
        df_dev_new = pd.read_excel(settings.DATA_FILE_NEW, sheet_name="DESARROLLO")
        
        # Load Old Data (2009-2016)
        print("Loading Old Data (2009-2016)...")
        df_nut_old = pd.read_excel(settings.DATA_FILE_OLD, sheet_name="NUTRICION")
        df_dev_old = pd.read_excel(settings.DATA_FILE_OLD, sheet_name="DESARROLLO")
        
        print(f"Loaded rows: New(Nut={len(df_nut_new)}, Dev={len(df_dev_new)}), Old(Nut={len(df_nut_old)}, Dev={len(df_dev_old)})")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Consolidating datasets...")
    intermediate_dir = settings.INTERMEDIATE_DATA_DIR
    df = preprocessing.consolidar_datasets(
        df_nut_new, df_dev_new, 
        df_nut_old, df_dev_old, 
        output_dir=str(intermediate_dir)
    )
    print(f"Consolidated dataset shape: {df.shape}")

    print("Preprocessing data...")
    df = preprocessing.limpiar_pacientes(df)
    
    # Save preprocessed v1
    df.to_csv(intermediate_dir / "tani_preprocessed_1.csv", index=False)
    
    print("Building features...")
    if 'Edad' in df.columns:
        df['edad_meses'] = df['Edad'].apply(build_features.edad_a_meses)
    
    # Calculate flags
    # Ensure columns exist before calculating
    if '(C) - Cog' in df.columns:
        df['flg_cognitivo'] = build_features.calcular_flg_desarrollo(df, '(C) - Cog')
    if '(L) - Len' in df.columns:
        df['flg_lenguaje'] = build_features.calcular_flg_desarrollo(df, '(L) - Len')
    if '(M) - FF' in df.columns:
        df['flg_motora_fina'] = build_features.calcular_flg_desarrollo(df, '(M) - FF')
    if '(M) - FG' in df.columns:
        df['flg_motora_gruesa'] = build_features.calcular_flg_desarrollo(df, '(M) - FG')
    if '(S) - Soc' in df.columns:
        df['flg_social'] = build_features.calcular_flg_desarrollo(df, '(S) - Soc')
        
    df['flg_alguna'] = build_features.calcular_flg_alguna(df)
    
    # Save final processed
    df.to_csv(intermediate_dir / "tani_preprocessed_final_v2.csv", index=False)

    print("Visualizing...")
    output_path = settings.PROJECT_ROOT / "reports" / "figures" / "eda_histograms.png"
    
    # Select some numeric columns to plot
    numeric_cols = ['edad_meses', 'Peso', 'Talla']
    # Filter only existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if numeric_cols:
        visualize.plot_grid_hist(
            df, 
            cols=numeric_cols, 
            title="EDA Histograms", 
            save_path=str(output_path)
        )
    else:
        print("No numeric columns found to visualize.")
    
    print("Done!")

if __name__ == "__main__":
    main()
