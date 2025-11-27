import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from config import settings
from data import preprocessing
from features import build_features

def main():
    print("=== Starting Preprocessing Pipeline ===")
    
    # 1. Loading Data
    print("\n[1/4] Loading data...")
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

    # 2. Consolidation
    print("\n[2/4] Consolidating datasets...")
    intermediate_dir = settings.INTERMEDIATE_DATA_DIR
    df = preprocessing.consolidar_datasets(
        df_nut_new, df_dev_new, 
        df_nut_old, df_dev_old, 
        output_dir=str(intermediate_dir)
    )
    print(f"Consolidated dataset shape: {df.shape}")

    # 3. Preprocessing / Cleaning
    print("\n[3/4] Cleaning data...")
    df = preprocessing.limpiar_pacientes(df)
    
    # Birth Features
    print("Calculating birth features...")
    df_birth = build_features.calcular_features_nacimiento(df)
    df = pd.concat([df, df_birth], axis=1)
    
    # Save intermediate preprocessed
    df.to_csv(intermediate_dir / "tani_preprocessed_step1.csv", index=False)
    
    # 4. Feature Engineering
    print("\n[4/4] Building features...")
    if 'Edad' in df.columns:
        df['edad_meses'] = df['Edad'].apply(build_features.edad_a_meses)
        
    # Anemia Flag
    print("Calculating anemia flag...")
    df['flg_anemia'] = build_features.calcular_flg_anemia(df)
    
    # Calculate development flags
    print("Calculating development flags...")
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
    df['flg_total'] = build_features.calcular_flg_total(df)
    df['flg_lenguaje_social'] = build_features.calcular_flg_lenguaje_social(df)

    # Process Nutritional Z-scores
    print("Processing nutritional Z-scores...")
    z_cols = {'P/T': 'PT', 'T/E': 'TE', 'P/E': 'PE'}
    for col, suffix in z_cols.items():
        if col in df.columns:
            # Parse Z-score (mixed types)
            df[f'_{suffix}_z'] = df[col].apply(preprocessing.parse_z)
            
            # Categorize
            if suffix == 'PT':
                df[f'cat_{suffix}'] = df[f'_{suffix}_z'].apply(build_features.categoria_PT)
            elif suffix == 'TE':
                df[f'cat_{suffix}'] = df[f'_{suffix}_z'].apply(build_features.categoria_TE)
            elif suffix == 'PE':
                df[f'cat_{suffix}'] = df[f'_{suffix}_z'].apply(build_features.categoria_PE)
    
    # Detailed Nutritional Features
    print("Calculating detailed nutritional features...")
    df_nut_det = build_features.calcular_features_nutricionales_detalladas(df)
    df = pd.concat([df, df_nut_det], axis=1)
    
    # --- Additional Features for Target Population ---
    print("Calculating additional features for filtering...")
    
    # 1. Control Esperado
    if 'control_esperado' not in df.columns:
        df['control_esperado'] = build_features.calcular_control_esperado(df['edad_meses'])
        
    # 2. Primer Alguna (First deficit)
    s_primer_alguna = build_features.calcular_primer_alguna_por_hc(df)
    df['primer_alguna'] = df['N_HC'].map(s_primer_alguna)
    
    # 3. Primer Control Esperado
    df['primer_control_esperado'] = build_features.calcular_primer_control_esperado(df)
    
    # 4. Ultimo Control
    df = build_features.calcular_ultimo_control(df)
    
    # 5. Cantidad Controles antes de primer alguna
    s_cant = build_features.cant_controles_primer_alguna_por_hc(df)
    df['cant_controles_primer_alguna'] = df['N_HC'].map(s_cant)
    
    # --- Filtering Target Population ---
    print("\n[5/5] Filtering Target Population (2023+, controls 1-3)...")
    df_filtered = preprocessing.filtrar_poblacion_objetivo(df)
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Save final processed dataset
    output_file = settings.PROCESSED_DATA_DIR / "tani_analytical_dataset.csv"
    print(f"\nSaving final dataset to: {output_file}")
    
    # Ensure directory exists
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df_filtered.to_csv(output_file, index=False)
    
    # --- 6. Counseling Data Integration & Window Features ---
    print("\n[6/6] Processing Counseling Data & Window Features...")
    try:
        print("Loading Counseling Data...")
        # Load only CONSEJERÍAS sheet as per notebook logic for merging
        df_consejerias = pd.read_excel(settings.DATA_FILE_UBIGEO, sheet_name="CONSEJERÍAS")
        
        # Rename columns
        df_consejerias.rename(columns={
            'Nº_HCL': 'N_HC',
            'Nº_Control': 'N_Control'
        }, inplace=True)
        
        # Deduplicate
        keys = ['Fecha','N_HC', 'N_Control']
        mask_dup = df_consejerias.duplicated(subset=keys, keep=False)
        df_consejerias = df_consejerias[~mask_dup].copy()
        
        # Calculate Counseling Flags
        counseling_cols = [
            'Consejería Lactancia Materna', 'Consejería Higiene Corporal',
            'Consejería Higiene Bucal', 'Consejería Suplementación con Hierro',
            'Consejería Actividades Desarrollo', 'Consejería Cuidados post vacunas'
        ]
        
        # Map to flag names expected by features_6prev_window
        flag_map = {
            'Consejería Lactancia Materna': 'flg_consj_lact_materna',
            'Consejería Higiene Corporal': 'flg_consj_higne_corporal',
            'Consejería Higiene Bucal': 'flg_consj_higne_bucal',
            'Consejería Suplementación con Hierro': 'flg_consj_supl_hierro',
            'Consejería Actividades Desarrollo': 'flg_consj_desarrollo',
            'Consejería Cuidados post vacunas': 'flg_consj_vacunas'
        }
        
        for col, flag_name in flag_map.items():
            if col in df_consejerias.columns:
                df_consejerias[flag_name] = build_features.calcular_flg_consejeria(df_consejerias, col)
        
        # Select columns for merge
        cols_merge_keys = ['Fecha', 'N_HC', 'Tipo_Paciente']
        cols_flags = list(flag_map.values())
        cols_to_use = cols_merge_keys + [c for c in cols_flags if c in df_consejerias.columns]
        
        # Merge with filtered dataset
        # Note: We merge on df_filtered because we only care about the target population
        print("Merging counseling data...")
        df_merged = df_filtered.merge(
            df_consejerias[cols_to_use],
            how='left',
            on=cols_merge_keys
        )
        
        # Calculate Counseling Intensity
        df_merged['intensidad_consejeria'] = build_features.calcular_intensidad_consejeria(df_merged)
        
        # Calculate Window Features
        print("Calculating Window Features (this may take a while)...")
        df_features = build_features.features_6prev_window(df_merged)
        
        # Calculate First Year Features
        print("Calculating First Year Features...")
        df_first_year = build_features.calcular_features_primer_anio(df_merged)
        
        # Merge First Year Features into Patient Features
        if not df_first_year.empty:
            df_features = df_features.merge(df_first_year, on='N_HC', how='left')
            # Fill NaNs for patients with no first year data (unlikely if filtered correctly, but possible)
            fill_cols = ['n_controles_primer_anio', 'flg_desnutricion_primer_anio', 'flg_anemia_primer_anio']
            for c in fill_cols:
                if c in df_features.columns:
                    df_features[c] = df_features[c].fillna(0)
                    
        # Calculate Milestone Features (12m, 24m, 36m)
        print("Calculating Milestone Features (12m, 24m, 36m)...")
        df_hitos = build_features.calcular_features_hitos(df_merged, hitos_meses=[12, 24, 36])
        
        if not df_hitos.empty:
            df_features = df_features.merge(df_hitos, on='N_HC', how='left')
            # Note: We do NOT fill NaNs here yet, the cleaning script will handle them (imputation)
        
        # Save Patient Features
        output_features = settings.PROCESSED_DATA_DIR / "tani_patient_features.csv"
        print(f"Saving patient features to: {output_features}")
        df_features.to_csv(output_features, index=False)
        
        # 7. Clean for Modeling
        print("\n[7/7] Cleaning for Modeling...")
        from data.clean_patient_features import clean_patient_features
        output_model_ready = settings.PROCESSED_DATA_DIR / "tani_model_ready.csv"
        clean_patient_features(output_features, output_model_ready)
        
    except Exception as e:
        print(f"Error processing counseling/window features: {e}")
        # Don't fail the whole pipeline if this part fails, but log it
        import traceback
        traceback.print_exc()

    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
