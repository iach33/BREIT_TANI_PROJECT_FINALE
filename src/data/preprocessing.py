import pandas as pd
import numpy as np
import re

def asignar_codigos_nhc(df, columna_nombre='Nombre_Paciente', columna_nhc='N_HC', codigo_inicial=13000):
    """
    Asigna códigos N_HC únicos a pacientes agrupados por nombre
    """
    df_copy = df.copy()
    
    # Identificar filas sin N_HC
    mask_null = df_copy[columna_nhc].isnull()
    
    # Obtener nombres únicos de pacientes que no tienen N_HC
    nombres_sin_hc = df_copy.loc[mask_null, columna_nombre].unique()
    
    # Crear diccionario de mapeo
    mapeo_nuevos_codigos = {}
    codigo_actual = codigo_inicial
    
    for nombre in nombres_sin_hc:
        mapeo_nuevos_codigos[nombre] = codigo_actual
        codigo_actual += 1
        
    # Función para aplicar el mapeo
    def llenar_nhc(row):
        if pd.isna(row[columna_nhc]):
            return mapeo_nuevos_codigos.get(row[columna_nombre], np.nan)
        return row[columna_nhc]
    
    df_copy[columna_nhc] = df_copy.apply(llenar_nhc, axis=1)
    
    return df_copy

def concatenar_dataframes(df1, df2, resetear_index=True):
    """
    Concatena dos dataframes verticalmente
    """
    print("=== INFORMACIÓN ANTES DE CONCATENAR ===")
    print(f"DataFrame 1: {df1.shape[0]:,} filas x {df1.shape[1]} columnas")
    print(f"DataFrame 2: {df2.shape[0]:,} filas x {df2.shape[1]} columnas")

    # Verificar que tengan las mismas columnas
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)
    
    if cols_df1 != cols_df2:
        print("ADVERTENCIA: Los dataframes tienen columnas diferentes")
        print(f"Columnas solo en DF1: {cols_df1 - cols_df2}")
        print(f"Columnas solo en DF2: {cols_df2 - cols_df1}")
    
    df_concat = pd.concat([df1, df2], axis=0, ignore_index=resetear_index)
    
    print("=== INFORMACIÓN DESPUÉS DE CONCATENAR ===")
    print(f"DataFrame Resultante: {df_concat.shape[0]:,} filas x {df_concat.shape[1]} columnas")
    
    return df_concat

def limpiar_pacientes(df):
    """
    Limpia y normaliza columnas de información clínica del parto y nacimiento.
    """
    df = df.copy()

    # Edad_Gestacional -> extraer solo número de semanas
    def limpiar_edad_gestacional(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        s = re.sub(r'[^0-9]', '', s)  # conservar solo números
        try:
            return int(s)
        except ValueError:
            return np.nan

    if 'Edad_Gestacional' in df.columns:
        df['Edad_Gestacional'] = df['Edad_Gestacional'].apply(limpiar_edad_gestacional)

    # Tipo_parto_mama -> valores vacíos o nulos a "Normal"
    if 'parto_mama' in df.columns:
        df['parto_mama'] = df['parto_mama'].fillna('').replace('', 'Normal')

    # Complicacion_parto_mama -> agrupar categorías similares
    def normalizar_complicacion(x):
        if pd.isna(x) or str(x).strip() == '':
            return np.nan
        
        s = str(x).strip().lower()

        if re.search(r'cesar|cesárea|cesarea|cecarea|casarea', s):
            return 'Parto con cesarea'
        if re.search(r'aritmia|arritmia', s):
            return 'Parto con arritmia'
        if re.search(r'anemia', s):
            return 'Parto con anemia'
        if re.search(r'preeclampsia|preclamsia|preeclamsia', s):
            return 'Parto con preeclampsia'
        if re.search(r'hemorragia', s):
            return 'Parto con hemorragia'
        if re.search(r'infeccion|urinaria|itu', s):
            return 'Parto con infeccion'
        
        return 'Otras complicaciones'

    if 'complicacion_parto_mama' in df.columns:
        df['complicacion_parto_mama'] = df['complicacion_parto_mama'].apply(normalizar_complicacion)

    return df

def tipificar_parto_bebe(diagnostico):
    if pd.isna(diagnostico):
        return 'Sin Diagnóstico'
    
    diagnostico = str(diagnostico).upper()
    
    if 'CESAREA' in diagnostico or 'CESÁREA' in diagnostico:
        return 'Cesárea'
    elif 'NORMAL' in diagnostico or 'EUTOCICO' in diagnostico:
        return 'Parto Normal'
    elif 'PREMATURO' in diagnostico or 'PRETÉRMINO' in diagnostico:
        return 'Prematuro'
    else:
        return 'Otros'

def tipificar_complicacion_parto_mama(texto):
    if pd.isna(texto):
        return 'Sin Complicación'
    
    texto = str(texto).upper()
    
    if 'PREECLAMPSIA' in texto:
        return 'Preeclampsia'
    elif 'INFECCION' in texto or 'ITU' in texto:
        return 'Infección'
    elif 'HEMORRAGIA' in texto:
        return 'Hemorragia'
    elif 'ANEMIA' in texto:
        return 'Anemia'
    else:
        return 'Otras Complicaciones'

def parse_z(x, eps=1e-6):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # Casos especiales que se mantienen tal cual
    codigos_especiales = {"DA", "DC", "DG", "N", "NB", "NN", "O", "R", "S", "SA"}
    if s in codigos_especiales:
        return s  # mantener como está
        
    sin_dato_set = {"S/A", "SIN DATO", "NA", "N/A", ""}
    if s in sin_dato_set:
        return np.nan

    # Normalizar separadores y signos "raros"
    s = s.replace(",", ".")          # coma -> punto decimal
    s = s.replace(" ", "")           # espacios
    s = s.replace("+-", "")          # +- -> 
    
    # Intentar convertir a float
    try:
        return float(s)
    except ValueError:
        return np.nan

def detectar_outliers_iqr(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

def consolidar_datasets(df_nut_new, df_dev_new, df_nut_old, df_dev_old, output_dir=None):
    """
    Realiza la consolidación de los datasets de nutrición y desarrollo (antiguo y nuevo).
    Retorna el dataframe consolidado final y guarda intermedios si output_dir se especifica.
    """
    
    # --- 0. Estandarización de nombres de columnas ---
    def estandarizar_cols(df):
        rename_map = {
            'Nº_HC': 'N_HC',
            'Nº_Control': 'N_Control',
            'N°_HC': 'N_HC',
            'N°_Control': 'N_Control'
        }
        return df.rename(columns=rename_map)

    df_nut_new = estandarizar_cols(df_nut_new)
    df_dev_new = estandarizar_cols(df_dev_new)
    df_nut_old = estandarizar_cols(df_nut_old)
    df_dev_old = estandarizar_cols(df_dev_old)

    # --- 1. Deduplicación ---
    # Keys para deduplicación
    keys_new = ['Fecha', 'N_HC', 'Tipo_Paciente', 'N_Control']
    keys_old = ['Fecha', 'Nombre_Paciente', 'Tipo_Paciente', 'N_Control']
    
    # Identificar duplicados
    mask_dup_nut_new = df_nut_new.duplicated(subset=keys_new, keep=False)
    mask_dup_dev_new = df_dev_new.duplicated(subset=keys_new, keep=False)
    mask_dup_nut_old = df_nut_old.duplicated(subset=keys_old, keep=False)
    mask_dup_dev_old = df_dev_old.duplicated(subset=keys_old, keep=False)
    
    # Guardar duplicados si se requiere
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        df_nut_new[mask_dup_nut_new].to_csv(f"{output_dir}/TANI_nutricion_casos_duplicados_base_actual_2017_2025.csv", sep=";", index=False)
        df_dev_new[mask_dup_dev_new].to_csv(f"{output_dir}/TANI_desarrollo_casos_duplicados_base_actual_2017_2025.csv", sep=";", index=False)
        df_nut_old[mask_dup_nut_old].to_csv(f"{output_dir}/TANI_nutricion_casos_duplicados_base_pre_2017.csv", sep=";", index=False)
        df_dev_old[mask_dup_dev_old].to_csv(f"{output_dir}/TANI_desarrollo_casos_duplicados_base_pre_2017.csv", sep=";", index=False)

    # Filtrar únicos
    df_nut_new_clean = df_nut_new[~mask_dup_nut_new].copy()
    df_dev_new_clean = df_dev_new[~mask_dup_dev_new].copy()
    df_nut_old_clean = df_nut_old[~mask_dup_nut_old].copy()
    df_dev_old_clean = df_dev_old[~mask_dup_dev_old].copy()
    
    # --- 2. Merge (Nutrición + Desarrollo) ---
    
    # Reducir columnas de desarrollo para evitar duplicidad, manteniendo las claves y las de interés
    # (Asumiendo columnas de interés basándonos en el notebook)
    cols_dev_interest = ['(M) - FG', '(M) - FF', '(C) - Cog', '(L) - Len', '(S) - Soc']
    
    # Merge Old
    # Asegurar que las columnas existen
    cols_dev_old_merge = keys_old + [c for c in cols_dev_interest if c in df_dev_old_clean.columns]
    df_consolidado_old = df_nut_old_clean.merge(
        df_dev_old_clean[cols_dev_old_merge],
        how='inner',
        on=keys_old
    )
    
    # Merge New
    cols_dev_new_merge = keys_new + [c for c in cols_dev_interest if c in df_dev_new_clean.columns]
    df_consolidado_new = df_nut_new_clean.merge(
        df_dev_new_clean[cols_dev_new_merge],
        how='inner',
        on=keys_new
    )
    
    # --- 3. Concatenación Final ---
    # Alinear columnas antes de concatenar (usando las columnas del nuevo como referencia o intersección)
    # En el notebook usan vars_finales = df_consolidado.columns.tolist() y luego df_consolidado_old[vars_finales]
    # Aquí haremos una concatenación simple y pandas alineará por nombre.
    
    df_final = pd.concat([df_consolidado_old, df_consolidado_new], ignore_index=True)
    
    if output_dir:
        df_final.to_csv(f"{output_dir}/tani_consolidado_v1.csv", index=False)
        
    return df_final

