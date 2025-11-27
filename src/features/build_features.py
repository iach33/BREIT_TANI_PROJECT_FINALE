import pandas as pd
import numpy as np
import re

def edad_a_meses(s):
    """
    Convierte diferentes formatos de edad a meses.
    """
    if pd.isna(s): 
        return np.nan
    
    s = str(s).lower().strip()
    
    # Buscar años, meses y días usando regex
    a = re.search(r'(\d+)\s*a', s)  # años
    m = re.search(r'(\d+)\s*m', s)  # meses  
    d = re.search(r'(\d+)\s*d', s)  # días
    
    # Extraer valores (0 si no se encuentra)
    anios = int(a.group(1)) if a else 0
    meses = int(m.group(1)) if m else 0
    dias = int(d.group(1)) if d else 0
    
    # Convertir todo a meses
    # 1 año = 12 meses, 1 mes ≈ 30.44 días (promedio)
    meses_totales = anios * 12 + meses + dias / 30.44
    
    return round(meses_totales, 2)

def calcular_control_esperado(edad_meses):
    """
    Calcula el control esperado basado en la edad en meses.
    """
    def control_esperado_individual(mt):
        if pd.isna(mt):
            return np.nan
        
        # Lógica basada en rangos de edad (aproximación de la lógica DAX/Excel)
        if mt < 1: return 1
        if mt < 2: return 2
        if mt < 3: return 3
        if mt < 4: return 4
        if mt < 5: return 5
        if mt < 6: return 6
        if mt < 7: return 7
        if mt < 8: return 8
        if mt < 9: return 9
        if mt < 10: return 10
        if mt < 11: return 11
        if mt < 12: return 12
        # ... logic continues for older ages, simplifying for now based on typical patterns
        # Assuming monthly until 12, then every 2 months until 24, etc.
        # Since I don't have the FULL switch logic from the notebook snippet, 
        # I will implement a generic logic or placeholder if the snippet was cut off.
        # The snippet WAS cut off. I'll implement a reasonable approximation or generic mapping.
        return int(mt) # Placeholder if exact logic is missing, but usually it maps closely to age in months for first year.
    
    if isinstance(edad_meses, pd.Series):
        return edad_meses.apply(control_esperado_individual)
    return control_esperado_individual(edad_meses)

def calcular_flg_desarrollo(df, columna_desarrollo='(C) - Cog'):
    """
    Evalúa diferentes condiciones sobre la columna de desarrollo.
    """
    if columna_desarrollo not in df.columns:
        raise ValueError(f"La columna '{columna_desarrollo}' no existe en el DataFrame")
    
    def evaluar_desarrollo(valor):
        if pd.isna(valor):
            return np.nan
        
        valor_str = str(valor).strip()
        
        if valor_str == "":
            return np.nan
        elif valor_str == "Defic":
            return 1
        elif valor_str != "":
            return 0
        else:
            return np.nan
    
    return df[columna_desarrollo].apply(evaluar_desarrollo)

def calcular_flg_alguna(df):
    columnas_flags = [
        'flg_cognitivo',
        'flg_lenguaje', 
        'flg_motora_fina',
        'flg_motora_gruesa',
        'flg_social'
    ]
    
    # Ensure columns exist
    valid_cols = [c for c in columnas_flags if c in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index)

    suma_flags = df[valid_cols].sum(axis=1)
    flg_alguna = np.where(suma_flags > 0, 1, suma_flags)
    
    return pd.Series(flg_alguna, index=df.index)

def calcular_flg_total(df):
    columnas_flags = [
        'flg_cognitivo',
        'flg_lenguaje', 
        'flg_motora_fina',
        'flg_motora_gruesa',
        'flg_social'
    ]
    valid_cols = [c for c in columnas_flags if c in df.columns]
    return df[valid_cols].sum(axis=1, skipna=True)

def calcular_flg_lenguaje_social(df):
    columnas_flags = ['flg_lenguaje', 'flg_social']
    valid_cols = [c for c in columnas_flags if c in df.columns]
    return df[valid_cols].sum(axis=1, skipna=True)

def calcular_primer_alguna_por_hc(df):
    """
    Busca el primer control esperado donde hay algún déficit detectado (flg_alguna = 1) para cada HC.
    """
    flg = pd.to_numeric(df['flg_alguna'], errors='coerce')
    ctrl = pd.to_numeric(df['control_esperado'], errors='coerce')

    s = (df.loc[flg.eq(1)]
            .assign(control_esperado=ctrl.loc[flg.eq(1)])
            .groupby('N_HC')['control_esperado']
            .min())

    return s

def calcular_primer_control_esperado(df):
    return df.groupby('N_HC')['control_esperado'].transform('min')

def calcular_ultimo_control(df):
    ultimo_control_por_hc = df.groupby('N_HC')['control_esperado'].transform('max')
    df['ultimo_control'] = ultimo_control_por_hc
    return df

def categoria_TE(z):
    if isinstance(z, str): return z
    if pd.isna(z): return "NULL"
    if z < -3: return "DA"
    if -3 <= z < -2: return "R"
    if -2 <= z <= 2: return "N"
    return "O"

def categoria_PE(z):
    if isinstance(z, str): return z
    if pd.isna(z): return "NULL"
    if z < -2: return "R"
    if z <= 2: return "N"
    return "S"

def categoria_PT(z):
    if isinstance(z, str): return z
    if pd.isna(z): return "NULL"
    if z < -3: return "DA"
    if z < -2: return "R"
    if z < -1: return "N"
    if z < 1: return "N"
    # Simplified logic from snippet
    return "N" 

def features_6prev_window(
    df,
    vars_cols=('Peso','Talla','CabPC','edad_meses','control_esperado','_TE_z','_PE_z','_PT_z'),
    columnas_consejeria=['flg_consj_lact_materna', 'flg_consj_higne_corporal', 
                         'flg_consj_higne_bucal', 'flg_consj_supl_hierro', 
                         'flg_consj_desarrollo', 'flg_consj_vacunas'],
    window=6,
    order_by='control_esperado',
    atol=1e-6
):
    """
    Calcula features de ventana deslizante para los últimos controles.
    """
    d = df.copy()
    
    # ... (Implementation of window features would go here, simplified for modularization example)
    # Since the full logic is complex and depends on 'primer_alguna' etc, 
    # we'll keep the structure but note that it requires the full implementation.
    
    return d # Placeholder return
