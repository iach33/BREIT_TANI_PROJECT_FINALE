import pandas as pd
import numpy as np
import os
from pathlib import Path
from config import settings

def cargar_tablas_oms():
    """
    Carga las tablas OMS desde archivos Excel ubicados en settings.EXTERNAL_DATA_DIR.
    Devuelve un diccionario con los DataFrames cargados.
    """
    path_base = settings.EXTERNAL_DATA_DIR
    
    archivos_oms = {
        'wfa_boys': "wfa_Boys.xlsx",
        'wfa_girls': "wfa_Girls.xlsx",
        'lhfa_boys': "lhfa_Boys.xlsx",
        'lhfa_girls': "lhfa_Girls.xlsx",
        'wfl_boys': "wfl_boys.xlsx",
        'wfl_girls': "wfl_girls.xlsx",
        'wfh_boys': "wfh_boys.xlsx",
        'wfh_girls': "wfh_girls.xlsx",
    }

    tablas = {}
    print(f"Cargando tablas OMS desde {path_base}...")
    
    for clave, archivo in archivos_oms.items():
        ruta = path_base / archivo

        if not ruta.exists():
            print(f"⚠️  {clave}: archivo no encontrado → {ruta}")
            continue

        try:
            df = pd.read_excel(ruta)

            # Validación mínima de columnas OMS
            if not {'L', 'M', 'S'}.issubset(df.columns):
                print(f"❌ {archivo}: faltan columnas L, M o S")
                continue

            tablas[clave] = df

        except Exception as e:
            print(f"❌ Error al cargar {archivo}: {e}")

    print(f"✓ Tablas cargadas: {len(tablas)}/{len(archivos_oms)}")
    return tablas

def calcular_zscore_lms(valor, L, M, S):
    """
    Calcula Z-score usando el método LMS de la OMS
    """
    if pd.isna(valor) or pd.isna(L) or pd.isna(M) or pd.isna(S):
        return np.nan

    try:
        valor = float(valor)
        L = float(L)
        M = float(M)
        S = float(S)
    except (ValueError, TypeError):
        return np.nan

    if L != 0:
        zscore = (((valor / M) ** L) - 1) / (L * S)
    else:
        zscore = np.log(valor / M) / S

    return zscore

def interpolar_lms(edad_o_talla, sexo, tabla, col_referencia='Month'):
    """
    Interpola valores LMS para edad o talla específica
    """
    if pd.isna(edad_o_talla):
        return np.nan, np.nan, np.nan

    try:
        edad_o_talla = float(edad_o_talla)
    except (ValueError, TypeError):
        return np.nan, np.nan, np.nan

    df_sexo = tabla.copy()

    # Buscar columna de referencia
    if col_referencia not in df_sexo.columns:
        col_alternativas = {
            'Month': ['Month', 'Age', 'Months'],
            'Length': ['Length', 'Height', 'Lengthcm']
        }

        for col_alt in col_alternativas.get(col_referencia, [col_referencia]):
            if col_alt in df_sexo.columns:
                col_referencia = col_alt
                break

    if col_referencia not in df_sexo.columns:
        return np.nan, np.nan, np.nan

    # Ensure reference column is numeric
    df_sexo[col_referencia] = pd.to_numeric(df_sexo[col_referencia], errors='coerce')
    df_sexo = df_sexo.dropna(subset=[col_referencia])

    df_sexo = df_sexo.sort_values(col_referencia)
    ref_vals = df_sexo[col_referencia].values

    if len(ref_vals) == 0:
        return np.nan, np.nan, np.nan

    if edad_o_talla < ref_vals.min():
        edad_o_talla = ref_vals.min()
    elif edad_o_talla > ref_vals.max():
        edad_o_talla = ref_vals.max()

    try:
        L = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo['L'])
        M = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo['M'])
        S = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo['S'])
        return L, M, S
    except:
        return np.nan, np.nan, np.nan

def calcular_zscores_oms(df, tablas):
    """
    Calcula Z-scores para peso-edad, talla-edad y peso-talla
    """
    df_result = df.copy()

    # Inicializar columnas
    df_result['zscore_peso_edad'] = np.nan
    df_result['zscore_talla_edad'] = np.nan
    df_result['zscore_peso_talla'] = np.nan

    print("Calculando Z-scores OMS...")
    total = len(df_result)
    
    # Optimización: Vectorizar en lugar de iterrows para velocidad
    # Sin embargo, la interpolación es compleja de vectorizar completamente con np.interp sobre grupos.
    # Mantendremos el enfoque iterativo pero optimizado o usaremos apply con cuidado.
    # Dado el tamaño (miles de filas), iterrows es lento.
    # Mejor aproximación: Agrupar por sexo y aplicar función.
    
    # Para mantener fidelidad al código del usuario, usaremos una versión optimizada del loop
    # o aplicaremos la función fila por fila si es necesario, pero vectorizar es mejor.
    
    # Vamos a usar apply que es un poco más rápido que iterrows, o un loop simple.
    # El código del usuario usa iterrows. Lo mantendré pero añadiré prints de progreso.
    
    # NOTA: Para producción con 400k filas, iterrows es muy lento.
    # Voy a intentar optimizar cargando las tablas en memoria y haciendo lookups.
    
    # Pero para seguir instrucciones "tal cual", implemento la lógica.
    # Voy a usar un enfoque híbrido: Diccionarios de interpolación para acelerar.
    
    # Pre-computar interpoladores podría ser complejo.
    # Usaremos el código del usuario pero adaptado a una función que recibe una fila.
    
    def procesar_fila(row):
        sexo = row['Sexo']
        edad = row['edad_meses']
        peso = row['Peso']
        talla = row['Talla']
        
        res = {'z_pe': np.nan, 'z_te': np.nan, 'z_pt': np.nan}
        
        if sexo == 'M':
            t_wfa = tablas.get('wfa_boys')
            t_lhfa = tablas.get('lhfa_boys')
            t_wfh = tablas.get('wfh_boys')
        else:
            t_wfa = tablas.get('wfa_girls')
            t_lhfa = tablas.get('lhfa_girls')
            t_wfh = tablas.get('wfh_girls')
            
        # 1. Peso/Edad
        if pd.notna(peso) and pd.notna(edad) and t_wfa is not None:
            L, M, S = interpolar_lms(edad, sexo, t_wfa, 'Month')
            res['z_pe'] = calcular_zscore_lms(peso, L, M, S)
            
        # 2. Talla/Edad
        if pd.notna(talla) and pd.notna(edad) and t_lhfa is not None:
            L, M, S = interpolar_lms(edad, sexo, t_lhfa, 'Month')
            res['z_te'] = calcular_zscore_lms(talla, L, M, S)
            
        # 3. Peso/Talla
        if pd.notna(peso) and pd.notna(talla) and t_wfh is not None:
            L, M, S = interpolar_lms(talla, sexo, t_wfh, 'Length')
            res['z_pt'] = calcular_zscore_lms(peso, L, M, S)
            
        return pd.Series([res['z_pe'], res['z_te'], res['z_pt']])

    # Aplicar
    # Advertencia: Esto puede ser lento. Si el usuario tiene prisa, deberíamos vectorizar.
    # Pero la interpolación depende del valor exacto de edad/talla.
    
    # Usaremos apply con axis=1
    print(f"Procesando {total} registros (esto puede tardar)...")
    cols_z = df_result.apply(procesar_fila, axis=1)
    cols_z.columns = ['zscore_peso_edad', 'zscore_talla_edad', 'zscore_peso_talla']
    
    df_result.update(cols_z)
    
    print("✓ Cálculo Z-scores OMS completado")
    return df_result
