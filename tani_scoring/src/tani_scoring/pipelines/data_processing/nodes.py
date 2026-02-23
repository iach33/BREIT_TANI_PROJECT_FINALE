"""
Nodos del pipeline de procesamiento de datos.

Adapta funciones de:
  - src/data/preprocessing.py
  - src/features/build_features.py (funciones de tracking y nacimiento)
  - src/pipelines/01_preprocessing.py (orquestacion de consejeria)

Soporta dos formatos de Excel:
  - Formato "LISTA": una sola hoja con nutricion + desarrollo + consejeria
  - Formato separado: hojas DESNUTRICION + DESARROLLO (y consejeria aparte)
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (adaptados de preprocessing.py)
# ---------------------------------------------------------------------------

def _estandarizar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas comunes."""
    rename_map = {
        "Nº_HC": "N_HC",
        "Nº_Control": "N_Control",
        "N°_HC": "N_HC",
        "N°_Control": "N_Control",
        "Nº_HCL": "N_HC",
    }
    return df.rename(columns=rename_map)


def _parse_z(x, eps=1e-6):
    """Parsea valores mixtos de Z-score (numerico, texto, NA)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    codigos_especiales = {"DA", "DC", "DG", "N", "NB", "NN", "O", "R", "S", "SA"}
    if s in codigos_especiales:
        return s

    sin_dato = {"S/A", "SIN DATO", "NA", "N/A", ""}
    if s in sin_dato:
        return np.nan

    s = s.replace(",", ".").replace(" ", "").replace("+-", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def _categoria_TE(z):
    if isinstance(z, str):
        return z
    if pd.isna(z):
        return "NULL"
    if z < -3:
        return "DA"
    if -3 <= z < -2:
        return "R"
    if -2 <= z <= 2:
        return "N"
    return "O"


def _categoria_PE(z):
    if isinstance(z, str):
        return z
    if pd.isna(z):
        return "NULL"
    if z < -2:
        return "R"
    if z <= 2:
        return "N"
    return "S"


def _categoria_PT(z):
    if isinstance(z, str):
        return z
    if pd.isna(z):
        return "NULL"
    if z < -3:
        return "DA"
    if z < -2:
        return "R"
    return "N"


def _edad_a_meses(s):
    """Convierte formatos de edad '2a 3m 5d' a float meses."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    a = re.search(r"(\d+)\s*a", s)
    m = re.search(r"(\d+)\s*m", s)
    d = re.search(r"(\d+)\s*d", s)
    anios = int(a.group(1)) if a else 0
    meses = int(m.group(1)) if m else 0
    dias = int(d.group(1)) if d else 0
    return round(anios * 12 + meses + dias / 30.44, 2)


def _control_esperado(mt):
    """Mapea edad en meses a numero de control esperado."""
    if pd.isna(mt):
        return np.nan
    if mt < 1:
        return 1
    if mt < 2:
        return 2
    if mt < 3:
        return 3
    if mt < 4:
        return 4
    if mt < 5:
        return 5
    if mt < 6:
        return 6
    if mt < 7:
        return 7
    if mt < 8:
        return 8
    if mt < 9:
        return 9
    if mt < 10:
        return 10
    if mt < 11:
        return 11
    if mt < 12:
        return 12
    return int(mt)


def _calcular_flg_consejeria(valor):
    """Convierte valores de consejeria (boolean, texto, numerico) a 1/0."""
    if pd.isna(valor):
        return np.nan
    # Manejar booleanos nativos de Python/pandas
    if isinstance(valor, (bool, np.bool_)):
        return 1 if valor else 0
    v = str(valor).strip().upper()
    if v in ("VERDADERO", "TRUE", "1"):
        return 1
    if v in ("FALSO", "FALSE", "0"):
        return 0
    return np.nan


# ---------------------------------------------------------------------------
# Nodos del pipeline
# ---------------------------------------------------------------------------


def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """
    Carga datos del Excel mensual.

    Soporta dos formatos:
      - Hoja unica "LISTA" (formato produccion): toda la data consolidada
      - Hojas separadas "DESNUTRICION" + "DESARROLLO" (formato entrenamiento)
    """
    logger.info("Cargando datos desde %s", raw_data_path)
    xl = pd.ExcelFile(raw_data_path)
    sheet_names = xl.sheet_names
    logger.info("Hojas encontradas: %s", sheet_names)

    if "LISTA" in sheet_names:
        # Formato produccion: una sola hoja con todo
        df = pd.read_excel(xl, sheet_name="LISTA")
        df = _estandarizar_cols(df)
        logger.info("Formato LISTA: %d filas x %d columnas", *df.shape)

        # Deduplicar
        keys = ["Fecha", "N_HC", "Tipo_Paciente", "N_Control"]
        available_keys = [k for k in keys if k in df.columns]
        if available_keys:
            n_before = len(df)
            df = df.drop_duplicates(subset=available_keys, keep="first")
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.warning("Filas duplicadas eliminadas: %d", n_dropped)

    elif "DESNUTRICION" in sheet_names and "DESARROLLO" in sheet_names:
        # Formato con hojas separadas
        df_nut = pd.read_excel(xl, sheet_name="DESNUTRICION")
        df_dev = pd.read_excel(xl, sheet_name="DESARROLLO")
        df_nut = _estandarizar_cols(df_nut)
        df_dev = _estandarizar_cols(df_dev)

        keys = ["Fecha", "N_HC", "Tipo_Paciente", "N_Control"]

        # Deduplicar
        mask_nut = df_nut.duplicated(subset=keys, keep=False)
        mask_dev = df_dev.duplicated(subset=keys, keep=False)
        df_nut = df_nut[~mask_nut].copy()
        df_dev = df_dev[~mask_dev].copy()

        # Merge inner
        cols_dev_interest = ["(M) - FG", "(M) - FF", "(C) - Cog", "(L) - Len", "(S) - Soc"]
        cols_dev_merge = keys + [c for c in cols_dev_interest if c in df_dev.columns]
        df = df_nut.merge(df_dev[cols_dev_merge], how="inner", on=keys)
        logger.info("Formato separado: %d filas x %d columnas", *df.shape)
    else:
        # Intentar primera hoja
        df = pd.read_excel(xl, sheet_name=0)
        df = _estandarizar_cols(df)
        logger.warning("Formato no reconocido. Usando primera hoja: %d filas x %d columnas", *df.shape)

    xl.close()
    logger.info("Columnas cargadas: %s", df.columns.tolist())
    return df


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia datos clinicos: Edad_Gestacional, parto_mama, complicacion_parto_mama.
    Adaptado de preprocessing.limpiar_pacientes().
    """
    df = df.copy()

    # Edad_Gestacional -> solo numero
    def _limpiar_edad_gestacional(x):
        if pd.isna(x):
            return np.nan
        s = re.sub(r"[^0-9]", "", str(x).strip().lower())
        try:
            return int(s)
        except ValueError:
            return np.nan

    if "Edad_Gestacional" in df.columns:
        df["Edad_Gestacional"] = df["Edad_Gestacional"].apply(_limpiar_edad_gestacional)

    if "parto_mama" in df.columns:
        df["parto_mama"] = df["parto_mama"].fillna("").replace("", "Normal")

    # Normalizar complicacion
    def _normalizar_complicacion(x):
        if pd.isna(x) or str(x).strip() == "":
            return np.nan
        s = str(x).strip().lower()
        if re.search(r"cesar|cesárea|cesarea|cecarea|casarea", s):
            return "Parto con cesarea"
        if re.search(r"aritmia|arritmia", s):
            return "Parto con arritmia"
        if re.search(r"anemia", s):
            return "Parto con anemia"
        if re.search(r"preeclampsia|preclamsia|preeclamsia", s):
            return "Parto con preeclampsia"
        if re.search(r"hemorragia", s):
            return "Parto con hemorragia"
        if re.search(r"infeccion|urinaria|itu", s):
            return "Parto con infeccion"
        return "Otras complicaciones"

    if "complicacion_parto_mama" in df.columns:
        df["complicacion_parto_mama"] = df["complicacion_parto_mama"].apply(
            _normalizar_complicacion
        )

    logger.info("Limpieza de pacientes completada")
    return df


def calculate_birth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera flags de nacimiento: prematuro, bajo_peso_nacer, macrosomia.
    Adaptado de build_features.calcular_features_nacimiento().
    """
    df = df.copy()
    col_diag = "Diag_Nacimiento"

    if col_diag in df.columns:
        s = df[col_diag].astype(str).str.upper()
        df["flg_prematuro"] = np.where(
            s.str.contains("PRETÉRMINO|PRETERMINO|PREMATURO"), 1, 0
        )
        df["flg_bajo_peso_nacer"] = np.where(
            s.str.contains("BPN|BAJO PESO"), 1, 0
        )
        df["flg_macrosomia"] = np.where(
            s.str.contains("MACROSÓMICO|MACROSOMICO"), 1, 0
        )
    else:
        df["flg_prematuro"] = 0
        df["flg_bajo_peso_nacer"] = 0
        df["flg_macrosomia"] = 0

    logger.info("Features de nacimiento calculadas")
    return df


def convert_age_to_months(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte Edad textual a edad_meses numerico."""
    df = df.copy()
    if "Edad" in df.columns:
        df["edad_meses"] = df["Edad"].apply(_edad_a_meses)
    logger.info("Edad convertida a meses")
    return df


def parse_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parsea columnas P/T, T/E, P/E a valores numericos y categorias.
    Adaptado de preprocessing.parse_z() + build_features.categoria_*.
    """
    df = df.copy()
    z_cols = {"P/T": "PT", "T/E": "TE", "P/E": "PE"}

    for col, suffix in z_cols.items():
        if col in df.columns:
            df[f"_{suffix}_z"] = df[col].apply(_parse_z)
            if suffix == "PT":
                df[f"cat_{suffix}"] = df[f"_{suffix}_z"].apply(_categoria_PT)
            elif suffix == "TE":
                df[f"cat_{suffix}"] = df[f"_{suffix}_z"].apply(_categoria_TE)
            elif suffix == "PE":
                df[f"cat_{suffix}"] = df[f"_{suffix}_z"].apply(_categoria_PE)

    logger.info("Z-scores parseados y categorizados")
    return df


def process_counseling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa columnas de consejeria.

    Soporta dos formatos:
      - Columnas C_* (boolean True/False) del formato LISTA
      - Columnas ya procesadas flg_consj_* (si vienen de un merge previo)

    Mapeo de columnas del formato LISTA:
      C_lact_materna    -> flg_consj_lact_materna
      C_hig_corporal    -> flg_consj_higne_corporal
      C_hig_bucal       -> flg_consj_higne_bucal
      C_supl_hierro     -> flg_consj_supl_hierro
      C_act_dit         -> flg_consj_desarrollo
      C_cuid_vacuna     -> flg_consj_vacunas
    """
    df = df.copy()

    # Mapeo de columnas formato LISTA -> flags del modelo
    col_map = {
        "C_lact_materna": "flg_consj_lact_materna",
        "C_hig_corporal": "flg_consj_higne_corporal",
        "C_hig_bucal": "flg_consj_higne_bucal",
        "C_supl_hierro": "flg_consj_supl_hierro",
        "C_act_dit": "flg_consj_desarrollo",
        "C_cuid_vacuna": "flg_consj_vacunas",
    }

    # Mapeo alternativo: formato con columnas largas (hoja CONSEJERÍAS)
    col_map_alt = {
        "Consejería Lactancia Materna": "flg_consj_lact_materna",
        "Consejería Higiene Corporal": "flg_consj_higne_corporal",
        "Consejería Higiene Bucal": "flg_consj_higne_bucal",
        "Consejería Suplementación con Hierro": "flg_consj_supl_hierro",
        "Consejería Actividades Desarrollo": "flg_consj_desarrollo",
        "Consejería Cuidados post vacunas": "flg_consj_vacunas",
    }

    processed = False

    # Intentar formato LISTA (columnas C_*)
    for col_src, col_dst in col_map.items():
        if col_src in df.columns:
            df[col_dst] = df[col_src].apply(_calcular_flg_consejeria)
            processed = True

    # Intentar formato largo (Consejería *)
    if not processed:
        for col_src, col_dst in col_map_alt.items():
            if col_src in df.columns:
                df[col_dst] = df[col_src].apply(_calcular_flg_consejeria)
                processed = True

    if not processed:
        logger.warning("No se encontraron columnas de consejeria en el dataset")
        for col_dst in col_map.values():
            df[col_dst] = np.nan

    # Intensidad de consejeria
    cols_consj = [c for c in df.columns if c.startswith("flg_consj_")]
    if cols_consj:
        df["intensidad_consejeria"] = df[cols_consj].sum(axis=1)
    else:
        df["intensidad_consejeria"] = 0

    n_flags = sum(1 for c in col_map.values() if c in df.columns and df[c].notna().any())
    logger.info("Consejeria procesada: %d flags activos", n_flags)
    return df


def calculate_control_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula control_esperado, primer_alguna, primer_control_esperado,
    ultimo_control, cant_controles_primer_alguna.

    En scoring, flg_alguna aun no existe en este punto, asi que
    primer_alguna sera NaN y cant_controles usa ultimo_control como umbral.
    """
    df = df.copy()

    # control_esperado
    if "edad_meses" in df.columns and "control_esperado" not in df.columns:
        df["control_esperado"] = df["edad_meses"].apply(_control_esperado)

    # primer_alguna (primer control con flg_alguna == 1)
    if "flg_alguna" in df.columns:
        flg = pd.to_numeric(df["flg_alguna"], errors="coerce")
        ctrl = pd.to_numeric(df["control_esperado"], errors="coerce")
        s_primer = (
            df.loc[flg.eq(1)]
            .assign(control_esperado=ctrl.loc[flg.eq(1)])
            .groupby("N_HC")["control_esperado"]
            .min()
        )
        df["primer_alguna"] = df["N_HC"].map(s_primer)
    else:
        df["primer_alguna"] = np.nan

    # primer_control_esperado
    df["primer_control_esperado"] = df.groupby("N_HC")["control_esperado"].transform(
        "min"
    )

    # ultimo_control
    df["ultimo_control"] = df.groupby("N_HC")["control_esperado"].transform("max")

    # cant_controles_primer_alguna
    d = df.copy()
    for c in ["control_esperado", "primer_alguna", "ultimo_control"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    def _per_hc(g):
        if g["primer_alguna"].notna().any():
            thr = g["primer_alguna"].dropna().iloc[0]
        elif g["ultimo_control"].notna().any():
            thr = g["ultimo_control"].dropna().iloc[0]
        else:
            return np.nan
        return (g["control_esperado"] < thr).sum()

    s_cant = d.groupby("N_HC", group_keys=False).apply(_per_hc, include_groups=False)
    df["cant_controles_primer_alguna"] = df["N_HC"].map(s_cant)

    logger.info("Control tracking calculado")
    return df


def filter_scoreable_population(
    df: pd.DataFrame,
    min_controls: int,
) -> pd.DataFrame:
    """
    Filtra pacientes que pueden ser scored.

    Criterios adaptados para datos de produccion (export parcial, no historial
    completo desde nacimiento):
    - cant_controles_primer_alguna >= min_controls (controles previos al ultimo)
    - ultimo_control >= 19 (paciente tiene al menos ~19 meses de edad)

    NOTA: El filtro de primer_control_esperado in [1,2,3] del pipeline de
    entrenamiento se omite porque en datos de produccion el primer registro
    del export no necesariamente corresponde al primer control desde nacimiento.
    """
    df = df.copy()

    # Asegurar tipo fecha
    if "Fecha" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    mask = (
        (df["cant_controles_primer_alguna"] >= min_controls)
        & (df["ultimo_control"] >= 19)
    )

    nhc_validos = df.loc[mask, "N_HC"].unique()
    df_filtered = df[df["N_HC"].isin(nhc_validos)].copy()

    n_total = df["N_HC"].nunique()
    logger.info(
        "Poblacion filtrada: %d / %d pacientes unicos, %d filas",
        len(nhc_validos),
        n_total,
        len(df_filtered),
    )
    if len(nhc_validos) == 0:
        logger.warning(
            "ATENCION: Ningun paciente paso el filtro. "
            "Verificar que la data tenga pacientes con suficiente historial "
            "(controles >= %d, ultimo_control >= 19)",
            min_controls,
        )
    return df_filtered
