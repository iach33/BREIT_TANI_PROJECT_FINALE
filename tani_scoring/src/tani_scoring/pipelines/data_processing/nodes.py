"""
Nodos del pipeline de procesamiento de datos.

Adapta funciones de:
  - src/data/preprocessing.py
  - src/features/build_features.py (funciones de tracking y nacimiento)
  - src/pipelines/01_preprocessing.py (orquestacion de consejeria)
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
    """Convierte VERDADERO/FALSO textual a 1/0."""
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().upper()
    if v in ("VERDADERO", "TRUE", "1"):
        return 1
    if v in ("FALSO", "FALSE", "0"):
        return 0
    return np.nan


# ---------------------------------------------------------------------------
# Nodos del pipeline
# ---------------------------------------------------------------------------


def load_nutrition_data(raw_data_path: str) -> pd.DataFrame:
    """Carga la hoja DESNUTRICION del Excel mensual."""
    logger.info("Cargando datos de nutricion desde %s", raw_data_path)
    df = pd.read_excel(raw_data_path, sheet_name="DESNUTRICION")
    df = _estandarizar_cols(df)
    logger.info("Nutricion: %d filas x %d columnas", *df.shape)
    return df


def load_development_data(raw_data_path: str) -> pd.DataFrame:
    """Carga la hoja DESARROLLO del Excel mensual."""
    logger.info("Cargando datos de desarrollo desde %s", raw_data_path)
    df = pd.read_excel(raw_data_path, sheet_name="DESARROLLO")
    df = _estandarizar_cols(df)
    logger.info("Desarrollo: %d filas x %d columnas", *df.shape)
    return df


def merge_nutrition_development(
    df_nutrition: pd.DataFrame,
    df_development: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida nutricion y desarrollo: deduplicar, merge inner en keys comunes.
    Adaptado de preprocessing.consolidar_datasets() (solo parte 'new').
    """
    keys = ["Fecha", "N_HC", "Tipo_Paciente", "N_Control"]

    # Deduplicar
    mask_nut = df_nutrition.duplicated(subset=keys, keep=False)
    mask_dev = df_development.duplicated(subset=keys, keep=False)
    df_nut_clean = df_nutrition[~mask_nut].copy()
    df_dev_clean = df_development[~mask_dev].copy()

    if mask_nut.sum() > 0:
        logger.warning("Nutricion: %d filas duplicadas eliminadas", mask_nut.sum())
    if mask_dev.sum() > 0:
        logger.warning("Desarrollo: %d filas duplicadas eliminadas", mask_dev.sum())

    # Columnas de interes del desarrollo
    cols_dev_interest = ["(M) - FG", "(M) - FF", "(C) - Cog", "(L) - Len", "(S) - Soc"]
    cols_dev_merge = keys + [c for c in cols_dev_interest if c in df_dev_clean.columns]

    df_consolidated = df_nut_clean.merge(
        df_dev_clean[cols_dev_merge],
        how="inner",
        on=keys,
    )
    logger.info("Dataset consolidado: %d filas x %d columnas", *df_consolidated.shape)
    return df_consolidated


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


def calculate_control_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula control_esperado, primer_alguna, primer_control_esperado,
    ultimo_control, cant_controles_primer_alguna.
    Adaptado de build_features.
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

    s_cant = d.groupby("N_HC", group_keys=False).apply(_per_hc)
    df["cant_controles_primer_alguna"] = df["N_HC"].map(s_cant)

    logger.info("Control tracking calculado")
    return df


def filter_scoreable_population(
    df: pd.DataFrame,
    min_controls: int,
) -> pd.DataFrame:
    """
    Filtra pacientes que pueden ser scored.
    Adaptado de preprocessing.filtrar_poblacion_objetivo() para prediccion:
    - Sin filtro de anio (toda la data es reciente en produccion)
    - primer_control_esperado in [1,2,3]
    - cant_controles_primer_alguna >= min_controls
    - ultimo_control >= 19
    """
    df = df.copy()

    # Asegurar tipo fecha
    if "Fecha" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Fecha"]):
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    mask = (
        df["primer_control_esperado"].isin([1, 2, 3])
        & (df["cant_controles_primer_alguna"] >= min_controls)
        & (df["ultimo_control"] >= 19)
    )

    nhc_validos = df.loc[mask, "N_HC"].unique()
    df_filtered = df[df["N_HC"].isin(nhc_validos)].copy()

    logger.info(
        "Poblacion filtrada: %d pacientes unicos, %d filas",
        len(nhc_validos),
        len(df_filtered),
    )
    return df_filtered


def load_and_merge_counseling(
    df: pd.DataFrame,
    counseling_data_path: str,
) -> pd.DataFrame:
    """
    Carga consejeria, calcula flags, merge con dataset filtrado.
    Adaptado de 01_preprocessing.py lineas 140-193.
    """
    df = df.copy()

    logger.info("Cargando datos de consejeria desde %s", counseling_data_path)
    df_cons = pd.read_excel(counseling_data_path, sheet_name="CONSEJERÍAS")
    df_cons = _estandarizar_cols(df_cons)

    # Deduplicar
    keys = ["Fecha", "N_HC", "N_Control"]
    mask_dup = df_cons.duplicated(subset=keys, keep=False)
    df_cons = df_cons[~mask_dup].copy()

    # Flags de consejeria
    flag_map = {
        "Consejería Lactancia Materna": "flg_consj_lact_materna",
        "Consejería Higiene Corporal": "flg_consj_higne_corporal",
        "Consejería Higiene Bucal": "flg_consj_higne_bucal",
        "Consejería Suplementación con Hierro": "flg_consj_supl_hierro",
        "Consejería Actividades Desarrollo": "flg_consj_desarrollo",
        "Consejería Cuidados post vacunas": "flg_consj_vacunas",
    }

    for col, flag_name in flag_map.items():
        if col in df_cons.columns:
            df_cons[flag_name] = df_cons[col].apply(_calcular_flg_consejeria)

    # Columnas para merge
    merge_keys = ["Fecha", "N_HC", "Tipo_Paciente"]
    cols_flags = [v for v in flag_map.values() if v in df_cons.columns]
    cols_to_use = merge_keys + cols_flags
    cols_to_use = [c for c in cols_to_use if c in df_cons.columns]

    df_merged = df.merge(df_cons[cols_to_use], how="left", on=merge_keys)

    # Intensidad de consejeria
    cols_consj = [c for c in df_merged.columns if "flg_consj_" in c]
    if cols_consj:
        df_merged["intensidad_consejeria"] = df_merged[cols_consj].sum(axis=1)
    else:
        df_merged["intensidad_consejeria"] = 0

    logger.info("Datos de consejeria integrados. Shape: %s", df_merged.shape)
    return df_merged
