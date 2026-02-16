"""
Nodos del pipeline de ingenieria de features.

Adapta funciones de:
  - src/features/build_features.py (flags, window, primer anio, hitos)
  - src/features/oms_zscores.py (z-scores OMS)
  - src/data/clean_patient_features.py (imputacion)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calcular_flg_desarrollo(df: pd.DataFrame, columna: str) -> pd.Series:
    """Evalua columna de desarrollo: 'Defic' -> 1, otro texto -> 0, vacio -> NaN."""
    if columna not in df.columns:
        return pd.Series(np.nan, index=df.index)

    def _eval(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip()
        if v == "":
            return np.nan
        if v == "Defic":
            return 1
        return 0

    return df[columna].apply(_eval)


def _calculate_slope(series, x_values=None):
    """Pendiente de regresion lineal simple sobre una serie temporal."""
    series_clean = series.dropna()
    if len(series_clean) < 2:
        return np.nan
    x = np.arange(len(series_clean)) if x_values is None else x_values[: len(series_clean)]
    y = series_clean.values
    try:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den == 0:
            return np.nan
        return num / den
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# OMS Z-scores (adaptado de src/features/oms_zscores.py)
# ---------------------------------------------------------------------------


def _calcular_zscore_lms(valor, L, M, S):
    """Z-score con metodo LMS de la OMS."""
    if pd.isna(valor) or pd.isna(L) or pd.isna(M) or pd.isna(S):
        return np.nan
    try:
        valor, L, M, S = float(valor), float(L), float(M), float(S)
    except (ValueError, TypeError):
        return np.nan
    if L != 0:
        return (((valor / M) ** L) - 1) / (L * S)
    return np.log(valor / M) / S


def _interpolar_lms(edad_o_talla, tabla, col_referencia="Month"):
    """Interpola valores LMS para una edad o talla especifica."""
    if pd.isna(edad_o_talla):
        return np.nan, np.nan, np.nan
    try:
        edad_o_talla = float(edad_o_talla)
    except (ValueError, TypeError):
        return np.nan, np.nan, np.nan

    df_sexo = tabla.copy()

    # Buscar columna de referencia
    if col_referencia not in df_sexo.columns:
        alt_map = {
            "Month": ["Month", "Age", "Months"],
            "Length": ["Length", "Height", "Lengthcm"],
        }
        for alt in alt_map.get(col_referencia, [col_referencia]):
            if alt in df_sexo.columns:
                col_referencia = alt
                break

    if col_referencia not in df_sexo.columns:
        return np.nan, np.nan, np.nan

    df_sexo[col_referencia] = pd.to_numeric(df_sexo[col_referencia], errors="coerce")
    df_sexo = df_sexo.dropna(subset=[col_referencia]).sort_values(col_referencia)
    ref_vals = df_sexo[col_referencia].values

    if len(ref_vals) == 0:
        return np.nan, np.nan, np.nan

    edad_o_talla = np.clip(edad_o_talla, ref_vals.min(), ref_vals.max())

    try:
        L = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo["L"])
        M = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo["M"])
        S = np.interp(edad_o_talla, df_sexo[col_referencia], df_sexo["S"])
        return L, M, S
    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Nodos del pipeline
# ---------------------------------------------------------------------------


def calculate_development_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula flags de deficit de desarrollo por dominio.
    Adaptado de build_features: flg_cognitivo, lenguaje, motora, social, alguna, total.
    """
    df = df.copy()

    col_map = {
        "(C) - Cog": "flg_cognitivo",
        "(L) - Len": "flg_lenguaje",
        "(M) - FF": "flg_motora_fina",
        "(M) - FG": "flg_motora_gruesa",
        "(S) - Soc": "flg_social",
    }

    for col, flag in col_map.items():
        df[flag] = _calcular_flg_desarrollo(df, col)

    # flg_alguna: al menos un deficit
    flag_cols = ["flg_cognitivo", "flg_lenguaje", "flg_motora_fina", "flg_motora_gruesa", "flg_social"]
    valid = [c for c in flag_cols if c in df.columns]
    if valid:
        suma = df[valid].sum(axis=1)
        df["flg_alguna"] = np.where(suma > 0, 1, suma)
    else:
        df["flg_alguna"] = np.nan

    # flg_total: suma de flags
    df["flg_total"] = df[valid].sum(axis=1, skipna=True) if valid else 0

    # flg_lenguaje_social
    ls_cols = [c for c in ["flg_lenguaje", "flg_social"] if c in df.columns]
    df["flg_lenguaje_social"] = df[ls_cols].sum(axis=1, skipna=True) if ls_cols else 0

    logger.info("Flags de desarrollo calculados")
    return df


def calculate_anemia_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag de anemia basado en Hb. Criterio OMS ninos 6-59 meses: Hb < 11 g/dL.
    Adaptado de build_features.calcular_flg_anemia().
    """
    df = df.copy()
    col_hb = "Tam_hb"
    col_edad = "edad_meses"

    if col_hb not in df.columns or col_edad not in df.columns:
        df["flg_anemia"] = np.nan
        return df

    def _eval(row):
        hb, edad = row[col_hb], row[col_edad]
        if pd.isna(hb) or pd.isna(edad):
            return np.nan
        if edad < 6:
            return np.nan
        try:
            hb_val = float(hb)
            if hb_val < 4 or hb_val > 20:
                return np.nan
            return 1 if hb_val < 11.0 else 0
        except ValueError:
            return np.nan

    df["flg_anemia"] = df.apply(_eval, axis=1)
    logger.info("Flag de anemia calculado")
    return df


def calculate_nutritional_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags nutricionales basados en Z-scores: DC, DA, sobrepeso, obesidad.
    Adaptado de build_features.calcular_features_nutricionales_detalladas().
    """
    df = df.copy()

    if "_TE_z" in df.columns:
        z = pd.to_numeric(df["_TE_z"], errors="coerce")
        df["flg_desnutricion_cronica"] = np.where(z.isna(), np.nan, np.where(z < -2, 1, 0))
    else:
        df["flg_desnutricion_cronica"] = np.nan

    if "_PT_z" in df.columns:
        z = pd.to_numeric(df["_PT_z"], errors="coerce")
        df["flg_desnutricion_aguda"] = np.where(z.isna(), np.nan, np.where(z < -2, 1, 0))
        df["flg_sobrepeso"] = np.where(z.isna(), np.nan, np.where(z > 2, 1, 0))
        df["flg_obesidad"] = np.where(z.isna(), np.nan, np.where(z > 3, 1, 0))
    else:
        df["flg_desnutricion_aguda"] = np.nan
        df["flg_sobrepeso"] = np.nan
        df["flg_obesidad"] = np.nan

    logger.info("Flags nutricionales calculados")
    return df


def calculate_oms_zscores(
    df: pd.DataFrame,
    oms_tables_path: str,
) -> pd.DataFrame:
    """
    Calcula Z-scores OMS interpolando tablas de referencia LMS.
    Adaptado de oms_zscores.calcular_zscores_oms().

    Args:
        df: DataFrame con columnas Peso, Talla, edad_meses, Sexo.
        oms_tables_path: Directorio con los 8 archivos Excel de tablas OMS.
    """
    df = df.copy()
    path_base = Path(oms_tables_path)

    archivos = {
        "wfa_boys": "wfa_Boys.xlsx",
        "wfa_girls": "wfa_Girls.xlsx",
        "lhfa_boys": "lhfa_Boys.xlsx",
        "lhfa_girls": "lhfa_Girls.xlsx",
        "wfl_boys": "wfl_boys.xlsx",
        "wfl_girls": "wfl_girls.xlsx",
        "wfh_boys": "wfh_boys.xlsx",
        "wfh_girls": "wfh_girls.xlsx",
    }

    tablas = {}
    for clave, archivo in archivos.items():
        ruta = path_base / archivo
        if not ruta.exists():
            logger.warning("Tabla OMS no encontrada: %s", ruta)
            continue
        try:
            t = pd.read_excel(ruta)
            if {"L", "M", "S"}.issubset(t.columns):
                tablas[clave] = t
            else:
                logger.warning("Tabla %s sin columnas L/M/S", archivo)
        except Exception as e:
            logger.error("Error cargando %s: %s", archivo, e)

    if not tablas:
        logger.warning("No se cargaron tablas OMS, saltando calculo")
        df["zscore_peso_edad"] = np.nan
        df["zscore_talla_edad"] = np.nan
        df["zscore_peso_talla"] = np.nan
        return df

    logger.info("Tablas OMS cargadas: %d/8", len(tablas))

    def _procesar_fila(row):
        sexo = row.get("Sexo")
        edad = row.get("edad_meses")
        peso = row.get("Peso")
        talla = row.get("Talla")

        res = [np.nan, np.nan, np.nan]

        if sexo == "M":
            t_wfa = tablas.get("wfa_boys")
            t_lhfa = tablas.get("lhfa_boys")
            t_wfh = tablas.get("wfh_boys")
        else:
            t_wfa = tablas.get("wfa_girls")
            t_lhfa = tablas.get("lhfa_girls")
            t_wfh = tablas.get("wfh_girls")

        if pd.notna(peso) and pd.notna(edad) and t_wfa is not None:
            L, M, S = _interpolar_lms(edad, t_wfa, "Month")
            res[0] = _calcular_zscore_lms(peso, L, M, S)

        if pd.notna(talla) and pd.notna(edad) and t_lhfa is not None:
            L, M, S = _interpolar_lms(edad, t_lhfa, "Month")
            res[1] = _calcular_zscore_lms(talla, L, M, S)

        if pd.notna(peso) and pd.notna(talla) and t_wfh is not None:
            L, M, S = _interpolar_lms(talla, t_wfh, "Length")
            res[2] = _calcular_zscore_lms(peso, L, M, S)

        return pd.Series(res)

    logger.info("Calculando Z-scores OMS para %d filas...", len(df))
    cols_z = df.apply(_procesar_fila, axis=1)
    cols_z.columns = ["zscore_peso_edad", "zscore_talla_edad", "zscore_peso_talla"]
    df[cols_z.columns] = cols_z

    logger.info("Z-scores OMS calculados")
    return df


def calculate_window_features(
    df: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    """
    Features de ventana: estadisticas, slopes, consejeria en los N controles previos.

    **MODIFICADO para prediccion**: SIEMPRE usa ultimo_control como referencia
    (no primer_alguna). No calcula 'deficit' como output.

    Adaptado de build_features.features_6prev_window().
    """
    vars_cols = [
        "Peso", "Talla", "CabPC", "edad_meses", "control_esperado",
        "_TE_z", "_PE_z", "_PT_z",
        "zscore_peso_edad", "zscore_talla_edad", "zscore_peso_talla",
    ]
    columnas_consejeria = [
        "flg_consj_lact_materna", "flg_consj_higne_corporal",
        "flg_consj_higne_bucal", "flg_consj_supl_hierro",
        "flg_consj_desarrollo", "flg_consj_vacunas",
    ]
    window = window_size
    order_by = "control_esperado"
    atol = 1e-6

    d = df.copy()
    d["control_esperado"] = pd.to_numeric(d["control_esperado"], errors="coerce")
    d["ultimo_control"] = pd.to_numeric(d["ultimo_control"], errors="coerce")

    # Filtrar a columnas existentes
    vars_cols = [c for c in vars_cols if c in d.columns]
    columnas_consejeria = [c for c in columnas_consejeria if c in d.columns]

    for c in vars_cols + columnas_consejeria:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["flg_motora_fina", "flg_motora_gruesa", "flg_cognitivo"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if order_by == "Fecha":
        d["__order__"] = pd.to_datetime(d["Fecha"], dayfirst=True, errors="coerce")
    else:
        d["__order__"] = pd.to_numeric(d[order_by], errors="coerce")

    def _per_hc(g):
        g = g.sort_values("__order__").reset_index(drop=True)

        # PREDICCION: siempre usar ultimo_control como referencia
        if g["ultimo_control"].notna().any():
            ref_val = g.loc[g["ultimo_control"].notna(), "ultimo_control"].iloc[0]
        else:
            out = {
                f"pre{window}_{stat}__{c}": np.nan
                for c in vars_cols
                for stat in ("mean", "min", "max", "std")
            }
            for c in columnas_consejeria:
                out[f"{c}_valor"] = np.nan
                out[f"{c}_sum_prev"] = np.nan
            out.update({
                f"pre{window}_n__rows": 0,
                "ultima ventana": np.nan,
                "slope_peso": np.nan,
                "slope_talla": np.nan,
                "slope_cab_pc": np.nan,
                "Cantidad_acompañantes": 0,
                "flg_desnutricion": 0,
                "porc_desnutricion": 0.0,
                "flg_asiste_control_esperado": 0,
                "flg_alguna_vez_motora": 0,
                "flg_alguna_vez_cognitivo": 0,
                "flg_anemia_window": 0,
                "flg_desnutricion_cronica_window": 0,
                "flg_desnutricion_aguda_window": 0,
                "flg_sobrepeso_window": 0,
                "flg_obesidad_window": 0,
                "flg_prematuro": 0,
                "flg_bajo_peso_nacer": 0,
                "flg_macrosomia": 0,
                "intensidad_consejeria_window_sum": 0,
            })
            return pd.Series(out)

        # Encontrar posicion del evento de referencia
        mask = np.isclose(
            g["control_esperado"].to_numpy(), ref_val, atol=atol, equal_nan=False
        )
        idx = np.flatnonzero(mask)

        if idx.size == 0:
            out = {
                f"pre{window}_{stat}__{c}": np.nan
                for c in vars_cols
                for stat in ("mean", "min", "max", "std")
            }
            for c in columnas_consejeria:
                out[f"{c}_valor"] = np.nan
                out[f"{c}_sum_prev"] = np.nan
            out.update({
                f"pre{window}_n__rows": 0,
                "ultima ventana": ref_val,
                "slope_peso": np.nan,
                "slope_talla": np.nan,
                "slope_cab_pc": np.nan,
                "Cantidad_acompañantes": 0,
                "flg_desnutricion": 0,
                "porc_desnutricion": 0.0,
                "flg_asiste_control_esperado": 0,
                "flg_alguna_vez_motora": 0,
                "flg_alguna_vez_cognitivo": 0,
                "flg_anemia_window": 0,
                "flg_desnutricion_cronica_window": 0,
                "flg_desnutricion_aguda_window": 0,
                "flg_sobrepeso_window": 0,
                "flg_obesidad_window": 0,
                "flg_prematuro": 0,
                "flg_bajo_peso_nacer": 0,
                "flg_macrosomia": 0,
                "intensidad_consejeria_window_sum": 0,
            })
            return pd.Series(out)

        pos = idx[0]
        start = max(0, pos - window)
        prev = g.loc[start : pos - 1]
        current_row = g.loc[pos]

        vals: dict[str, Any] = {}

        # Estadisticas por variable
        for c in vars_cols:
            s = prev[c]
            vals[f"pre{window}_mean__{c}"] = s.mean(skipna=True)
            vals[f"pre{window}_min__{c}"] = s.min(skipna=True)
            vals[f"pre{window}_max__{c}"] = s.max(skipna=True)
            vals[f"pre{window}_std__{c}"] = s.std(skipna=True, ddof=1)

        # Slopes
        vals["slope_peso"] = _calculate_slope(prev["Peso"]) if "Peso" in prev.columns else np.nan
        vals["slope_talla"] = _calculate_slope(prev["Talla"]) if "Talla" in prev.columns else np.nan
        vals["slope_cab_pc"] = _calculate_slope(prev["CabPC"]) if "CabPC" in prev.columns else np.nan

        # Consejeria
        for c in columnas_consejeria:
            vals[f"{c}_valor"] = current_row[c] if pd.notna(current_row[c]) else np.nan
            vals[f"{c}_sum_prev"] = prev[c].sum(skipna=True) if len(prev) > 0 else 0

        # Acompanantes
        if "Acompaña_control" in prev.columns:
            vals["Cantidad_acompañantes"] = prev["Acompaña_control"].nunique()
        else:
            vals["Cantidad_acompañantes"] = 0

        # Desnutricion por diagnostico
        if "Dx_Nutricional" in prev.columns:
            cnt = prev["Dx_Nutricional"].astype(str).str.contains(r"D\.", na=False, regex=True).sum()
            vals["flg_desnutricion"] = 1 if cnt > 0 else 0
            vals["porc_desnutricion"] = cnt / len(prev) if len(prev) > 0 else 0.0
        else:
            vals["flg_desnutricion"] = 0
            vals["porc_desnutricion"] = 0.0

        # Asistencia consecutiva
        if len(prev) > 0 and "control_esperado" in prev.columns:
            controles = prev["control_esperado"].dropna().values
            if len(controles) >= 2:
                diffs = np.diff(controles)
                vals["flg_asiste_control_esperado"] = 1 if np.all(np.abs(diffs - 1) < atol) else 0
            else:
                vals["flg_asiste_control_esperado"] = 0
        else:
            vals["flg_asiste_control_esperado"] = 0

        # Deficit motor alguna vez
        mf = prev["flg_motora_fina"].sum() if "flg_motora_fina" in prev.columns else 0
        mg = prev["flg_motora_gruesa"].sum() if "flg_motora_gruesa" in prev.columns else 0
        vals["flg_alguna_vez_motora"] = 1 if (mf > 0 or mg > 0) else 0

        # Deficit cognitivo alguna vez
        if "flg_cognitivo" in prev.columns:
            vals["flg_alguna_vez_cognitivo"] = 1 if prev["flg_cognitivo"].sum() > 0 else 0
        else:
            vals["flg_alguna_vez_cognitivo"] = 0

        # Anemia en ventana
        if "flg_anemia" in prev.columns:
            vals["flg_anemia_window"] = 1 if prev["flg_anemia"].max() == 1 else 0
        else:
            vals["flg_anemia_window"] = 0

        # Flags nutricionales en ventana
        for f in ["flg_desnutricion_cronica", "flg_desnutricion_aguda", "flg_sobrepeso", "flg_obesidad"]:
            if f in prev.columns:
                vals[f"{f}_window"] = 1 if prev[f].max() == 1 else 0
            else:
                vals[f"{f}_window"] = 0

        # Flags de nacimiento (estaticos por paciente)
        for f in ["flg_prematuro", "flg_bajo_peso_nacer", "flg_macrosomia"]:
            if f in prev.columns:
                vals[f] = prev[f].max()
            else:
                vals[f] = 0

        # Intensidad consejeria en ventana
        if "intensidad_consejeria" in prev.columns:
            vals["intensidad_consejeria_window_sum"] = prev["intensidad_consejeria"].sum()
        else:
            vals["intensidad_consejeria_window_sum"] = 0

        vals[f"pre{window}_n__rows"] = len(prev)
        vals["ultima ventana"] = ref_val

        return pd.Series(vals)

    # Aplicar por N_HC
    resultados = []
    for nhc, g in d.groupby("N_HC"):
        result = _per_hc(g)
        result["N_HC"] = nhc
        resultados.append(result)

    if not resultados:
        return pd.DataFrame()

    feats = pd.DataFrame(resultados)
    cols = feats.columns.tolist()
    if "N_HC" in cols:
        cols.remove("N_HC")
        feats = feats[["N_HC"] + cols]

    logger.info("Window features calculados: %d pacientes x %d features", *feats.shape)
    return feats


def calculate_first_year_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features agregados del primer anio de vida (0-12 meses).
    Adaptado de build_features.calcular_features_primer_anio().
    """
    if not all(c in df.columns for c in ["N_HC", "edad_meses"]):
        return pd.DataFrame(columns=["N_HC", "n_controles_primer_anio",
                                      "flg_desnutricion_primer_anio", "flg_anemia_primer_anio"])

    df_fy = df[df["edad_meses"] <= 12].copy()
    if df_fy.empty:
        return pd.DataFrame(columns=["N_HC", "n_controles_primer_anio",
                                      "flg_desnutricion_primer_anio", "flg_anemia_primer_anio"])

    def _agg(g):
        out = {"n_controles_primer_anio": len(g)}
        nut_cols = [c for c in ["flg_desnutricion_cronica", "flg_desnutricion_aguda", "flg_obesidad"]
                    if c in g.columns]
        if nut_cols:
            out["flg_desnutricion_primer_anio"] = 1 if g[nut_cols].max().max() == 1 else 0
        else:
            out["flg_desnutricion_primer_anio"] = 0
        if "flg_anemia" in g.columns:
            out["flg_anemia_primer_anio"] = 1 if g["flg_anemia"].max() == 1 else 0
        else:
            out["flg_anemia_primer_anio"] = 0
        return pd.Series(out)

    features = df_fy.groupby("N_HC").apply(_agg, include_groups=False).reset_index()
    logger.info("Features primer anio: %d pacientes", len(features))
    return features


def calculate_milestone_features(
    df: pd.DataFrame,
    milestone_months: list[int],
) -> pd.DataFrame:
    """
    Z-scores en hitos de edad (12, 24, 36 meses +/- 1 mes tolerancia).
    Adaptado de build_features.calcular_features_hitos().
    """
    required = ["N_HC", "edad_meses", "_PT_z", "_TE_z", "_PE_z"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(columns=["N_HC"])

    tolerancia = 1
    features_list = []

    for hito in milestone_months:
        mask = (df["edad_meses"] >= hito - tolerancia) & (df["edad_meses"] <= hito + tolerancia)
        df_hito = df[mask].copy()
        if df_hito.empty:
            continue
        df_hito["distancia_hito"] = abs(df_hito["edad_meses"] - hito)
        df_hito = df_hito.sort_values(["N_HC", "distancia_hito"])
        df_hito = df_hito.drop_duplicates(subset=["N_HC"], keep="first")
        cols_z = ["_PT_z", "_TE_z", "_PE_z"]
        df_res = df_hito[["N_HC"] + cols_z].copy()
        rename_map = {c: f"z_{c.replace('_', '').replace('z', '')}_{hito}m" for c in cols_z}
        df_res = df_res.rename(columns=rename_map)
        features_list.append(df_res)

    if not features_list:
        return pd.DataFrame(columns=["N_HC"])

    df_final = features_list[0]
    for df_next in features_list[1:]:
        df_final = df_final.merge(df_next, on="N_HC", how="outer")

    logger.info("Features de hitos calculados: %d pacientes", len(df_final))
    return df_final


def merge_patient_features(
    df_window: pd.DataFrame,
    df_first_year: pd.DataFrame,
    df_milestones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina window features + primer anio + hitos en un dataset por paciente.
    """
    df = df_window.copy()

    if not df_first_year.empty and "N_HC" in df_first_year.columns:
        df = df.merge(df_first_year, on="N_HC", how="left")
        for c in ["n_controles_primer_anio", "flg_desnutricion_primer_anio", "flg_anemia_primer_anio"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)

    if not df_milestones.empty and "N_HC" in df_milestones.columns:
        df = df.merge(df_milestones, on="N_HC", how="left")

    logger.info("Features de paciente combinados: %s", df.shape)
    return df


def clean_for_prediction(
    df: pd.DataFrame,
    selected_features_path: str,
) -> pd.DataFrame:
    """
    Limpia y prepara datos para prediccion: imputacion, seleccion de features.
    Adaptado de clean_patient_features.py.

    Args:
        df: DataFrame con features por paciente.
        selected_features_path: Ruta al JSON con lista de features seleccionadas.
    """
    df = df.copy()

    # Cargar features seleccionadas
    with open(selected_features_path) as f:
        selected_features = json.load(f)

    logger.info("Features seleccionadas: %d", len(selected_features))

    # 1. Eliminar columnas constantes
    const_cols = [c for c in df.columns if df[c].nunique() <= 1 and c != "N_HC"]
    if const_cols:
        logger.info("Eliminando %d columnas constantes", len(const_cols))
        df = df.drop(columns=const_cols)

    # 2. Imputar valores faltantes
    cols = [c for c in df.columns if c not in ("N_HC", "deficit")]
    for col in cols:
        if df[col].isna().sum() == 0:
            continue
        if any(p in col for p in ("flg_", "n_", "Cantidad_", "intensidad_")):
            df[col] = df[col].fillna(0)
        elif "slope_" in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)

    # 3. Seleccionar solo features del modelo
    available = [f for f in selected_features if f in df.columns]
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        logger.warning("Features faltantes (se llenaran con 0): %s", missing)
        for f in missing:
            df[f] = 0

    # Mantener N_HC + features seleccionadas
    df_ready = df[["N_HC"] + selected_features].copy()

    logger.info("Dataset listo para prediccion: %s (nulls: %d)", df_ready.shape, df_ready.isna().sum().sum())
    return df_ready
