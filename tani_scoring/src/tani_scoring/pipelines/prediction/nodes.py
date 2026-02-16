"""
Nodos del pipeline de prediccion (scoring).

Carga el modelo entrenado y genera scores de riesgo por paciente.
"""

import json
import logging
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_model_artifacts(
    model_path: str,
    selected_features_path: str,
    metadata_path: str,
) -> tuple[object, list[str], dict]:
    """
    Carga artefactos del modelo exportado.

    Returns:
        (model, feature_list, metadata)
    """
    logger.info("Cargando modelo desde %s", model_path)
    model = joblib.load(model_path)

    with open(selected_features_path) as f:
        feature_list = json.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(
        "Modelo cargado: %s (AUC entrenamiento: %.4f, features: %d)",
        metadata.get("model_name", "unknown"),
        metadata.get("auc", 0.0),
        len(feature_list),
    )
    return model, feature_list, metadata


def generate_predictions(
    model: object,
    df_model_ready: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Genera probabilidades de deficit usando el modelo cargado.

    El imblearn Pipeline ya incluye imputer + scaler (sin SMOTE),
    asi que solo necesita X con las features correctas.
    """
    # Separar N_HC
    nhc = df_model_ready["N_HC"].copy()
    X = df_model_ready[feature_list].copy()

    logger.info("Generando predicciones para %d pacientes...", len(X))

    # predict_proba -> probabilidad de clase positiva (deficit)
    probas = model.predict_proba(X)[:, 1]

    df_pred = pd.DataFrame({
        "N_HC": nhc,
        "risk_score": probas,
    })

    logger.info(
        "Predicciones generadas. Score medio: %.4f, max: %.4f",
        probas.mean(),
        probas.max(),
    )
    return df_pred


def classify_risk(
    df_predictions: pd.DataFrame,
    threshold_high: float,
    threshold_medium: float,
) -> pd.DataFrame:
    """
    Asigna categorias de riesgo basadas en thresholds.
    Alto >= threshold_high, Medio >= threshold_medium, Bajo < threshold_medium.
    """
    df = df_predictions.copy()

    def _classify(score):
        if score >= threshold_high:
            return "Alto"
        if score >= threshold_medium:
            return "Medio"
        return "Bajo"

    df["risk_category"] = df["risk_score"].apply(_classify)
    df["scoring_date"] = date.today().isoformat()

    counts = df["risk_category"].value_counts()
    logger.info("Clasificacion de riesgo: %s", counts.to_dict())
    return df


def generate_score_report(
    df_scored: pd.DataFrame,
    df_patient_features: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Genera reporte final de scores con metadata del paciente.

    Output: N_HC, risk_score, risk_category, scoring_date, n_controls, last_control_date
    """
    df = df_scored.copy()

    # Agregar metadata del paciente
    if "N_HC" in df_patient_features.columns:
        # Numero de controles en la ventana
        if "pre6_n__rows" in df_patient_features.columns:
            meta = df_patient_features[["N_HC", "pre6_n__rows"]].rename(
                columns={"pre6_n__rows": "n_controls"}
            )
            df = df.merge(meta, on="N_HC", how="left")
        else:
            df["n_controls"] = np.nan

        # Ultimo control
        if "ultima ventana" in df_patient_features.columns:
            meta_uc = df_patient_features[["N_HC", "ultima ventana"]].rename(
                columns={"ultima ventana": "last_control_number"}
            )
            df = df.merge(meta_uc, on="N_HC", how="left")

    # Ordenar por riesgo descendente
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Guardar
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    logger.info("Reporte de scores guardado en %s (%d pacientes)", output, len(df))
    return df
