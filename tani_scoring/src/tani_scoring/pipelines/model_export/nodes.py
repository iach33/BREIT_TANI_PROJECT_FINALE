"""
Nodos del pipeline de exportacion de modelo (one-time).

Entrena el mejor modelo usando los datos de training existentes
y exporta artefactos (modelo, features, metadata) para uso en prediccion.
"""

import json
import logging
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def run_feature_selection(
    model_ready_path: str,
    iv_threshold: float = 0.02,
    corr_threshold: float = 0.9,
) -> list[str]:
    """
    Ejecuta seleccion de features: IV filtering + correlation filtering.
    Adaptado de selection.select_features().

    Args:
        model_ready_path: Ruta al CSV model-ready con columna 'deficit'.
        iv_threshold: Umbral minimo de Information Value.
        corr_threshold: Umbral maximo de correlacion.

    Returns:
        Lista de features seleccionadas.
    """
    logger.info("Cargando datos para seleccion de features: %s", model_ready_path)
    df = pd.read_csv(model_ready_path)

    target_col = "deficit"
    X = df.drop(columns=["N_HC", "ultima ventana", target_col], errors="ignore")
    y = df[target_col]

    # 1. Information Value
    logger.info("Calculando IV para %d features...", len(X.columns))
    iv_values = {}
    for col in X.columns:
        iv_values[col] = _calculate_iv(df, col, target_col)

    iv_series = pd.Series(iv_values).sort_values(ascending=False)
    selected_iv = iv_series[iv_series >= iv_threshold].index.tolist()
    logger.info("Features con IV >= %.3f: %d / %d", iv_threshold, len(selected_iv), len(X.columns))

    # 2. Correlation filtering
    X_iv = X[selected_iv]
    corr_matrix = X_iv.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    logger.info("Features eliminadas por correlacion (>%.2f): %d", corr_threshold, len(drop_corr))

    final_features = [f for f in selected_iv if f not in drop_corr]
    logger.info("Features finales seleccionadas: %d", len(final_features))
    return final_features


def _calculate_iv(df, feature, target):
    """Calcula Information Value para un feature."""
    df_temp = df[[feature, target]].copy()
    if np.issubdtype(df_temp[feature].dtype, np.number) and df_temp[feature].nunique() > 10:
        try:
            df_temp["bin"] = pd.qcut(df_temp[feature], q=10, duplicates="drop")
        except Exception:
            df_temp["bin"] = pd.cut(df_temp[feature], bins=10)
    else:
        df_temp["bin"] = df_temp[feature]

    grouped = df_temp.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["Total", "Bad"]
    grouped["Good"] = grouped["Total"] - grouped["Bad"]

    total_bad = grouped["Bad"].sum()
    total_good = grouped["Good"].sum()
    if total_bad == 0 or total_good == 0:
        return 0.0

    grouped["Dist_Bad"] = grouped["Bad"] / total_bad
    grouped["Dist_Good"] = grouped["Good"] / total_good
    grouped["WoE"] = np.log((grouped["Dist_Good"] + 1e-5) / (grouped["Dist_Bad"] + 1e-5))
    grouped["IV"] = (grouped["Dist_Good"] - grouped["Dist_Bad"]) * grouped["WoE"]
    return grouped["IV"].sum()


def train_best_model(
    model_ready_path: str,
    selected_features: list[str],
) -> tuple[object, dict]:
    """
    Entrena todos los modelos y selecciona el mejor por AUC.
    Adaptado de train_model.train_models().

    NOTA CRITICA sobre SMOTE: El modelo para prediccion NO debe incluir SMOTE.
    Se entrena con SMOTE pero se exporta un pipeline sin SMOTE
    (imputer + scaler + classifier con parametros ya entrenados).

    Returns:
        (prediction_pipeline, metrics_dict)
    """
    logger.info("Cargando datos para entrenamiento: %s", model_ready_path)
    df = pd.read_csv(model_ready_path)

    target_col = "deficit"
    drop_cols = ["N_HC", "ultima ventana", target_col]
    X = df[[c for c in selected_features if c in df.columns]].copy()
    y = df[target_col].astype(int)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    # Modelos base
    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1),
    }

    param_grids = {
        "LogisticRegression": {
            "classifier__C": loguniform(1e-4, 100),
            "classifier__penalty": ["l2"],
        },
        "RandomForest": {
            "classifier__n_estimators": randint(50, 300),
            "classifier__max_depth": randint(3, 20),
            "classifier__min_samples_split": randint(2, 20),
            "classifier__min_samples_leaf": randint(1, 10),
        },
        "XGBoost": {
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__n_estimators": randint(50, 300),
            "classifier__max_depth": randint(3, 10),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
        "LightGBM": {
            "classifier__learning_rate": loguniform(0.01, 0.3),
            "classifier__n_estimators": randint(50, 300),
            "classifier__max_depth": randint(3, 10),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
            "classifier__num_leaves": randint(20, 100),
        },
    }

    best_model = None
    best_auc = 0.0
    best_name = ""

    for name, model in base_models.items():
        logger.info("Entrenando %s...", name)

        # Pipeline con SMOTE (para entrenamiento)
        train_pipeline = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ])

        # Baseline
        train_pipeline.fit(X_train, y_train)
        auc_base = roc_auc_score(y_test, train_pipeline.predict_proba(X_test)[:, 1])
        logger.info("%s Baseline AUC: %.4f", name, auc_base)

        if auc_base > best_auc:
            best_auc = auc_base
            best_name = f"{name}_Baseline"
            best_model = train_pipeline

        # Optimizado
        if name in param_grids:
            search = RandomizedSearchCV(
                train_pipeline,
                param_distributions=param_grids[name],
                n_iter=20,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
            search.fit(X_train, y_train)
            auc_opt = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])
            logger.info("%s Optimized AUC: %.4f", name, auc_opt)

            if auc_opt > best_auc:
                best_auc = auc_opt
                best_name = f"{name}_Optimized"
                best_model = search.best_estimator_

    logger.info("Mejor modelo: %s (AUC=%.4f)", best_name, best_auc)

    # EXPORTAR: Pipeline SIN SMOTE para prediccion
    # Extraer componentes entrenados del mejor pipeline
    imputer = best_model.named_steps["imputer"]
    scaler = best_model.named_steps["scaler"]
    classifier = best_model.named_steps["classifier"]

    # Crear pipeline de prediccion (sin SMOTE)
    prediction_pipeline = SkPipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("classifier", classifier),
    ])

    metrics = {
        "model_name": best_name,
        "auc": round(best_auc, 4),
        "n_features": len(selected_features),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return prediction_pipeline, metrics


def export_artifacts(
    trained_pipeline: object,
    selected_features: list[str],
    metrics: dict,
    output_dir: str,
) -> None:
    """
    Serializa el modelo, features y metadata a disco.

    Genera:
        - model.joblib: Pipeline sklearn (imputer + scaler + classifier)
        - selected_features.json: Lista de features
        - model_metadata.json: AUC, fecha, etc.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Modelo
    model_path = out / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    logger.info("Modelo guardado: %s", model_path)

    # Features
    features_path = out / "selected_features.json"
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)
    logger.info("Features guardadas: %s (%d features)", features_path, len(selected_features))

    # Metadata
    metadata = {
        **metrics,
        "training_date": date.today().isoformat(),
        "model_file": "model.joblib",
        "features_file": "selected_features.json",
    }
    metadata_path = out / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata guardada: %s", metadata_path)
