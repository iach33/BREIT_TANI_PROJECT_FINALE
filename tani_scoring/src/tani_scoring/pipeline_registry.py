"""Project pipelines."""

from kedro.pipeline import Pipeline

from tani_scoring.pipelines.data_processing.pipeline import create_pipeline as dp_pipeline
from tani_scoring.pipelines.feature_engineering.pipeline import create_pipeline as fe_pipeline
from tani_scoring.pipelines.prediction.pipeline import create_pipeline as pred_pipeline
from tani_scoring.pipelines.model_export.pipeline import create_pipeline as export_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = dp_pipeline()
    feature_engineering = fe_pipeline()
    prediction = pred_pipeline()
    model_export = export_pipeline()

    return {
        # Pipeline por defecto: scoring completo (data -> features -> predict)
        "__default__": data_processing + feature_engineering + prediction,
        # Pipelines individuales
        "data_processing": data_processing,
        "feature_engineering": feature_engineering,
        "prediction": prediction,
        # Pipeline one-time para exportar modelo entrenado
        "model_export": model_export,
    }
