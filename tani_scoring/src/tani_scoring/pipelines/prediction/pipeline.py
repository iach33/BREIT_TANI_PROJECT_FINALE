"""Pipeline de prediccion (scoring)."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    classify_risk,
    generate_predictions,
    generate_score_report,
    load_model_artifacts,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_model_artifacts,
                inputs=[
                    "params:model_path",
                    "params:selected_features_path",
                    "params:model_metadata_path",
                ],
                outputs=["trained_model", "feature_list", "model_metadata"],
                name="load_model_artifacts",
            ),
            node(
                func=generate_predictions,
                inputs=["trained_model", "model_ready", "feature_list"],
                outputs="predictions",
                name="generate_predictions",
            ),
            node(
                func=classify_risk,
                inputs=[
                    "predictions",
                    "params:risk_threshold_high",
                    "params:risk_threshold_medium",
                ],
                outputs="scored",
                name="classify_risk",
            ),
            node(
                func=generate_score_report,
                inputs=["scored", "patient_features", "params:score_output_path"],
                outputs="score_report",
                name="generate_score_report",
            ),
        ]
    )
