"""Pipeline de exportacion de modelo (one-time)."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import export_artifacts, run_feature_selection, train_best_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_feature_selection,
                inputs=[
                    "params:model_ready_training_path",
                    "params:iv_threshold",
                    "params:corr_threshold",
                ],
                outputs="export_selected_features",
                name="run_feature_selection",
            ),
            node(
                func=train_best_model,
                inputs=[
                    "params:model_ready_training_path",
                    "export_selected_features",
                ],
                outputs=["export_trained_pipeline", "export_metrics"],
                name="train_best_model",
            ),
            node(
                func=export_artifacts,
                inputs=[
                    "export_trained_pipeline",
                    "export_selected_features",
                    "export_metrics",
                    "params:model_artifacts_dir",
                ],
                outputs=None,
                name="export_artifacts",
            ),
        ]
    )
