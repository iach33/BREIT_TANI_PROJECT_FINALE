"""Pipeline de ingenieria de features."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_anemia_flag,
    calculate_development_flags,
    calculate_first_year_features,
    calculate_milestone_features,
    calculate_nutritional_flags,
    calculate_oms_zscores,
    calculate_window_features,
    clean_for_prediction,
    merge_patient_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_development_flags,
                inputs="with_counseling",
                outputs="with_dev_flags",
                name="calculate_development_flags",
            ),
            node(
                func=calculate_anemia_flag,
                inputs="with_dev_flags",
                outputs="with_anemia",
                name="calculate_anemia_flag",
            ),
            node(
                func=calculate_nutritional_flags,
                inputs="with_anemia",
                outputs="with_nutrition_flags",
                name="calculate_nutritional_flags",
            ),
            node(
                func=calculate_oms_zscores,
                inputs=["with_nutrition_flags", "params:oms_tables_path"],
                outputs="with_oms",
                name="calculate_oms_zscores",
            ),
            node(
                func=calculate_window_features,
                inputs=["with_oms", "params:window_size"],
                outputs="df_window",
                name="calculate_window_features",
            ),
            node(
                func=calculate_first_year_features,
                inputs="with_oms",
                outputs="df_first_year",
                name="calculate_first_year_features",
            ),
            node(
                func=calculate_milestone_features,
                inputs=["with_oms", "params:milestone_months"],
                outputs="df_milestones",
                name="calculate_milestone_features",
            ),
            node(
                func=merge_patient_features,
                inputs=["df_window", "df_first_year", "df_milestones"],
                outputs="patient_features",
                name="merge_patient_features",
            ),
            node(
                func=clean_for_prediction,
                inputs=["patient_features", "params:selected_features_path"],
                outputs="model_ready",
                name="clean_for_prediction",
            ),
        ]
    )
