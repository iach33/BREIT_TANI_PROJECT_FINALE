"""Pipeline de procesamiento de datos."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_birth_features,
    calculate_control_tracking,
    clean_patients,
    convert_age_to_months,
    filter_scoreable_population,
    load_and_merge_counseling,
    load_development_data,
    load_nutrition_data,
    merge_nutrition_development,
    parse_zscores,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_nutrition_data,
                inputs="params:raw_data_path",
                outputs="raw_nutrition",
                name="load_nutrition_data",
            ),
            node(
                func=load_development_data,
                inputs="params:raw_data_path",
                outputs="raw_development",
                name="load_development_data",
            ),
            node(
                func=merge_nutrition_development,
                inputs=["raw_nutrition", "raw_development"],
                outputs="consolidated",
                name="merge_nutrition_development",
            ),
            node(
                func=clean_patients,
                inputs="consolidated",
                outputs="cleaned",
                name="clean_patients",
            ),
            node(
                func=calculate_birth_features,
                inputs="cleaned",
                outputs="with_birth",
                name="calculate_birth_features",
            ),
            node(
                func=convert_age_to_months,
                inputs="with_birth",
                outputs="with_age",
                name="convert_age_to_months",
            ),
            node(
                func=parse_zscores,
                inputs="with_age",
                outputs="with_zscores",
                name="parse_zscores",
            ),
            node(
                func=calculate_control_tracking,
                inputs="with_zscores",
                outputs="tracked",
                name="calculate_control_tracking",
            ),
            node(
                func=filter_scoreable_population,
                inputs=["tracked", "params:min_controls_for_scoring"],
                outputs="filtered",
                name="filter_scoreable_population",
            ),
            node(
                func=load_and_merge_counseling,
                inputs=["filtered", "params:counseling_data_path"],
                outputs="with_counseling",
                name="load_and_merge_counseling",
            ),
        ]
    )
