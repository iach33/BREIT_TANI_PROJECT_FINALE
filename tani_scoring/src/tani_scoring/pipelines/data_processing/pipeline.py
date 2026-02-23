"""Pipeline de procesamiento de datos."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calculate_birth_features,
    calculate_control_tracking,
    clean_patients,
    convert_age_to_months,
    filter_scoreable_population,
    load_raw_data,
    parse_zscores,
    process_counseling_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_raw_data,
                inputs="params:raw_data_path",
                outputs="consolidated",
                name="load_raw_data",
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
                func=process_counseling_columns,
                inputs="with_zscores",
                outputs="with_counseling",
                name="process_counseling_columns",
            ),
            node(
                func=calculate_control_tracking,
                inputs="with_counseling",
                outputs="tracked",
                name="calculate_control_tracking",
            ),
            node(
                func=filter_scoreable_population,
                inputs=["tracked", "params:min_controls_for_scoring"],
                outputs="filtered",
                name="filter_scoreable_population",
            ),
        ]
    )
