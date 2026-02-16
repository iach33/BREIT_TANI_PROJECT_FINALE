"""Tests para nodos de feature_engineering."""

import numpy as np
import pandas as pd
import pytest

from tani_scoring.pipelines.feature_engineering.nodes import (
    _calcular_flg_desarrollo,
    _calculate_slope,
    _calcular_zscore_lms,
    calculate_anemia_flag,
    calculate_development_flags,
    calculate_nutritional_flags,
    calculate_first_year_features,
    calculate_milestone_features,
    merge_patient_features,
)


class TestFlgDesarrollo:
    def test_defic(self):
        df = pd.DataFrame({"(C) - Cog": ["Defic"]})
        result = _calcular_flg_desarrollo(df, "(C) - Cog")
        assert result.iloc[0] == 1

    def test_normal(self):
        df = pd.DataFrame({"(C) - Cog": ["Normal"]})
        result = _calcular_flg_desarrollo(df, "(C) - Cog")
        assert result.iloc[0] == 0

    def test_nan(self):
        df = pd.DataFrame({"(C) - Cog": [np.nan]})
        result = _calcular_flg_desarrollo(df, "(C) - Cog")
        assert pd.isna(result.iloc[0])

    def test_columna_inexistente(self):
        df = pd.DataFrame({"otra": [1]})
        result = _calcular_flg_desarrollo(df, "(C) - Cog")
        assert pd.isna(result.iloc[0])


class TestCalculateSlope:
    def test_serie_ascendente(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert _calculate_slope(s) == pytest.approx(1.0)

    def test_serie_constante(self):
        s = pd.Series([5.0, 5.0, 5.0])
        assert _calculate_slope(s) == pytest.approx(0.0)

    def test_un_valor(self):
        s = pd.Series([3.0])
        assert np.isnan(_calculate_slope(s))

    def test_con_nans(self):
        s = pd.Series([1.0, np.nan, 3.0])
        slope = _calculate_slope(s)
        assert slope == pytest.approx(2.0)  # solo usa [1.0, 3.0] -> indices [0, 1]


class TestZscoreLMS:
    def test_zscore_normal(self):
        # L=1, M=10, S=0.1 -> Z = ((val/M)^L - 1) / (L*S)
        z = _calcular_zscore_lms(10.0, 1.0, 10.0, 0.1)
        assert z == pytest.approx(0.0)

    def test_zscore_con_l_cero(self):
        # L=0 -> Z = log(val/M) / S
        z = _calcular_zscore_lms(10.0, 0.0, 10.0, 0.1)
        assert z == pytest.approx(0.0)

    def test_nan_input(self):
        assert np.isnan(_calcular_zscore_lms(np.nan, 1, 10, 0.1))


class TestCalculateDevelopmentFlags:
    def test_flags_completos(self):
        df = pd.DataFrame({
            "(C) - Cog": ["Defic", "Normal"],
            "(L) - Len": ["Normal", "Defic"],
            "(M) - FF": ["Normal", "Normal"],
            "(M) - FG": ["Normal", "Normal"],
            "(S) - Soc": ["Normal", "Normal"],
        })
        result = calculate_development_flags(df)
        assert result["flg_cognitivo"].iloc[0] == 1
        assert result["flg_lenguaje"].iloc[1] == 1
        assert result["flg_alguna"].iloc[0] == 1
        assert result["flg_alguna"].iloc[1] == 1
        assert result["flg_total"].iloc[0] == 1
        assert result["flg_total"].iloc[1] == 1


class TestCalculateAnemiaFlag:
    def test_anemia_positiva(self):
        df = pd.DataFrame({"Tam_hb": [9.5], "edad_meses": [12.0]})
        result = calculate_anemia_flag(df)
        assert result["flg_anemia"].iloc[0] == 1

    def test_sin_anemia(self):
        df = pd.DataFrame({"Tam_hb": [12.0], "edad_meses": [12.0]})
        result = calculate_anemia_flag(df)
        assert result["flg_anemia"].iloc[0] == 0

    def test_menor_6_meses(self):
        df = pd.DataFrame({"Tam_hb": [9.5], "edad_meses": [3.0]})
        result = calculate_anemia_flag(df)
        assert pd.isna(result["flg_anemia"].iloc[0])


class TestCalculateNutritionalFlags:
    def test_desnutricion_cronica(self):
        df = pd.DataFrame({"_TE_z": [-2.5], "_PT_z": [0.0]})
        result = calculate_nutritional_flags(df)
        assert result["flg_desnutricion_cronica"].iloc[0] == 1
        assert result["flg_desnutricion_aguda"].iloc[0] == 0

    def test_sobrepeso(self):
        df = pd.DataFrame({"_TE_z": [0.0], "_PT_z": [2.5]})
        result = calculate_nutritional_flags(df)
        assert result["flg_sobrepeso"].iloc[0] == 1
        assert result["flg_obesidad"].iloc[0] == 0


class TestCalculateFirstYearFeatures:
    def test_primer_anio(self):
        df = pd.DataFrame({
            "N_HC": [1, 1, 1, 2, 2],
            "edad_meses": [3.0, 6.0, 9.0, 3.0, 15.0],
            "flg_anemia": [0, 1, 0, 0, 1],
        })
        result = calculate_first_year_features(df)
        r1 = result[result["N_HC"] == 1].iloc[0]
        assert r1["n_controles_primer_anio"] == 3
        assert r1["flg_anemia_primer_anio"] == 1

    def test_sin_datos_primer_anio(self):
        df = pd.DataFrame({
            "N_HC": [1],
            "edad_meses": [15.0],
            "flg_anemia": [0],
        })
        result = calculate_first_year_features(df)
        assert result.empty or len(result[result["N_HC"] == 1]) == 0


class TestMergePatientFeatures:
    def test_merge_completo(self):
        df_window = pd.DataFrame({"N_HC": [1, 2], "slope_peso": [0.1, 0.2]})
        df_fy = pd.DataFrame({"N_HC": [1], "n_controles_primer_anio": [5]})
        df_hitos = pd.DataFrame({"N_HC": [1, 2], "z_PT_12m": [-0.5, 0.3]})

        result = merge_patient_features(df_window, df_fy, df_hitos)
        assert len(result) == 2
        assert "n_controles_primer_anio" in result.columns
        assert "z_PT_12m" in result.columns
        # N_HC=2 no tiene primer anio -> debe ser 0
        assert result.loc[result["N_HC"] == 2, "n_controles_primer_anio"].iloc[0] == 0
