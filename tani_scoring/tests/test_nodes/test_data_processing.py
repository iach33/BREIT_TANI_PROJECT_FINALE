"""Tests para nodos de data_processing."""

import numpy as np
import pandas as pd
import pytest

from tani_scoring.pipelines.data_processing.nodes import (
    _edad_a_meses,
    _control_esperado,
    _parse_z,
    _categoria_TE,
    _categoria_PE,
    _categoria_PT,
    _calcular_flg_consejeria,
    clean_patients,
    calculate_birth_features,
    convert_age_to_months,
    calculate_control_tracking,
    filter_scoreable_population,
    process_counseling_columns,
)


class TestEdadAMeses:
    def test_formato_completo(self):
        assert _edad_a_meses("2a 3m 5d") == pytest.approx(27.16, abs=0.1)

    def test_solo_anios(self):
        assert _edad_a_meses("1a") == 12.0

    def test_solo_meses(self):
        assert _edad_a_meses("6m") == 6.0

    def test_solo_dias(self):
        assert _edad_a_meses("15d") == pytest.approx(0.49, abs=0.1)

    def test_nan(self):
        assert np.isnan(_edad_a_meses(np.nan))

    def test_cero(self):
        assert _edad_a_meses("0a0m0d") == 0.0

    def test_formato_sin_espacios(self):
        """Formato compacto como '2a3m' del export real."""
        assert _edad_a_meses("2a3m") == pytest.approx(27.0, abs=0.1)

    def test_solo_anios_sin_meses(self):
        """Formato '4a' del export real."""
        assert _edad_a_meses("4a") == 48.0


class TestControlEsperado:
    def test_recien_nacido(self):
        assert _control_esperado(0.5) == 1

    def test_seis_meses(self):
        assert _control_esperado(6.0) == 7

    def test_doce_meses(self):
        assert _control_esperado(11.5) == 12

    def test_mayor_doce(self):
        assert _control_esperado(18.0) == 18

    def test_nan(self):
        assert np.isnan(_control_esperado(np.nan))


class TestParseZ:
    def test_numerico(self):
        assert _parse_z("1.5") == 1.5

    def test_negativo(self):
        assert _parse_z("-2.3") == -2.3

    def test_coma_decimal(self):
        assert _parse_z("1,5") == 1.5

    def test_codigo_especial(self):
        assert _parse_z("DA") == "DA"

    def test_sin_dato(self):
        assert np.isnan(_parse_z("S/A"))

    def test_nan(self):
        assert np.isnan(_parse_z(np.nan))


class TestCategorias:
    def test_te_da(self):
        assert _categoria_TE(-3.5) == "DA"

    def test_te_normal(self):
        assert _categoria_TE(0.5) == "N"

    def test_pe_riesgo(self):
        assert _categoria_PE(-2.5) == "R"

    def test_pe_normal(self):
        assert _categoria_PE(0.0) == "N"

    def test_pt_string(self):
        assert _categoria_PT("DA") == "DA"

    def test_pt_normal(self):
        assert _categoria_PT(-0.5) == "N"


class TestFlgConsejeria:
    def test_verdadero(self):
        assert _calcular_flg_consejeria("VERDADERO") == 1

    def test_falso(self):
        assert _calcular_flg_consejeria("FALSO") == 0

    def test_true(self):
        assert _calcular_flg_consejeria("True") == 1

    def test_nan(self):
        assert np.isnan(_calcular_flg_consejeria(np.nan))

    def test_boolean_true(self):
        """Formato boolean nativo del export LISTA."""
        assert _calcular_flg_consejeria(True) == 1

    def test_boolean_false(self):
        """Formato boolean nativo del export LISTA."""
        assert _calcular_flg_consejeria(False) == 0

    def test_numpy_bool(self):
        assert _calcular_flg_consejeria(np.bool_(True)) == 1


class TestCleanPatients:
    def test_limpia_edad_gestacional(self):
        df = pd.DataFrame({"Edad_Gestacional": ["38 sem", "36s", None]})
        result = clean_patients(df)
        assert result["Edad_Gestacional"].iloc[0] == 38
        assert result["Edad_Gestacional"].iloc[1] == 36
        assert pd.isna(result["Edad_Gestacional"].iloc[2])

    def test_parto_mama_default(self):
        df = pd.DataFrame({"parto_mama": [None, "", "Cesarea"]})
        result = clean_patients(df)
        assert result["parto_mama"].iloc[0] == "Normal"
        assert result["parto_mama"].iloc[2] == "Cesarea"

    def test_sin_columnas_opcionales(self):
        """Pipeline debe funcionar sin columnas clinicas opcionales."""
        df = pd.DataFrame({"N_HC": [1, 2], "Peso": [5.0, 6.0]})
        result = clean_patients(df)
        assert len(result) == 2


class TestCalculateBirthFeatures:
    def test_flags_nacimiento(self):
        df = pd.DataFrame({
            "Diag_Nacimiento": ["Prematuro", "BPN", "Macrosomico", "Normal"]
        })
        result = calculate_birth_features(df)
        assert result["flg_prematuro"].iloc[0] == 1
        assert result["flg_bajo_peso_nacer"].iloc[1] == 1
        assert result["flg_macrosomia"].iloc[2] == 1
        assert result["flg_prematuro"].iloc[3] == 0

    def test_sin_columna(self):
        df = pd.DataFrame({"otra_col": [1, 2, 3]})
        result = calculate_birth_features(df)
        assert (result["flg_prematuro"] == 0).all()


class TestProcessCounselingColumns:
    def test_formato_lista_boolean(self):
        """Columnas C_* con valores boolean del formato LISTA."""
        df = pd.DataFrame({
            "C_lact_materna": [True, False, True],
            "C_hig_corporal": [True, True, False],
            "C_hig_bucal": [False, True, True],
            "C_supl_hierro": [False, False, True],
            "C_act_dit": [True, True, False],
            "C_cuid_vacuna": [True, False, False],
        })
        result = process_counseling_columns(df)
        assert result["flg_consj_lact_materna"].iloc[0] == 1
        assert result["flg_consj_lact_materna"].iloc[1] == 0
        assert result["flg_consj_higne_corporal"].iloc[0] == 1
        assert result["flg_consj_desarrollo"].iloc[0] == 1  # C_act_dit -> flg_consj_desarrollo
        assert result["flg_consj_vacunas"].iloc[2] == 0
        # Intensidad = sum de 6 flags
        assert result["intensidad_consejeria"].iloc[0] == 4  # True+True+False+False+True+True

    def test_sin_columnas_consejeria(self):
        df = pd.DataFrame({"N_HC": [1, 2]})
        result = process_counseling_columns(df)
        assert "flg_consj_lact_materna" in result.columns
        assert result["intensidad_consejeria"].iloc[0] == 0


class TestFilterScoreablePopulation:
    def test_filtro_basico(self):
        df = pd.DataFrame({
            "N_HC": [1, 1, 2, 2],
            "Fecha": pd.to_datetime(["2024-01-01"] * 4),
            "primer_control_esperado": [1, 1, 4, 4],
            "cant_controles_primer_alguna": [8, 8, 3, 3],
            "ultimo_control": [20, 20, 15, 15],
        })
        result = filter_scoreable_population(df, min_controls=6)
        # N_HC=1 cumple: cant_ctrl >= 6 y ultimo >= 19
        # N_HC=2 no cumple: cant_ctrl < 6 y ultimo < 19
        assert set(result["N_HC"].unique()) == {1}

    def test_filtro_sin_primer_control(self):
        """Filtro ya no requiere primer_control_esperado in [1,2,3]."""
        df = pd.DataFrame({
            "N_HC": [1, 1],
            "Fecha": pd.to_datetime(["2024-01-01"] * 2),
            "primer_control_esperado": [10, 10],  # No empieza en control 1-3
            "cant_controles_primer_alguna": [3, 3],
            "ultimo_control": [24, 24],
        })
        result = filter_scoreable_population(df, min_controls=2)
        # Pasa porque ya no filtramos por primer_control_esperado
        assert len(result) == 2
