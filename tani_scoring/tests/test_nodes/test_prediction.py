"""Tests para nodos de prediction."""

import numpy as np
import pandas as pd
import pytest

from tani_scoring.pipelines.prediction.nodes import (
    classify_risk,
    generate_predictions,
    generate_score_report,
)


class TestGeneratePredictions:
    def test_score_inversion(self):
        """Score 0-100 donde 100=mejor (prob baja) y 0=peor (prob alta)."""
        # Mock model que retorna probabilidades conocidas
        class MockModel:
            def predict_proba(self, X):
                # 3 pacientes con probs: 0.0, 0.5, 1.0
                return np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

        df = pd.DataFrame({
            "N_HC": [1, 2, 3],
            "feat_a": [0.1, 0.2, 0.3],
            "feat_b": [0.4, 0.5, 0.6],
        })
        result = generate_predictions(MockModel(), df, ["feat_a", "feat_b"])
        # prob=0.0 -> score=100, prob=0.5 -> score=50, prob=1.0 -> score=0
        assert result["risk_score"].iloc[0] == 100
        assert result["risk_score"].iloc[1] == 50
        assert result["risk_score"].iloc[2] == 0

    def test_output_columns(self):
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.8, 0.2]])

        df = pd.DataFrame({"N_HC": [1], "f1": [0.5]})
        result = generate_predictions(MockModel(), df, ["f1"])
        assert "N_HC" in result.columns
        assert "risk_score" in result.columns


class TestClassifyRisk:
    def test_categorias_score_100(self):
        """Thresholds empiricos: Alto<64, 64<=Medio<88, Bajo>=88."""
        df = pd.DataFrame({
            "N_HC": [1, 2, 3, 4],
            "risk_score": [30, 63, 64, 88],
        })
        result = classify_risk(df, threshold_high=64, threshold_medium=88)
        assert result["risk_category"].iloc[0] == "Alto"   # 30 < 64
        assert result["risk_category"].iloc[1] == "Alto"   # 63 < 64
        assert result["risk_category"].iloc[2] == "Medio"  # 64 >= 64, < 88
        assert result["risk_category"].iloc[3] == "Bajo"   # 88 >= 88

    def test_scoring_date(self):
        df = pd.DataFrame({"N_HC": [1], "risk_score": [50]})
        result = classify_risk(df, 64, 88)
        assert "scoring_date" in result.columns
        assert result["scoring_date"].iloc[0] is not None

    def test_boundary_values(self):
        """Verificar valores exactos en los limites."""
        df = pd.DataFrame({
            "N_HC": [1, 2, 3, 4, 5],
            "risk_score": [0, 63, 64, 87, 88],
        })
        result = classify_risk(df, 64, 88)
        cats = result["risk_category"].tolist()
        assert cats == ["Alto", "Alto", "Medio", "Medio", "Bajo"]


class TestGenerateScoreReport:
    def test_sort_ascending(self, tmp_path):
        """Pacientes de mayor riesgo (score mas bajo) primero."""
        df_scored = pd.DataFrame({
            "N_HC": [1, 2, 3],
            "risk_score": [80, 40, 60],
            "risk_category": ["Medio", "Alto", "Alto"],
            "scoring_date": ["2026-01-01"] * 3,
        })
        df_features = pd.DataFrame({
            "N_HC": [1, 2, 3],
            "pre6_n__rows": [3.0, 2.0, 4.0],
        })
        output = str(tmp_path / "report.csv")
        result = generate_score_report(df_scored, df_features, output)
        # Primer paciente debe ser el de score mas bajo (mayor riesgo)
        assert result["risk_score"].iloc[0] == 40
        assert result["risk_score"].iloc[-1] == 80

    def test_saves_csv(self, tmp_path):
        df_scored = pd.DataFrame({
            "N_HC": [1],
            "risk_score": [50],
            "risk_category": ["Alto"],
            "scoring_date": ["2026-01-01"],
        })
        df_features = pd.DataFrame({"N_HC": [1]})
        output = str(tmp_path / "sub" / "report.csv")
        generate_score_report(df_scored, df_features, output)
        result = pd.read_csv(output)
        assert len(result) == 1
