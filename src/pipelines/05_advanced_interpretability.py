"""
Pipeline 05: Advanced Interpretability Analysis

Generates comprehensive SHAP visualizations and analysis for the best model.
"""

import sys
from pathlib import Path
import pandas as pd
import pickle

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import settings
from models.train_model import load_data, get_feature_target_split, train_models
from models.evaluate_model import evaluate_models
from models.interpretability import select_best_model
from models.interpretability_advanced import generate_comprehensive_shap_analysis


def main():
    """Execute advanced interpretability pipeline."""

    print("=" * 80)
    print("PIPELINE 05: ADVANCED INTERPRETABILITY ANALYSIS")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading model-ready dataset...")
    df = load_data(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")
    X, y = get_feature_target_split(df)

    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {len(X.columns)}")

    # 2. Train models (to get best model)
    print("\n2. Training models...")
    trained_models, X_test, y_test = train_models(X, y)

    # 3. Evaluate models
    print("\n3. Evaluating models...")
    results_df = evaluate_models(trained_models, X_test, y_test)

    # 4. Select best model
    print("\n4. Selecting best model for interpretation...")
    best_model, best_model_name = select_best_model(results_df, trained_models)

    # 5. Run comprehensive SHAP analysis
    print(f"\n5. Running comprehensive SHAP analysis on {best_model_name}...")

    shap_results = generate_comprehensive_shap_analysis(
        model_pipeline=best_model,
        X_test=X_test,
        y_test=y_test,
        model_name=best_model_name,
        feature_names=X.columns.tolist(),
        top_n_features=10
    )

    # 6. Display top insights
    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 80)
    print(shap_results['shap_stats'].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("FEATURE DIRECTION ANALYSIS (Top 10)")
    print("=" * 80)
    print(shap_results['feature_directions'].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("✓ ADVANCED INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated visualizations:")
    print(f"  - Summary plots (beeswarm, bar)")
    print(f"  - Waterfall plots (high/low risk cases)")
    print(f"  - Dependence plots (top 6 features)")
    print(f"  - Interaction plot (top 2 features)")
    print(f"  - SHAP heatmap (30 cases × 15 features)")
    print(f"  - Force plot (high risk case)")
    print(f"\nStatistics saved:")
    print(f"  - shap_statistics.csv")
    print(f"  - shap_feature_directions.csv")


if __name__ == "__main__":
    main()
