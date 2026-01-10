"""
Pipeline 04: Temporal Validation

This pipeline evaluates models on a temporal holdout set to assess
performance on future, unseen time periods.

Critical for longitudinal data where we need to validate predictive
ability on data collected AFTER the training period.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import settings
from models.temporal_validation import (
    create_temporal_split,
    evaluate_temporal_performance,
    compare_random_vs_temporal
)
from models.train_model import load_data, get_feature_target_split, train_models


def main():
    """Execute temporal validation pipeline."""

    print("=" * 80)
    print("PIPELINE 04: TEMPORAL VALIDATION")
    print("=" * 80)

    # 1. Load datasets
    print("\n1. Loading datasets...")
    df_model = load_data(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")
    df_analytical = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_analytical_dataset.csv")

    print(f"   Model-ready dataset: {df_model.shape}")
    print(f"   Analytical dataset: {df_analytical.shape}")

    # 2. Create temporal split
    print("\n2. Creating temporal split (80th percentile cutoff)...")
    temporal_split = create_temporal_split(
        df_model=df_model,
        df_analytical=df_analytical,
        temporal_percentile=0.8
    )

    # 3. Train models on temporal training set
    print("\n3. Training models on TEMPORAL training set...")
    print(f"   Training on data up to: {temporal_split['cutoff_date'].date()}")

    # Reconstruct temporal training dataset with all columns for train_models()
    df_train_temporal_full = df_model[df_model['N_HC'].isin(temporal_split['train_patients'])].copy()
    X_train_temp, y_train_temp = get_feature_target_split(df_train_temporal_full)

    # Train models (this will do internal random split for hyperparameter tuning)
    trained_models, X_test_random, y_test_random = train_models(X_train_temp, y_train_temp)

    print(f"\n   ✓ Trained {len(trained_models)} models")

    # 4. Evaluate on temporal test set
    print("\n4. Evaluating models on TEMPORAL test set...")
    print(f"   Testing on data after: {temporal_split['cutoff_date'].date()}")

    temporal_results = evaluate_temporal_performance(trained_models, temporal_split)

    print("\n" + "=" * 80)
    print("TEMPORAL TEST SET PERFORMANCE")
    print("=" * 80)
    print(temporal_results.to_string(index=False))

    # Save temporal results
    temporal_results.to_csv(settings.REPORTS_DIR / 'model_comparison_temporal.csv', index=False)
    print(f"\n   ✓ Saved temporal results to: model_comparison_temporal.csv")

    # 5. Load random split results for comparison
    print("\n5. Comparing Random vs Temporal performance...")
    try:
        random_results = pd.read_csv(settings.REPORTS_DIR / 'model_comparison.csv')

        comparison = compare_random_vs_temporal(random_results, temporal_results)

        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON: Random Test vs Temporal Test")
        print("=" * 80)
        comparison_display = comparison[[
            'Model', 'AUC', 'AUC_Temporal', 'AUC_Degradation', 'AUC_Degradation_Pct',
            'Precision', 'Precision_Temporal', 'Recall', 'Recall_Temporal'
        ]]
        print(comparison_display.to_string(index=False))

        # Save comparison
        comparison.to_csv(settings.REPORTS_DIR / 'model_comparison_random_vs_temporal.csv', index=False)
        print(f"\n   ✓ Saved comparison to: model_comparison_random_vs_temporal.csv")

        # 6. Summary insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)

        best_temporal = temporal_results.iloc[0]
        print(f"Best model (Temporal Test): {best_temporal['Model']} (AUC={best_temporal['AUC_Temporal']:.4f})")

        avg_degradation = comparison['AUC_Degradation'].mean()
        max_degradation = comparison['AUC_Degradation'].max()
        max_degradation_model = comparison.loc[comparison['AUC_Degradation'].idxmax(), 'Model']

        print(f"\nAverage AUC degradation (Random → Temporal): {avg_degradation:.4f} ({avg_degradation/comparison['AUC'].mean()*100:.1f}%)")
        print(f"Max degradation: {max_degradation:.4f} in {max_degradation_model}")

        # Check if degradation is acceptable (<0.05 typically considered stable)
        if avg_degradation < 0.05:
            print("\n✓ GOOD: Models show minimal temporal degradation (generalize well to future data)")
        elif avg_degradation < 0.10:
            print("\n⚠ MODERATE: Models show some temporal degradation (monitor performance)")
        else:
            print("\n❌ HIGH: Models show significant temporal degradation (may need recalibration)")

    except FileNotFoundError:
        print("   ⚠ model_comparison.csv not found. Run pipeline 03_modeling.py first.")

    print("\n" + "=" * 80)
    print("✓ TEMPORAL VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
