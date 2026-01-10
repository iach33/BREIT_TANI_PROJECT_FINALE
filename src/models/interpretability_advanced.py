"""
Advanced SHAP interpretability analysis for TANI model.

Generates comprehensive SHAP visualizations to understand:
- Global feature importance
- Individual predictions
- Feature interactions
- Non-linear relationships
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import settings


def generate_comprehensive_shap_analysis(model_pipeline, X_test, y_test, model_name,
                                         feature_names=None, top_n_features=10):
    """
    Generates comprehensive SHAP analysis with multiple visualizations.

    Parameters:
    -----------
    model_pipeline : sklearn Pipeline
        Trained model pipeline with preprocessing steps
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test targets
    model_name : str
        Name of the model for labeling
    feature_names : list, optional
        Feature names (uses X_test.columns if None)
    top_n_features : int
        Number of top features to analyze in detail

    Returns:
    --------
    dict with SHAP values and analysis results
    """

    save_dir = settings.FIGURES_DIR / "interpretability"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE SHAP ANALYSIS")
    print("=" * 80)

    # 1. Transform test data through preprocessing steps
    print("\n1. Preprocessing test data...")
    X_test_transformed = X_test.copy()

    if feature_names is None:
        feature_names = X_test.columns.tolist()

    if 'imputer' in model_pipeline.named_steps:
        imputer = model_pipeline.named_steps['imputer']
        X_test_transformed = pd.DataFrame(
            imputer.transform(X_test_transformed),
            columns=feature_names
        )

    if 'scaler' in model_pipeline.named_steps:
        scaler = model_pipeline.named_steps['scaler']
        X_test_transformed = pd.DataFrame(
            scaler.transform(X_test_transformed),
            columns=feature_names
        )

    classifier = model_pipeline.named_steps['classifier']

    # 2. Calculate SHAP values
    print("2. Calculating SHAP values...")
    is_tree_model = any(name in model_name for name in ['XGBoost', 'LightGBM', 'RandomForest'])

    if is_tree_model:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_transformed)
    else:
        explainer = shap.LinearExplainer(classifier, X_test_transformed)
        shap_values = explainer.shap_values(X_test_transformed)

    # Handle SHAP format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # 3. Global Feature Importance
    print("3. Generating global importance plots...")

    # 3a. Summary Plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, show=False, max_display=15)
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (impact on prediction)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3b. Bar Plot (mean absolute SHAP)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, plot_type="bar",
                     show=False, max_display=15)
    plt.title(f'SHAP Global Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Get top features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n_features]
    top_features = [feature_names[i] for i in top_indices]

    print(f"\nTop {top_n_features} most important features:")
    for i, (idx, feat) in enumerate(zip(top_indices, top_features), 1):
        print(f"  {i}. {feat}: {mean_abs_shap[idx]:.4f}")

    # 5. Individual Predictions (Waterfall Plots)
    print("\n4. Generating individual prediction examples...")

    # Find high-risk case (predicted positive)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_pred_proba)

    # Find low-risk case (predicted negative)
    low_risk_idx = np.argmin(y_pred_proba)

    # Get base value (handle array or scalar)
    if hasattr(explainer, 'expected_value'):
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] if len(base_value) == 1 else base_value[1]  # Get positive class
        base_value = float(base_value)
    else:
        base_value = 0.0

    # Waterfall plot - High Risk Case
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=base_value,
            data=X_test_transformed.iloc[high_risk_idx],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.title(f'High-Risk Case Explanation\n(Predicted Prob: {y_pred_proba[high_risk_idx]:.3f}, True: {y_test.iloc[high_risk_idx]})',
             fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_waterfall_high_risk.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Waterfall plot - Low Risk Case
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[low_risk_idx],
            base_values=base_value,
            data=X_test_transformed.iloc[low_risk_idx],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.title(f'Low-Risk Case Explanation\n(Predicted Prob: {y_pred_proba[low_risk_idx]:.3f}, True: {y_test.iloc[low_risk_idx]})',
             fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_waterfall_low_risk.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Dependence Plots for Top Features
    print(f"\n5. Generating dependence plots for top {min(6, len(top_features))} features...")

    for i, feat_idx in enumerate(top_indices[:6]):
        feat_name = feature_names[feat_idx]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_test_transformed,
            feature_names=feature_names,
            show=False,
            alpha=0.5
        )
        plt.title(f'SHAP Dependence Plot: {feat_name}', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel(f'{feat_name} (feature value)', fontsize=11)
        plt.ylabel('SHAP value (impact on prediction)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Sanitize filename
        safe_name = feat_name.replace('/', '_').replace(' ', '_')
        plt.savefig(save_dir / f"shap_dependence_{i+1}_{safe_name}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Feature Interaction Analysis
    print("\n6. Analyzing feature interactions...")

    # Get top 2 features for interaction plot
    if len(top_features) >= 2:
        feat1_idx = top_indices[0]
        feat2_idx = top_indices[1]

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat1_idx,
            shap_values,
            X_test_transformed,
            feature_names=feature_names,
            interaction_index=feat2_idx,
            show=False,
            alpha=0.5
        )
        plt.title(f'Feature Interaction: {top_features[0]} vs {top_features[1]}',
                 fontsize=12, fontweight='bold', pad=15)
        plt.xlabel(f'{top_features[0]} (feature value)', fontsize=11)
        plt.ylabel('SHAP value (impact on prediction)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_interaction_top2.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 8. SHAP Heatmap for Sample Cases
    print("\n7. Generating SHAP heatmap for sample predictions...")

    # Select sample of diverse cases
    n_samples = min(30, len(X_test))
    sample_indices = np.linspace(0, len(X_test)-1, n_samples, dtype=int)

    plt.figure(figsize=(14, 10))

    # Create heatmap data
    shap_sample = shap_values[sample_indices][:, top_indices[:15]]
    feature_labels = [feature_names[i] for i in top_indices[:15]]

    sns.heatmap(
        shap_sample.T,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'SHAP value'},
        yticklabels=feature_labels,
        xticklabels=[f"Case {i}" for i in range(n_samples)],
        vmin=-np.abs(shap_sample).max(),
        vmax=np.abs(shap_sample).max()
    )
    plt.title(f'SHAP Values Heatmap - Top 15 Features Across {n_samples} Cases',
             fontsize=13, fontweight='bold', pad=20)
    plt.xlabel('Test Cases', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Force Plot (HTML - for individual cases)
    print("\n8. Generating force plot visualizations...")

    try:
        # Force plot for high-risk case
        force_plot = shap.force_plot(
            base_value,
            shap_values[high_risk_idx],
            X_test_transformed.iloc[high_risk_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'Force Plot - High Risk Case (Prob: {y_pred_proba[high_risk_idx]:.3f})',
                 fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / "shap_force_high_risk.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   Could not generate matplotlib force plot: {e}")

    # 10. Summary Statistics
    print("\n9. Computing SHAP summary statistics...")

    shap_stats = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
        'Mean_SHAP': shap_values.mean(axis=0),
        'Std_SHAP': shap_values.std(axis=0),
        'Max_SHAP': shap_values.max(axis=0),
        'Min_SHAP': shap_values.min(axis=0)
    })

    shap_stats = shap_stats.sort_values('Mean_Abs_SHAP', ascending=False)
    shap_stats.to_csv(save_dir / 'shap_statistics.csv', index=False)

    print(f"\n✓ Saved SHAP statistics to: {save_dir / 'shap_statistics.csv'}")

    # 11. Feature Directions
    print("\n10. Analyzing feature direction effects...")

    feature_directions = []
    for i, feat in enumerate(feature_names):
        positive_impact = (shap_values[:, i] > 0).sum()
        negative_impact = (shap_values[:, i] < 0).sum()
        mean_pos = shap_values[shap_values[:, i] > 0, i].mean() if positive_impact > 0 else 0
        mean_neg = shap_values[shap_values[:, i] < 0, i].mean() if negative_impact > 0 else 0

        feature_directions.append({
            'Feature': feat,
            'Positive_Impact_Count': positive_impact,
            'Negative_Impact_Count': negative_impact,
            'Mean_Positive_SHAP': mean_pos,
            'Mean_Negative_SHAP': mean_neg,
            'Net_Direction': 'Increases Risk' if shap_values[:, i].mean() > 0 else 'Decreases Risk'
        })

    directions_df = pd.DataFrame(feature_directions)
    directions_df = directions_df.sort_values('Positive_Impact_Count', ascending=False)
    directions_df.to_csv(save_dir / 'shap_feature_directions.csv', index=False)

    print(f"✓ Saved feature directions to: {save_dir / 'shap_feature_directions.csv'}")

    print("\n" + "=" * 80)
    print("✓ COMPREHENSIVE SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All visualizations saved to: {save_dir}")

    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'X_test_transformed': X_test_transformed,
        'top_features': top_features,
        'shap_stats': shap_stats,
        'feature_directions': directions_df
    }
