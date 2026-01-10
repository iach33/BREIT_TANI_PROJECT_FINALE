"""
Generate high-quality EDA visualizations for final report.

Creates polished, publication-ready plots focusing on key findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import settings


def create_age_deficit_distribution():
    """Create age group vs deficit rate visualization."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Load analytical for age
    df_analytical = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_analytical_dataset.csv")
    patient_age = df_analytical.groupby('N_HC')['edad_meses'].mean().reset_index()
    patient_age.columns = ['N_HC', 'edad_meses']

    df_with_age = df.merge(patient_age, on='N_HC', how='left')

    # Create age bins
    df_with_age['age_group'] = pd.cut(
        df_with_age['edad_meses'],
        bins=[0, 6, 12, 18, 24, 36, 60],
        labels=['0-6m', '6-12m', '12-18m', '18-24m', '24-36m', '36-60m']
    )

    # Calculate deficit rate by age group
    age_stats = df_with_age.groupby('age_group').agg({
        'deficit': ['sum', 'count', 'mean']
    }).reset_index()

    age_stats.columns = ['age_group', 'deficit_count', 'total_count', 'deficit_rate']
    age_stats['deficit_rate_pct'] = age_stats['deficit_rate'] * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(age_stats['age_group'], age_stats['deficit_rate_pct'],
                  color=['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val, count) in enumerate(zip(bars, age_stats['deficit_rate_pct'], age_stats['deficit_count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=df_with_age['deficit'].mean()*100, color='red', linestyle='--',
               linewidth=2, label=f'Overall Rate: {df_with_age["deficit"].mean()*100:.2f}%')

    ax.set_xlabel('Age Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Deficit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Developmental Deficit Prevalence by Age Group',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    max_rate = age_stats['deficit_rate_pct'].max()
    if pd.notna(max_rate) and max_rate > 0:
        ax.set_ylim(0, max_rate * 1.3)
    else:
        ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_deficit_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_deficit_by_age.png")
    return age_stats


def create_counseling_intensity_analysis():
    """Analyze counseling intensity vs deficit rate."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Create counseling bins
    df['counseling_bins'] = pd.cut(
        df['intensidad_consejeria_window_sum'],
        bins=[-0.1, 0, 2, 4, 6, 10, 100],
        labels=['0', '1-2', '3-4', '5-6', '7-10', '11+']
    )

    # Calculate deficit rate
    counseling_stats = df.groupby('counseling_bins').agg({
        'deficit': ['sum', 'count', 'mean']
    }).reset_index()

    counseling_stats.columns = ['counseling_bins', 'deficit_count', 'total_count', 'deficit_rate']
    counseling_stats['deficit_rate_pct'] = counseling_stats['deficit_rate'] * 100

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Deficit rate by counseling
    ax1 = axes[0]
    bars = ax1.bar(counseling_stats['counseling_bins'], counseling_stats['deficit_rate_pct'],
                   color=sns.color_palette('RdYlGn_r', n_colors=6),
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, val, count in zip(bars, counseling_stats['deficit_rate_pct'], counseling_stats['deficit_count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.axhline(y=df['deficit'].mean()*100, color='red', linestyle='--',
                linewidth=2, label='Overall Rate')
    ax1.set_xlabel('Counseling Sessions (Window)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Deficit Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Deficit Rate by Counseling Intensity', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Sample distribution
    ax2 = axes[1]
    ax2.bar(counseling_stats['counseling_bins'], counseling_stats['total_count'],
            color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.7)

    for i, (x, y) in enumerate(zip(counseling_stats['counseling_bins'], counseling_stats['total_count'])):
        ax2.text(i, y + 20, f'{int(y)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Counseling Sessions (Window)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax2.set_title('Patient Distribution by Counseling Intensity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_counseling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_counseling_analysis.png")
    return counseling_stats


def create_first_year_controls_analysis():
    """Analyze first-year control frequency vs deficit."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Create control bins
    df['controls_bins'] = pd.cut(
        df['n_controles_primer_anio'],
        bins=[-0.1, 2, 4, 6, 8, 10, 20],
        labels=['0-2', '3-4', '5-6', '7-8', '9-10', '11+']
    )

    # Calculate deficit rate
    controls_stats = df.groupby('controls_bins').agg({
        'deficit': ['sum', 'count', 'mean']
    }).reset_index()

    controls_stats.columns = ['controls_bins', 'deficit_count', 'total_count', 'deficit_rate']
    controls_stats['deficit_rate_pct'] = controls_stats['deficit_rate'] * 100

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(controls_stats['controls_bins'], controls_stats['deficit_rate_pct'],
                  color=sns.color_palette('RdYlGn_r', n_colors=6),
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    for bar, val, count in zip(bars, controls_stats['deficit_rate_pct'], controls_stats['deficit_count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}%\n({int(count)} deficits)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=df['deficit'].mean()*100, color='red', linestyle='--',
               linewidth=2, label=f'Overall Rate: {df["deficit"].mean()*100:.2f}%')

    ax.set_xlabel('Number of Controls in First Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Deficit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('First-Year Control Frequency vs Developmental Deficit Risk',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_first_year_controls.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_first_year_controls.png")
    return controls_stats


def create_growth_indicators_comparison():
    """Compare growth indicators between deficit and non-deficit groups."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Select key z-score variables
    zscore_vars = [col for col in df.columns if 'z_' in col.lower() or '_z' in col.lower()]
    zscore_vars = [col for col in zscore_vars if not any(x in col for x in ['slope', 'std', 'min', 'max'])]
    zscore_vars = zscore_vars[:6]  # Top 6

    if not zscore_vars:
        zscore_vars = ['pre6_mean__Peso', 'pre6_mean__Talla', 'pre6_mean__CabPC']

    # Calculate mean by deficit status
    comparison_data = []

    for var in zscore_vars:
        if var in df.columns:
            no_deficit_mean = df[df['deficit'] == 0][var].mean()
            deficit_mean = df[df['deficit'] == 1][var].mean()

            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(
                df[df['deficit'] == 0][var].dropna(),
                df[df['deficit'] == 1][var].dropna(),
                alternative='two-sided'
            )

            comparison_data.append({
                'variable': var.replace('pre6_mean__', '').replace('_', ' '),
                'no_deficit': no_deficit_mean,
                'deficit': deficit_mean,
                'difference': deficit_mean - no_deficit_mean,
                'pvalue': pval
            })

    comp_df = pd.DataFrame(comparison_data)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(comp_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, comp_df['no_deficit'], width,
                   label='No Deficit', color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, comp_df['deficit'], width,
                   label='Deficit', color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Growth Indicator', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Value', fontsize=13, fontweight='bold')
    ax.set_title('Growth Indicators: Deficit vs Non-Deficit Groups',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['variable'], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add significance stars
    for i, (pval, diff) in enumerate(zip(comp_df['pvalue'], comp_df['difference'])):
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        y_pos = max(comp_df['no_deficit'].iloc[i], comp_df['deficit'].iloc[i]) * 1.1
        ax.text(i, y_pos, sig, ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_growth_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_growth_comparison.png")
    return comp_df


def create_correlation_heatmap():
    """Create correlation heatmap of top predictors."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Load feature importance from SHAP stats
    shap_stats_path = settings.FIGURES_DIR / 'interpretability' / 'shap_statistics.csv'

    if shap_stats_path.exists():
        shap_stats = pd.read_csv(shap_stats_path)
        top_features = shap_stats.head(15)['Feature'].tolist()
    else:
        # Fallback to selecting by variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['N_HC', 'deficit', 'ultima ventana']]
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top_features = variances.head(15).index.tolist()

    # Calculate correlation matrix
    corr_matrix = df[top_features].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('Correlation Heatmap - Top 15 Features', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_correlation_heatmap.png")
    return corr_matrix


def create_risk_profile_summary():
    """Create summary visualization of high vs low risk profiles."""

    df = pd.read_csv(settings.PROCESSED_DATA_DIR / "tani_model_ready.csv")

    # Define high/low risk profiles based on deficit
    high_risk = df[df['deficit'] == 1]
    low_risk = df[df['deficit'] == 0].sample(n=min(500, len(df[df['deficit'] == 0])), random_state=42)

    # Key features
    features = {
        'Counseling\nIntensity': 'intensidad_consejeria_window_sum',
        'First Year\nControls': 'n_controles_primer_anio',
        'Vaccine\nCounseling': 'flg_consj_vacunas_sum_prev',
        'Hygiene\nCounseling': 'flg_consj_higne_corporal_sum_prev',
        'Max Age\n(months)': 'pre6_max__edad_meses'
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(features))
    width = 0.35

    high_risk_means = [high_risk[col].mean() for col in features.values()]
    low_risk_means = [low_risk[col].mean() for col in features.values()]

    bars1 = ax.bar(x - width/2, low_risk_means, width,
                   label='Low Risk (No Deficit)', color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, high_risk_means, width,
                   label='High Risk (Deficit)', color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Feature', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Value', fontsize=13, fontweight='bold')
    ax.set_title('Risk Profile Comparison: Key Features', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(features.keys())
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(settings.FIGURES_DIR / 'eda_risk_profile_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Generated: eda_risk_profile_comparison.png")


def main():
    """Generate all EDA visualizations for final report."""

    print("=" * 80)
    print("GENERATING EDA VISUALIZATIONS FOR FINAL REPORT")
    print("=" * 80)

    print("\n1. Age vs Deficit Distribution...")
    create_age_deficit_distribution()

    print("\n2. Counseling Intensity Analysis...")
    create_counseling_intensity_analysis()

    print("\n3. First-Year Controls Analysis...")
    create_first_year_controls_analysis()

    print("\n4. Growth Indicators Comparison...")
    create_growth_indicators_comparison()

    print("\n5. Correlation Heatmap...")
    create_correlation_heatmap()

    print("\n6. Risk Profile Summary...")
    create_risk_profile_summary()

    print("\n" + "=" * 80)
    print("✓ ALL EDA VISUALIZATIONS GENERATED")
    print("=" * 80)
    print(f"Saved to: {settings.FIGURES_DIR}")


if __name__ == "__main__":
    main()
