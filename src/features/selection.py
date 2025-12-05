import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from config import settings

def calculate_iv(df, feature, target):
    """
    Calculates Information Value (IV) for a single feature.
    Handles continuous variables by binning.
    """
    lst = []
    df = df[[feature, target]].copy()
    
    # Check if numeric and needs binning
    if np.issubdtype(df[feature].dtype, np.number) and df[feature].nunique() > 10:
        try:
            df['bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
        except:
            df['bin'] = pd.cut(df[feature], bins=10)
    else:
        df['bin'] = df[feature]
        
    # Calculate Good/Bad stats
    grouped = df.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    grouped['Good'] = grouped['Total'] - grouped['Bad']
    
    total_bad = grouped['Bad'].sum()
    total_good = grouped['Good'].sum()
    
    if total_bad == 0 or total_good == 0:
        return 0.0
        
    grouped['Dist_Bad'] = grouped['Bad'] / total_bad
    grouped['Dist_Good'] = grouped['Good'] / total_good
    
    # Avoid log(0)
    grouped['WoE'] = np.log((grouped['Dist_Good'] + 1e-5) / (grouped['Dist_Bad'] + 1e-5))
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']
    
    return grouped['IV'].sum()

def filter_correlation(df, threshold=0.9):
    """
    Removes highly correlated features.
    Returns list of features to DROP.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def get_base_model_importance(X, y):
    """
    Trains a simple RF to get feature importance.
    """
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns)

def plot_selection_report(iv_series, corr_matrix, importance_series, save_dir):
    """
    Generates plots for the selection process.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. IV Plot (Top 20)
    plt.figure(figsize=(10, 8))
    iv_series.sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Features by Information Value (IV)')
    plt.xlabel('IV')
    plt.tight_layout()
    plt.savefig(save_dir / "iv_top20.png")
    plt.close()
    
    # 2. Correlation Heatmap (Selected Features only if too many)
    # If too many features, heatmap is unreadable. Let's plot top 20 important ones.
    top_feats = importance_series.sort_values(ascending=False).head(20).index
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.loc[top_feats, top_feats], annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap (Top 20 Important Features)')
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_heatmap.png")
    plt.close()
    
    # 3. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    importance_series.sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Features by Random Forest Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_dir / "rf_importance_top20.png")
    plt.close()

def select_features(df, target_col='deficit', iv_threshold=0.02, corr_threshold=0.9):
    """
    Main function to execute feature selection.
    """
    print("--- Starting Feature Selection ---")
    
    X = df.drop(columns=['N_HC', 'ultima ventana', target_col], errors='ignore')
    y = df[target_col]
    
    # 1. Calculate IV for all features
    print("Calculating Information Value (IV)...")
    iv_values = {}
    for col in X.columns:
        iv_values[col] = calculate_iv(df, col, target_col)
    
    iv_series = pd.Series(iv_values).sort_values(ascending=False)
    
    # Filter by IV
    selected_iv = iv_series[iv_series >= iv_threshold].index.tolist()
    print(f"Features passing IV threshold ({iv_threshold}): {len(selected_iv)} / {len(X.columns)}")
    
    # 2. Correlation Filtering (on IV selected features)
    print("Checking Correlation...")
    X_iv = X[selected_iv]
    drop_corr = filter_correlation(X_iv, threshold=corr_threshold)
    print(f"Features dropped due to correlation (>{corr_threshold}): {len(drop_corr)}")
    
    final_features = [f for f in selected_iv if f not in drop_corr]
    print(f"Final feature count: {len(final_features)}")
    
    # 3. Base Model Importance (on Final Features)
    print("Calculating Base Model Importance...")
    importance_series = get_base_model_importance(X[final_features], y)
    
    # Generate Report
    report = pd.DataFrame({
        'IV': iv_series,
        'Selected': iv_series.index.isin(final_features),
        'Dropped_Corr': iv_series.index.isin(drop_corr),
        'RF_Importance': importance_series
    }).sort_values('IV', ascending=False)
    
    # Save Report and Plots
    save_dir = settings.FIGURES_DIR / "selection"
    report.to_csv(settings.REPORTS_DIR / "feature_selection_report.csv")
    plot_selection_report(iv_series, X.corr(), importance_series, save_dir)
    
    print(f"Selection Report saved to {settings.REPORTS_DIR / 'feature_selection_report.csv'}")
    print(f"Plots saved to {save_dir}")
    
    return final_features
