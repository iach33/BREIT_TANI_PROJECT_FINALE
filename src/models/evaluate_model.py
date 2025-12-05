import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from pathlib import Path
from config import settings

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / f"cm_{model_name}.png")
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name, save_path):
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path / f"roc_{model_name}.png")
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_path):
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1]
    top_n = 20
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.barh(range(top_n), importances[top_indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path / f"fi_{model_name}.png")
    plt.close()

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates trained models and saves plots.
    """
    save_dir = settings.FIGURES_DIR / "modeling"
    ensure_dir(save_dir)
    
    results = []
    
    for name, pipeline in trained_models.items():
        print(f"\n--- Evaluating {name} ---")
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"AUC: {auc:.4f}")
        print(f"Recall (Deficit): {report['1']['recall']:.4f}")
        print(f"Precision (Deficit): {report['1']['precision']:.4f}")
        
        results.append({
            'Model': name,
            'AUC': auc,
            'Recall': report['1']['recall'],
            'Precision': report['1']['precision'],
            'F1': report['1']['f1-score']
        })
        
        # Plots
        plot_confusion_matrix(y_test, y_pred, name, save_dir)
        plot_roc_curve(y_test, y_proba, name, save_dir)
        
        # Feature Importance (Extract classifier from pipeline)
        classifier = pipeline.named_steps['classifier']
        feature_names = X_test.columns
        plot_feature_importance(classifier, feature_names, name, save_dir)

    # Save summary metrics
    df_results = pd.DataFrame(results)
    df_results.to_csv(settings.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(df_results)
    
    return df_results
