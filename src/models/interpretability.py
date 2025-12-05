import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import settings

def select_best_model(results_df, trained_models):
    """
    Selects the best model based on AUC from the results dataframe.
    """
    best_row = results_df.loc[results_df['AUC'].idxmax()]
    best_model_name = best_row['Model']
    best_auc = best_row['AUC']
    
    print(f"\n--- Best Model Selection ---")
    print(f"Selected Model: {best_model_name} (AUC: {best_auc:.4f})")
    
    return trained_models[best_model_name], best_model_name

def explain_model(model_pipeline, X_test, model_name):
    """
    Generates SHAP plots for the given model.
    """
    save_dir = settings.FIGURES_DIR / "interpretability"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Calculating SHAP values for {model_name}...")
    
    # Extract step from pipeline
    # Note: SHAP needs the transformed X, not the raw X_test
    # We need to apply the preprocessing steps (imputer, scaler) manually
    
    try:
        preprocessor = model_pipeline.named_steps['scaler'] # Assuming scaler is the last step before classifier
        # But wait, we have imputer -> scaler -> smote (only train) -> classifier
        # For test data, we just need imputer -> scaler
        
        X_test_transformed = X_test.copy()
        
        if 'imputer' in model_pipeline.named_steps:
            imputer = model_pipeline.named_steps['imputer']
            X_test_transformed = pd.DataFrame(imputer.transform(X_test_transformed), columns=X_test.columns)
            
        if 'scaler' in model_pipeline.named_steps:
            scaler = model_pipeline.named_steps['scaler']
            X_test_transformed = pd.DataFrame(scaler.transform(X_test_transformed), columns=X_test.columns)
            
        classifier = model_pipeline.named_steps['classifier']
        
        # Create Explainer
        # TreeExplainer for Tree models, LinearExplainer for LogReg, etc.
        is_tree_model = any(name in model_name for name in ['XGBoost', 'LightGBM', 'RandomForest'])
        
        if is_tree_model:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_transformed)
        else:
            # Fallback for Linear models or others
            # Use a sample for KernelExplainer if dataset is large, but here X_test is small enough
            explainer = shap.LinearExplainer(classifier, X_test_transformed)
            shap_values = explainer.shap_values(X_test_transformed)

        # Handle SHAP values format (list for binary classification in some versions)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Positive class
        elif len(shap_values.shape) == 3:
             # If (N, M, 2), take positive class
             shap_values = shap_values[:, :, 1]
            
        # 1. Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_transformed, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(save_dir / "shap_summary.png")
        plt.close()
        
        # 2. Bar Plot (Global Importance)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", show=False)
        plt.title(f'SHAP Global Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(save_dir / "shap_importance.png")
        plt.close()
        
        print(f"SHAP plots saved to {save_dir}")
        
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        import traceback
        traceback.print_exc()
