import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from models import train_model, evaluate_model, interpretability
from features import selection
from config import settings
import pandas as pd

def main():
    print("=== Starting Modeling Pipeline ===")
    
    # 1. Load Data
    data_path = settings.PROCESSED_DATA_DIR / "tani_model_ready.csv"
    print(f"Loading data from {data_path}...")
    df = train_model.load_data(data_path)
    
    # 1.5 Feature Selection
    print("Performing Feature Selection...")
    selected_features = selection.select_features(df, target_col='deficit')
    
    # Filter dataframe to keep only selected features + metadata + target
    cols_to_keep = ['N_HC', 'ultima ventana', 'deficit'] + selected_features
    df = df[cols_to_keep]
    
    # 2. Split Data
    print("Splitting features and target...")
    X, y = train_model.get_feature_target_split(df)
    
    # 3. Train Models
    print("Training models (this may take a while)...")
    trained_models, X_test, y_test = train_model.train_models(X, y)
    
    # 4. Evaluate Models
    print("Evaluating models...")
    results_df = evaluate_model.evaluate_models(trained_models, X_test, y_test)
    
    # 5. Interpretability
    print("Running Interpretability Analysis...")
    best_model, best_model_name = interpretability.select_best_model(results_df, trained_models)
    interpretability.explain_model(best_model, X_test, best_model_name)
    
    print("\n=== Modeling Pipeline Completed Successfully ===")
    print(f"Results saved to {settings.REPORTS_DIR / 'model_comparison.csv'}")
    print(f"Figures saved to {settings.FIGURES_DIR / 'modeling'} and {settings.FIGURES_DIR / 'interpretability'}")

if __name__ == "__main__":
    main()
