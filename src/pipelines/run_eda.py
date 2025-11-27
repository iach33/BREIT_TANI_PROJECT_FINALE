import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
from config import settings
from visualization import visualize

def main():
    print("=== Starting EDA Pipeline ===")
    
    # 1. Load Processed Data
    input_file = settings.PROCESSED_DATA_DIR / "tani_analytical_dataset.csv"
    print(f"Loading processed data from: {input_file}")
    
    if not input_file.exists():
        print(f"Error: File not found. Please run 'src/pipelines/run_preprocessing.py' first.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # 2. Visualization
    print("Generating Visualizations...")
    
    # Define output path
    output_dir = settings.PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eda_histograms.png"
    
    # Select numeric columns to plot
    # We can be more specific now that we know the features
    numeric_cols = ['edad_meses', 'Peso', 'Talla', 'CabPC']
    # Filter only existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    if numeric_cols:
        print(f"Plotting histograms for: {numeric_cols}")
        visualize.plot_grid_hist(
            df, 
            cols=numeric_cols, 
            title="EDA Histograms - Analytical Dataset", 
            save_path=str(output_path)
        )
    else:
        print("No numeric columns found to visualize.")
    
    print("=== EDA Completed Successfully ===")

if __name__ == "__main__":
    main()
