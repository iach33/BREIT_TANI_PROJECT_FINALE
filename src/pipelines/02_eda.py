import sys
from pathlib import Path
import pandas as pd

# Add src to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import settings
from visualization import visualize

def main():
    print("=== Starting EDA Pipeline ===")
    
    # Load processed data
    # Use the model-ready dataset for EDA to reflect what the model sees
    input_file = settings.PROCESSED_DATA_DIR / "tani_model_ready.csv"
    print(f"Loading processed data from: {input_file}")
    
    if not input_file.exists():
        print("Processed data not found. Run 'src/pipelines/run_preprocessing.py' first.")
        return

    df = pd.read_csv(input_file)
    print(f"Dataset shape: {df.shape}")
    
    # Generate Histograms
    print("Generating histograms...")
    # Filter numeric columns for histogram
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Exclude ID and non-informative columns
    cols_to_plot = [c for c in numeric_cols if c not in ['N_HC', 'Unnamed: 0']]
    
    # Plot top 16 features to avoid overcrowding
    cols_to_plot = cols_to_plot[:16]
    
    output_hist = settings.FIGURES_DIR / "eda_histograms_model_ready.png"
    print(f"Saving histograms to: {output_hist}")
    
    visualize.plot_grid_hist(
        df, 
        cols_to_plot, 
        ncols=4, 
        save_path=output_hist
    )
    
    # Generate Histograms by Deficit
    if 'deficit' in df.columns:
        print("Generating histograms by deficit...")
        output_hist_deficit = settings.FIGURES_DIR / "eda_histograms_by_deficit.png"
        print(f"Saving histograms by deficit to: {output_hist_deficit}")
        
        visualize.plot_grid_hist_by_deficit(
            df,
            cols_to_plot,
            tag='deficit',
            ncols=4,
            save_path=output_hist_deficit
        )
    
    print("=== EDA Completed Successfully ===")

if __name__ == "__main__":
    main()
