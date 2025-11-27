from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# File paths
DATA_FILE_NEW = RAW_DATA_DIR / "DATA PROYECTO BREIT.xlsx"
DATA_FILE_OLD = RAW_DATA_DIR / "DATA 2009-2016.xlsx"

# Column names (Constants)
COL_FECHA = "Fecha"
COL_NHC = "N_HC" # Or Nº_HC, need to standardize
COL_EDAD = "Edad"
COL_PESO = "Peso"
COL_TALLA = "Talla"
COL_CABPC = "CabPC"
COL_SEXO = "Sexo"
