# TANI Scoring Pipeline

Pipeline de prediccion de deficit en desarrollo infantil basado en [Kedro](https://kedro.org/), para Asociacion Taller de los Ninos (TANI).

## Estructura

```
tani_scoring/
├── conf/base/           # Configuracion (parameters.yml, catalog.yml)
├── data/
│   ├── 01_raw/          # Input: Excel mensual + consejerias
│   ├── 04_models/       # Artefactos del modelo (model.joblib, features, metadata)
│   ├── 05_scores/       # Output: scores de prediccion
│   └── 06_external/     # Tablas OMS (8 archivos Excel)
├── src/tani_scoring/
│   ├── pipelines/
│   │   ├── data_processing/      # Carga, limpieza, merge
│   │   ├── feature_engineering/  # Feature engineering
│   │   ├── prediction/           # Scoring
│   │   └── model_export/         # Entrenamiento y exportacion (one-time)
│   ├── pipeline_registry.py
│   └── settings.py
└── tests/
```

## Pipelines

| Pipeline | Descripcion | Comando |
|----------|-------------|---------|
| `__default__` | Scoring completo (data → features → predict) | `kedro run` |
| `data_processing` | Solo carga y limpieza | `kedro run --pipeline data_processing` |
| `feature_engineering` | Solo ingenieria de features | `kedro run --pipeline feature_engineering` |
| `prediction` | Solo prediccion (requiere features) | `kedro run --pipeline prediction` |
| `model_export` | Entrenar y exportar modelo (one-time) | `kedro run --pipeline model_export` |

## Setup

```bash
cd tani_scoring
uv sync
```

## Uso

### 1. Exportar modelo (una vez, o al re-entrenar)

Requiere que exista `data/processed/tani_model_ready.csv` del pipeline de entrenamiento original.

```bash
kedro run --pipeline model_export
```

Esto genera en `data/04_models/`:
- `model.joblib` — Pipeline sklearn (imputer + scaler + classifier, sin SMOTE)
- `selected_features.json` — Lista de features
- `model_metadata.json` — AUC, fecha, numero de features

### 2. Preparar datos de entrada

Copiar a `data/01_raw/`:
- Archivo Excel principal (formato `DATA PROYECTO BREIT.xlsx`): hojas DESNUTRICION + DESARROLLO
- Archivo Excel de consejerias (formato `UBIGEO - CONSEJERÍAS.xlsx`): hoja CONSEJERÍAS

Copiar tablas OMS a `data/06_external/` (8 archivos Excel).

Ajustar `conf/base/parameters.yml` si los nombres de archivo son distintos.

### 3. Scoring mensual

```bash
kedro run
```

El reporte de scores se genera en `data/05_scores/score_report.csv` con:

| Columna | Descripcion |
|---------|-------------|
| `N_HC` | ID del paciente |
| `risk_score` | Probabilidad de deficit (0.0 - 1.0) |
| `risk_category` | "Alto" (>=0.7), "Medio" (0.3-0.7), "Bajo" (<0.3) |
| `scoring_date` | Fecha de ejecucion |
| `n_controls` | Controles en la ventana |
| `last_control_number` | Ultimo control esperado |

### 4. Visualizar DAG

```bash
kedro viz
```

## Diferencias vs pipeline de entrenamiento

| Aspecto | Entrenamiento (`src/`) | Prediccion (`tani_scoring/`) |
|---------|----------------------|------------------------------|
| Referencia en window | `primer_alguna` (primer deficit) | `ultimo_control` (estado actual) |
| Variable `deficit` | Se calcula como target | No se calcula (es lo que se predice) |
| Data antigua (2009-2016) | Se incluye para historico | No se incluye |
| Filtro temporal | Fecha >= 2023 | Sin filtro (toda la data es reciente) |
| SMOTE | En pipeline de entrenamiento | No (pipeline exportado sin SMOTE) |

## Tests

```bash
cd tani_scoring
pytest tests/ -v
```
