# TANI Project - Análisis de Desarrollo Infantil

Este proyecto implementa un pipeline de procesamiento de datos y análisis para la ONG TANI, enfocado en predecir riesgos de déficit en el desarrollo infantil (lenguaje, social, cognitivo, motor) basándose en datos históricos de controles de salud.

## 🚀 Requisitos Previos

Este proyecto utiliza **[uv](https://github.com/astral-sh/uv)** para la gestión de dependencias y entornos virtuales, lo que garantiza una ejecución rápida y reproducible.

1.  **Instalar uv**:
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## 📦 Instalación

Una vez instalado `uv`, clona este repositorio y sincroniza las dependencias:

```bash
# Instalar dependencias definidas en pyproject.toml
uv sync
```

## 📂 Estructura del Proyecto

```text
├── data/
│   ├── raw/            # Archivos Excel originales (DATA PROYECTO BREIT.xlsx, etc.)
│   └── processed/      # Datasets generados por el pipeline
├── notebooks/          # Notebooks de exploración (Jupyter/Quarto)
├── reports/            # Reportes generados y figuras
├── src/
│   ├── config/         # Configuraciones y rutas (settings.py)
│   ├── data/           # Scripts de limpieza y carga
│   ├── features/       # Ingeniería de características (build_features.py)
│   ├── pipelines/      # Scripts de ejecución (01_preprocessing.py, 02_eda.py)
│   └── visualization/  # Funciones de ploteo
└── pyproject.toml      # Definición de dependencias
```

## ⚙️ Ejecución de Pipelines

El proyecto está modularizado en pipelines que se ejecutan con `uv run`.

### 1. Preprocesamiento y Feature Engineering

Este pipeline carga los datos crudos, limpia, genera features (ventanas históricas, anemia, nutrición) y crea los datasets finales.

```bash
uv run src/pipelines/01_preprocessing.py
```

**Salidas generadas en `data/processed/`:**
*   `tani_analytical_dataset.csv`: Dataset longitudinal completo (historia de controles).
*   `tani_patient_features.csv`: Dataset a nivel paciente con features agregados.
*   `tani_model_ready.csv`: **Dataset final para modelado** (limpio, imputado y sin columnas constantes).

### 2. Análisis Exploratorio de Datos (EDA)

Genera visualizaciones automáticas basadas en el dataset listo para modelar.

```bash
uv run src/pipelines/02_eda.py
```

**Salidas generadas en `reports/figures/`:**
*   `eda_histograms_model_ready.png`: Histogramas de las variables numéricas.
*   `eda_histograms_by_deficit.png`: Comparación de distribuciones según el target `deficit`.

## 📊 Diccionario de Datos (Salidas)

Para un detalle completo de cada variable, consulta el [Diccionario de Datos](references/data_dictionary.md).

| Archivo | Descripción | Uso Principal |
| :--- | :--- | :--- |
| **`tani_model_ready.csv`** | Una fila por paciente. Contiene features de ventana (últimos 6 controles), features del primer año de vida, e intensidad de consejería. Sin nulos. | **Entrenamiento de Modelos** |
| `tani_patient_features.csv` | Igual que el anterior pero sin imputación de nulos y con todas las columnas generadas. | Análisis detallado / Debugging |
| `tani_analytical_dataset.csv` | Dataset transaccional (una fila por control). Contiene la historia completa día a día. | Análisis de series de tiempo / Deep Learning |

## 🎯 Target del Modelo

La variable objetivo es **`deficit`**:
*   `1`: El paciente presentó algún déficit (lenguaje, social, etc.) en el control inmediatamente posterior a la ventana analizada.
*   `0`: El paciente no presentó déficits.
