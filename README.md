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

### 3. Modelado Predictivo y Optimización
Entrena y evalúa modelos de Machine Learning (Logistic Regression, Random Forest, XGBoost, LightGBM).
*   **Optimización**: Para cada algoritmo, entrena una versión Baseline y una Optimizada (`RandomizedSearchCV`).
*   **Selección**: Elige automáticamente el mejor modelo basado en AUC.
*   **Interpretabilidad**: Genera gráficos SHAP del modelo ganador.

```bash
uv run src/pipelines/03_modeling.py
```

**Salidas generadas en `reports/`:**
*   `model_comparison.csv`: Tabla comparativa de métricas (AUC, Precision, Recall, F1).
*   `figures/modeling/`: Gráficos de evaluación (Matrices de Confusión, Curvas ROC, Feature Importance).
*   `figures/interpretability/`: **Análisis SHAP** del mejor modelo (Summary Plot, Global Importance).

## 📊 Diccionario de Datos (Salidas)

Para un detalle completo de cada variable, consulta el [Diccionario de Datos](references/data_dictionary.md).

| Archivo | Descripción | Uso Principal |
| :--- | :--- | :--- |
| **`tani_model_ready.csv`** | Una fila por paciente. Contiene features de ventana (últimos 6 controles), features del primer año de vida, e intensidad de consejería. Sin nulos. | **Entrenamiento de Modelos** |
| `tani_patient_features.csv` | Igual que el anterior pero sin imputación de nulos y con todas las columnas generadas. | Análisis detallado / Debugging |
| `tani_analytical_dataset.csv` | Dataset transaccional (una fila por control). Contiene la historia completa día a día. | Análisis de series de tiempo / Deep Learning |

*   `0`: El paciente no presentó déficits.

## 🧠 Metodología de Modelamiento

El pipeline de modelado (`src/pipelines/03_modeling.py`) sigue un enfoque riguroso para garantizar robustez y explicabilidad:

### 1. Selección de Variables (`src/features/selection.py`)
Antes del entrenamiento, se seleccionan las variables más relevantes para reducir ruido y dimensionalidad:
*   **Information Value (IV)**: Se descartan variables con bajo poder predictivo (IV < 0.02).
*   **Filtro de Correlación**: Se eliminan variables redundantes con correlación > 0.9.
*   **Importancia Base**: Se utiliza un Random Forest preliminar para validar la importancia.

### 2. Preparación de Datos
*   **Split**: División estratificada 80/20 (Train/Test).
*   **Imputación**: Mediana para valores faltantes.
*   **Escalamiento**: `StandardScaler` para normalizar features.
*   **Balanceo**: `SMOTE` aplicado solo al conjunto de entrenamiento para manejar el desbalance de clases (~3% de casos positivos).

### 3. Entrenamiento y Optimización (`src/models/train_model.py`)
Se entrenan 4 algoritmos, cada uno con dos estrategias:
*   **Algoritmos**: Logistic Regression, Random Forest, XGBoost, LightGBM.
*   **Estrategias**:
    1.  **Baseline**: Hiperparámetros por defecto.
    2.  **Optimized**: Búsqueda aleatoria (`RandomizedSearchCV`) con validación cruzada estratificada (3-fold).
*   **Total**: 8 modelos candidatos compiten por el mejor AUC.

### 4. Evaluación e Interpretabilidad (`src/models/interpretability.py`)
*   **Selección**: El modelo con mayor AUC en el set de prueba es declarado ganador.
*   **SHAP (SHapley Additive exPlanations)**: Se calculan los valores SHAP del modelo ganador para explicar:
    *   **Impacto Global**: Qué variables influyen más en la predicción.
    *   **Direccionalidad**: Cómo valores altos/bajos de una variable afectan la probabilidad de riesgo.
