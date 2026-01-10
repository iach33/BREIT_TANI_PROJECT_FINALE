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
│   ├── raw/              # Archivos Excel originales (DATA PROYECTO BREIT.xlsx, etc.)
│   ├── processed/        # Datasets generados por el pipeline
│   └── external/         # Tablas OMS para cálculo de z-scores
├── notebooks/            # Notebooks ejecutables de análisis
│   ├── 01_eda_comprehensive.qmd       # Análisis exploratorio profundo
│   └── 02_model_evaluation.qmd        # Evaluación de modelos y fairness
├── reports/              # Reportes finales y figuras
│   ├── final_report.qmd  # Reporte final del proyecto (MIT)
│   ├── final_report.pdf  # PDF generado
│   └── figures/          # Visualizaciones (EDA, modeling, interpretability)
├── src/
│   ├── config/           # Configuraciones y rutas (settings.py)
│   ├── data/             # Scripts de limpieza y carga
│   ├── features/         # Ingeniería de características (build_features.py, oms_zscores.py)
│   ├── models/           # Entrenamiento, evaluación e interpretabilidad
│   ├── pipelines/        # Scripts de ejecución secuencial
│   │   ├── 01_preprocessing.py  # Consolidación y limpieza
│   │   ├── 02_eda.py            # Análisis exploratorio básico
│   │   └── 03_modeling.py       # Entrenamiento de modelos
│   └── visualization/    # Funciones de ploteo
├── docs/                 # Documentación del proyecto
│   ├── rubrica.jpeg      # Rúbrica de evaluación MIT
│   └── ejemplo_reporte.md # Ejemplo de reporte anterior
├── CLAUDE.md             # Documentación para Claude Code (guía del proyecto)
└── pyproject.toml        # Definición de dependencias
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

### 4. Validación Temporal (Out-of-Time Validation)
Evalúa los modelos en un conjunto de test **temporal** (pacientes observados en periodos futuros).
*   **Split Temporal**: 80% entrenamiento (hasta Junio 2025), 20% test (después Junio 2025).
*   **Evaluación**: Compara rendimiento en test aleatorio vs test temporal.
*   **Degradación**: Mide caída de performance en datos futuros (drift temporal).

```bash
uv run src/pipelines/04_temporal_validation.py
```

**Salidas generadas en `reports/`:**
*   `model_comparison_temporal.csv`: Métricas en test set temporal.
*   `model_comparison_random_vs_temporal.csv`: Comparación de degradación entre test aleatorio y temporal.

**Hallazgos Clave**:
*   Random Forest Optimized: 6.4% degradación (AUC 0.810 → 0.758)
*   Logistic Regression: 5.6% degradación (más estable temporalmente)
*   XGBoost: 22.3% degradación (posible overfitting)

### 5. Interpretabilidad Avanzada (SHAP Comprehensivo)
Genera visualizaciones avanzadas de SHAP para interpretar el modelo ganador.
*   **Gráficos Globales**: Summary plot, bar plot de importancia
*   **Casos Individuales**: Waterfall plots (alto/bajo riesgo), force plots
*   **Relaciones No-Lineales**: Dependence plots para top 6 features
*   **Interacciones**: Interaction plot entre top 2 features
*   **Patrones**: Heatmap de SHAP values (30 casos × 15 features)

```bash
uv run src/pipelines/05_advanced_interpretability.py
```

**Salidas generadas en `reports/figures/interpretability/`:**
*   `shap_summary.png`, `shap_importance.png`: Importancia global
*   `shap_waterfall_high_risk.png`, `shap_waterfall_low_risk.png`: Explicaciones individuales
*   `shap_dependence_1_*.png` a `shap_dependence_6_*.png`: Dependence plots
*   `shap_interaction_top2.png`: Interacción entre top 2 features
*   `shap_heatmap.png`: Patrones de SHAP values
*   `shap_force_high_risk.png`: Force plot caso alto riesgo
*   `shap_statistics.csv`: Estadísticas de SHAP por feature
*   `shap_feature_directions.csv`: Análisis de direccionalidad

**Insights Clave**:
*   **Intensidad de consejería**: Efecto protector fuerte (SHAP: -0.023)
*   **Edad máxima en ventana**: Mayor edad = menor riesgo (SHAP: -0.019)
*   **Consejería en vacunas**: Proxy de engagement parental (SHAP: -0.022)
*   **Threshold crítico**: 5+ sesiones de consejería para protección óptima

---

## 📓 Notebooks de Análisis Comprehensivo

El proyecto incluye notebooks ejecutables en formato **Quarto** (`.qmd`) para análisis profundo y reproducible.

### 1. EDA Comprehensivo (`notebooks/01_eda_comprehensive.qmd`)

Análisis exploratorio riguroso alineado con estándares académicos (MIT):

**Contenido:**
*   **Data Quality Assessment**: Análisis de valores faltantes, outliers, distribuciones
*   **Univariate Analysis**: Estadísticas descriptivas robustas (media, mediana, skewness, kurtosis)
*   **Bivariate Analysis**: Correlaciones, tests estadísticos (Mann-Whitney U, Chi-cuadrado)
*   **Subgroup Analysis**: Análisis estratificado por edad y sexo
*   **Advanced Visualizations**: Violin plots, pairplots, correlation heatmaps, mutual information

**Ejecución:**
```bash
# Renderizar a HTML
quarto render notebooks/01_eda_comprehensive.qmd
```

**Salidas:**
*   `notebooks/01_eda_comprehensive.html`: Reporte HTML interactivo
*   `reports/figures/`: Gráficos avanzados (pairplots, heatmaps, boxplots, etc.)

### 2. Evaluación de Modelos y Fairness (`notebooks/02_model_evaluation.qmd`)

Evaluación rigurosa de robustez, estabilidad y equidad del modelo:

**Contenido:**
*   **Experimental Design**: Documentación de estrategia de split, cross-validation, manejo de desbalance
*   **Robustness Analysis**:
    - Learning curves (tamaño de datos vs performance)
    - Cross-validation stability (15 folds × 3 repeticiones)
    - Performance segmentado por edad
*   **Fairness Evaluation**:
    - Análisis de equidad por sexo (AUC parity, precision parity)
    - Trade-off precision-recall
*   **Ethical Considerations**: Costos de errores, limitaciones, recomendaciones de deployment

**Ejecución:**
```bash
# Renderizar a HTML
quarto render notebooks/02_model_evaluation.qmd
```

**Salidas:**
*   `notebooks/02_model_evaluation.html`: Reporte de evaluación completo
*   `reports/figures/modeling/`: Learning curves, stability plots, fairness comparisons

---

### 3. Reporte Final MIT (`reports/final_report.qmd`)

Reporte consolidado para entrega al MIT, integrando todos los análisis:

**Contenido:**
*   Executive Summary con hallazgos clave
*   Introducción y contexto (TANI, desarrollo infantil, objetivos)
*   Data Consolidation (pipeline de limpieza)
*   **EDA Summary** (con referencias a notebook 01)
*   Modeling Methodology y Feature Selection
*   **Model Results** (comparación de 8 modelos)
*   **Robustness & Fairness Analysis** (con referencias a notebook 02)
*   Ethical Considerations & Limitations
*   Conclusions & Recommendations (12 recomendaciones accionables)
*   References y Appendices

**Renderizado a PDF:**
```bash
# Generar PDF final para entrega
quarto render reports/final_report.qmd --to pdf
```

**Salida:**
*   `reports/final_report.pdf`: Reporte final listo para entrega MIT

---

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
