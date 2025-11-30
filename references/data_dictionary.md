# Diccionario de Datos - TANI Model Ready

Este documento describe las variables contenidas en el dataset final para modelado: `data/processed/tani_model_ready.csv`.

Este dataset tiene una estructura de **una fila por paciente** (`N_HC`), donde las variables resumen la historia clínica reciente (ventana de 6 controles) y eventos históricos clave (primer año, nacimiento).

## 1. Identificadores y Target

| Variable | Descripción | Tipo |
| :--- | :--- | :--- |
| `N_HC` | Número de Historia Clínica (Identificador único del paciente). | ID |
| `deficit` | **Variable Objetivo (Target)**. <br> `1`: El niño presentó algún déficit (lenguaje, social, motor, cognitivo) en el control *inmediatamente posterior* a la ventana observada. <br> `0`: No presentó déficits. | Binaria |
| `ultima ventana` | Fecha o índice del último control considerado en la ventana de observación. | Metadata |

## 2. Estadísticas de Ventana (Últimos 6 Controles)

Estas variables se calculan agregando los datos de los últimos 6 controles disponibles del paciente.
Prefijo: `pre6_`

| Variable (Patrón) | Descripción |
| :--- | :--- |
| `pre6_mean__[Variable]` | Promedio de la variable en los últimos 6 controles. |
| `pre6_min__[Variable]` | Valor mínimo en los últimos 6 controles. |
| `pre6_max__[Variable]` | Valor máximo en los últimos 6 controles. |
| `pre6_std__[Variable]` | Desviación estándar (variabilidad) en los últimos 6 controles. |

**Variables Agregadas:**
*   `Peso`, `Talla`, `CabPC` (Perímetro Cefálico).
*   `edad_meses`: Edad del niño.
*   `control_esperado`: Número de control según norma técnica.
*   `_TE_z`, `_PE_z`, `_PT_z`: Z-scores de Talla/Edad, Peso/Edad, Peso/Talla (Originales del dataset).
*   `zscore_peso_edad`, `zscore_talla_edad`, `zscore_peso_talla`: **Z-scores calculados con tablas OMS (LMS)**.

## 3. Tendencias (Slopes)

Calculadas usando regresión lineal sobre los puntos de la ventana (eje X = tiempo, eje Y = medida). Indican la velocidad de crecimiento.

| Variable | Descripción |
| :--- | :--- |
| `slope_peso` | Pendiente de ganancia de peso (kg/mes aprox). |
| `slope_talla` | Pendiente de crecimiento en talla (cm/mes aprox). |
| `slope_cab_pc` | Pendiente de crecimiento del perímetro cefálico. |

## 4. Consejería (Window)

Miden la recepción de charlas y orientaciones durante la ventana de observación.

| Variable | Descripción |
| :--- | :--- |
| `flg_consj_[Tipo]_valor` | Si recibió la consejería en el *último* control de la ventana (1/0). |
| `flg_consj_[Tipo]_sum_prev` | Cantidad total de veces que recibió esta consejería en la ventana (0-6). |
| `intensidad_consejeria_window_sum` | Suma total de todas las consejerías recibidas en la ventana. |

**Tipos de Consejería:**
*   `lact_materna`: Lactancia Materna.
*   `higne_corporal`: Higiene Corporal.
*   `higne_bucal`: Higiene Bucal.
*   `supl_hierro`: Suplementación con Hierro.
*   `desarrollo`: Estimulación del Desarrollo.
*   `vacunas`: Vacunación.

## 5. Flags de Estado (Window)

Resumen de diagnósticos durante la ventana de observación.

| Variable | Descripción |
| :--- | :--- |
| `flg_anemia_window` | 1 si tuvo anemia (Hb < 11) en *algún* control de la ventana. |
| `flg_desnutricion_cronica_window` | 1 si tuvo desnutrición crónica en *algún* control de la ventana. |
| `flg_desnutricion_aguda_window` | 1 si tuvo desnutrición aguda en *algún* control de la ventana. |
| `flg_sobrepeso_window` | 1 si tuvo sobrepeso en *algún* control de la ventana. |
| `flg_desnutricion` | Flag general de desnutrición en la ventana. |
| `porc_desnutricion` | Porcentaje de controles en la ventana con diagnóstico de desnutrición. |
| `flg_asiste_control_esperado` | 1 si el número de control real coincide con el esperado para la edad. |
| `Cantidad_acompañantes` | Promedio o conteo de acompañantes distintos (proxy de soporte familiar). |

## 6. Antecedentes de Nacimiento

Variables estáticas provenientes del diagnóstico de nacimiento.

| Variable | Descripción |
| :--- | :--- |
| `flg_bajo_peso_nacer` | 1 si nació con bajo peso (< 2500g). |
| `flg_macrosomia` | 1 si nació con macrosomía (> 4000g). |

## 7. Primer Año de Vida (0-12 meses)

Resumen del historial temprano del niño.

| Variable | Descripción |
| :--- | :--- |
| `n_controles_primer_anio` | Cantidad total de controles recibidos entre 0 y 12 meses. |
| `flg_desnutricion_primer_anio` | 1 si tuvo algún diagnóstico nutricional adverso en el primer año. |
| `flg_anemia_primer_anio` | 1 si tuvo anemia en el primer año. |

## 8. Hitos de Desarrollo (Milestones)

Estado nutricional (Z-scores) capturado en edades clave. Si el dato exacto no existe, se imputa con la mediana poblacional.

| Variable | Descripción |
| :--- | :--- |
| `z_PT_12m`, `z_TE_12m`, `z_PE_12m` | Z-scores (Peso/Talla, Talla/Edad, Peso/Edad) a los **12 meses**. |
| `z_PT_24m`, `z_TE_24m`, `z_PE_24m` | Z-scores a los **24 meses**. |
| `z_PT_36m`, `z_TE_36m`, `z_PE_36m` | Z-scores a los **36 meses**. |
