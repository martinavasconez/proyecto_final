# Data Mining – Proyecto Final: Trading Algorítmico con ML

##  Descripción General

Este proyecto implementa un **pipeline completo de ingesta, feature engineering y machine learning** para predecir la dirección diaria del precio de acciones (AAPL), con el objetivo de simular una estrategia de trading algorítmico y evaluar su rentabilidad en condiciones reales de mercado.

**Modalidad:** Individual (1 activo, sin Twitter, datos de mercado únicamente)

### Fases del Proyecto

**Proyecto #6 - Pipeline de Datos**
- Ingesta de datos históricos de mercado (OHLCV) desde Yahoo Finance
- Construcción de features diarias en PostgreSQL
- Servicio automatizado de feature engineering con Docker

**Proyecto Final - Machine Learning & Trading**
- Entrenamiento y comparación de 8 modelos de clasificación
- Selección del mejor modelo mediante validación temporal
- Simulación de inversión con capital inicial de USD 10,000
- Despliegue del modelo como API REST con FastAPI

---

##  Problema de Negocio

**Objetivo:** Predecir si una acción cerrará más alta o más baja que su precio de apertura en un día dado.

**Variable objetivo:**
```python
target_up = 1  # si close > open (día alcista)
target_up = 0  # si close <= open (día bajista o neutro)
```

**Aplicación práctica:**
- Estrategia de trading: comprar en apertura cuando se predice subida
- Gestión de riesgo: permanecer en efectivo cuando se predice bajada
- Optimización de timing: decidir cuándo operar y cuándo no

**Restricción crítica:** Solo se pueden usar features disponibles **antes o en el momento de la apertura** del mercado (sin data leakage).

---

##  Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    YAHOO FINANCE API                        │
│                  (Datos históricos OHLCV)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PROYECTO #6: DATA ENGINEERING                  │
├─────────────────────────────────────────────────────────────┤
│  1. jupyter-notebook                                         │
│     └─ 01_ingesta_prices_raw.ipynb                          │
│        └─ Descarga OHLCV → raw.prices_daily                 │
│                                                              │
│  2. postgres (Database)                                      │
│     ├─ Schema: raw                                           │
│     │  └─ raw.prices_daily (datos crudos)                   │
│     └─ Schema: analytics                                     │
│        └─ analytics.daily_features (features para ML)       │
│                                                              │
│  3. feature-builder (Docker Worker)                          │
│     └─ build_features.py (CLI)                              │
│        └─ Transforma raw → analytics                        │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           PROYECTO FINAL: MACHINE LEARNING                  │
├─────────────────────────────────────────────────────────────┤
│  4. ml_trading_classifier.ipynb                              │
│     ├─ EDA y preprocesamiento                               │
│     ├─ Entrenamiento de 8 modelos                           │
│     ├─ Selección del mejor modelo                           │
│     ├─ Evaluación en test                                   │
│     └─ Simulación de inversión (USD 10,000 en 2025)        │
│                                                              │
│  5. model-api (FastAPI)                                      │
│     ├─ main.py (endpoints REST)                             │
│     ├─ /predict (predicción individual)                     │
│     ├─ /predict/batch (predicciones múltiples)              │
│     └─ /model/info (información del modelo)                 │
└─────────────────────────────────────────────────────────────┘
```

---

##  Datos y Features

### Activos Analizados

**Ticker principal:** AAPL (Apple Inc.)

**Período histórico:** 2018-01-09 a 2025-12-05 (1,989 días bursátiles, ~8 años)

### Schema RAW: `raw.prices_daily`

Tabla con datos crudos descargados de Yahoo Finance:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | DATE | Fecha del día bursátil |
| `ticker` | TEXT | Símbolo del activo (AAPL) |
| `open` | DOUBLE PRECISION | Precio de apertura |
| `high` | DOUBLE PRECISION | Precio máximo del día |
| `low` | DOUBLE PRECISION | Precio mínimo del día |
| `close` | DOUBLE PRECISION | Precio de cierre |
| `adj_close` | DOUBLE PRECISION | Precio ajustado por splits/dividendos |
| `volume` | BIGINT | Volumen de transacciones |
| `run_id` | TEXT | ID de ejecución de ingesta |
| `ingested_at_utc` | TIMESTAMP | Timestamp de ingesta |
| `source_name` | TEXT | Fuente de datos (yfinance) |

**Estadísticas del período:**
- **Precio mínimo:** $35.55 (2018)
- **Precio máximo:** $286.19 (2025)
- **Crecimiento:** ~8x en 8 años
- **Volatilidad promedio:** 1.65% diaria

### Schema ANALYTICS: `analytics.daily_features`

**Grano:** 1 fila = 1 día bursátil por ticker

**Total de features:** 14 (12 numéricas + 2 categóricas)

#### Features Numéricas (12):

**Lags de precios del día anterior (5):**
- `open_prev_day`, `high_prev_day`, `low_prev_day`, `close_prev_day`, `volume_prev_day`

**Retornos y volatilidad (2):**
- `ret_prev_day`: Retorno del día anterior
- `volatility_prev_5`: Desviación estándar de retornos últimos 5 días

**Rolling features (2):**
- `volume_avg_7`: Volumen promedio últimos 7 días
- `price_avg_7`: Precio promedio últimos 7 días

**Indicadores técnicos (2):**
- `daily_range_prev`: (high - low) / close del día anterior
- `momentum_3`: Retorno acumulado últimos 3 días

**RSI (1):**
- `rsi_proxy`: Relative Strength Index (14 días)

#### Features Categóricas (2):
- `day_of_week`: 0=Lunes, 4=Viernes
- `month`: 1=Enero, 12=Diciembre

**Todas las features usan `.shift(1)` para prevenir data leakage** → Solo información disponible en la apertura.

---

## Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone <repo-url>
cd proyecto_trading_final
```

### 2. Configurar Variables de Ambiente

```bash
cp .env.example .env
nano .env  # Editar con tus credenciales
```

**Contenido de `.env`:**

```bash
# PostgreSQL
PG_HOST=
PG_PORT=
PG_DB=
PG_USER=
PG_PASSWORD=
PG_SCHEMA_RAW=
PG_SCHEMA_ANALYTICS=

# Ingesta de Mercado
TICKERS=
START_DATE=
END_DATE=
DATA_PROVIDER=
RUN_ID=

# Feature Builder
FB_MODE=
FB_OVERWRITE=

# API del Modelo
MODEL_PATH=
API_PORT=
API_ENV=
```

### 3. Levantar Infraestructura

```bash
# Levantar servicios
docker-compose up -d

# Verificar
docker-compose ps
```

### Servicios Disponibles

Al levantar la infraestructura con Docker Compose, se despliegan los siguientes servicios:

- **PostgreSQL**
  - Base de datos principal del proyecto
  - Puerto: `localhost:5432`
  - Contiene los schemas:
    - `raw` → datos crudos de mercado
    - `analytics` → features para machine learning

- **PgAdmin** (opcional)
  - Interfaz web para administrar PostgreSQL
  - URL: http://localhost:8081
  - Permite inspeccionar tablas, ejecutar queries y validar datos

- **Jupyter Notebook**
  - Entorno interactivo para:
    - Ingesta de datos
    - EDA
    - Entrenamiento y evaluación de modelos
  - URL: http://localhost:8888
  - Conectado directamente a PostgreSQL vía variables de entorno

- **feature-builder**
  - Servicio batch (no expone puertos)
  - Construye la tabla `analytics.daily_features`
  - Se ejecuta bajo demanda vía CLI (`docker compose run`)
  - Soporta modos:
    - `full`
    - `by-date-range`
  - Garantiza idempotencia y control por `run_id`

- **model-api**
  - API REST desarrollada con FastAPI
  - Carga el modelo entrenado desde `MODEL_PATH`
  - Puerto: `localhost:${API_PORT}` (por defecto `8000`)
  - Endpoints principales:
    - `/health`
    - `/model/info`
    - `/predict`
    - `/predict/batch`



---

## Ejecución del Proyecto

### FASE 1: Pipeline de Datos (Proyecto #6)

#### Paso 1: Ingesta de Datos RAW

**Notebook:** `01_ingesta_prices_raw.ipynb`

**Qué hace:**
1. Lee los tickers desde variables de ambiente
2. Descarga datos históricos OHLCV de Yahoo Finance
3. Estandariza columnas y tipos de datos
4. Carga a PostgreSQL en `raw.prices_daily`

**Output esperado:**
```
✓ Descargados 1,989 días para AAPL (2018-2025)
✓ Período: 2018-01-09 a 2025-12-05
✓ Total insertado: 1,989 registros
```

#### Paso 2: Construcción de Features

**Script CLI:** `build_features.py`

Este script transforma los datos crudos (`raw.prices_daily`) en features listas para machine learning y las guarda en `analytics.daily_features`.

# Ejecutar feature builder para AAPL

### Modo FULL (reconstrucción completa)
Procesa **todo el histórico disponible** para un ticker.

```bash
docker compose run feature-builder \
  --mode full \
  --ticker AAPL \
  --run-id run_001 \
  --overwrite true
```

**Output esperado:**
```
Raw cargado: 1,989 filas
Features creadas: 1,974 filas
Desde: 2018-01-31 Hasta: 2025-12-05
Filas eliminadas por NaN: 15 (rolling features)
```

### Modo BY-DATE-RANGE (procesamiento incremental)
Permite construir features solo para un rango específico de fechas, ideal para ejecuciones incrementales o ejemplos controlados

```bash
docker compose run --rm feature-builder \
  python build_features.py \
  --mode by-date-range \
  --ticker AAPL \
  --start-date 2025-01-01 \
  --end-date 2025-03-01 \
  --run-id ejemplo_range \
  --overwrite false
```

**Output esperado:**
```
Raw cargado: 39 filas
Features creadas: 34 filas
Desde: 2025-01-10 00:00:00 Hasta: 2025-02-28 00:00:00
Filas insertadas: 34
FEATURE BUILDER FINALIZADO CORRECTAMENTE
```

---

### FASE 2: Machine Learning (Proyecto Final)

#### Paso 1: Análisis Exploratorio (EDA)

**Notebook:** `ml_trading_classifier.ipynb`

**Balance de clases:**
- Días UP: 1,069 (53.75%)
- Días DOWN: 920 (46.25%)
- **Balanceado ✅** → No requiere técnicas de balanceo

**Estadísticas clave:**
- Retorno promedio diario: +0.112%
- Volatilidad promedio: 1.65%
- Rango de precios: $35.55 - $286.19

#### Paso 2: Creación del Target

```python
df['target_up'] = np.where(df['close'] > df['open'], 1, 0)
```

**Proporción final:** 53.90% días UP (después de dropna)

#### Paso 3: Feature Engineering

**Features creadas:** 14 totales
- 12 numéricas (lags, momentum, volatilidad, RSI)
- 2 categóricas (day_of_week, month)

**Registros finales:** 1,974 (eliminados 15 por NaN)

#### Paso 4: Split Temporal (CRÍTICO)

```python
# NO se usó train_test_split aleatorio 
# Se usó split por años 

Train: años ≤ 2023 → 1,489 registros (75.4%)
Val:   año 2024    → 252 registros (12.8%)
Test:  año 2025    → 233 registros (11.8%)
```

**Rangos temporales:**
- Train: 2018-01-31 → 2023-12-29
- Val:   2024-01-02 → 2024-12-31
- Test:  2025-01-02 → 2025-12-05

**Sin solapamiento temporal ✅** → Previene leakage

#### Paso 5: Preprocesamiento

```python
# Pipeline con ColumnTransformer
numeric_pipe = Pipeline([("scaling", StandardScaler())])
categorical_pipe = Pipeline([("encoding", OneHotEncoder())])

preprocessor = ColumnTransformer([
    ("num_block", numeric_pipe, numeric_features),
    ("cat_block", categorical_pipe, categorical_features)
])

# CRÍTICO: fit() solo con train
preprocessor.fit(X_train)
```

**Features finales después de OHE:** 29 columnas
- 12 numéricas escaladas
- 17 categóricas codificadas

#### Paso 6: Baseline

```python
baseline = DummyClassifier(strategy="most_frequent")
```

**Resultados en Validación:**
- Accuracy: 54.37%
- F1-Score: 70.44%

---

##  Modelos de Machine Learning

### Modelos Entrenados (8 totales)

**Configuración:**
- RandomizedSearchCV con 50 iteraciones
- TimeSeriesSplit con 5 folds
- Scoring: F1-Score
- Total de entrenamientos: 2,000 (8 × 50 × 5)

**Lista de modelos:**

| Tipo | Modelo | Status |
|------|--------|--------|
| **Lineales** | LogisticRegression | ✅ |
| **Lineales** | LinearSVC | ✅ |
| **Árbol** | DecisionTree | ✅ |
| **Random Forest** | RandomForest | ✅ |
| **Boosting** | GradientBoosting | ✅ |
| **Boosting** | AdaBoost | ✅ |
| **Boosting** | XGBoost | ✅ |
| **Boosting** | CatBoost | ✅ |

### Resultados en Validación (2024)

**Ranking por F1-Score:**

| Modelo | Val F1 | Val Accuracy | Val ROC-AUC | CV Score |
|--------|--------|--------------|-------------|----------|
| **LogisticRegression** | **0.7044** | 0.5437 | 0.5122 | 0.6863 |
| LinearSVC | 0.6571 | 0.5278 | N/A | 0.5733 |
| GradientBoosting | 0.6523 | 0.5516 | 0.5314 | 0.5795 |
| DecisionTree | 0.5782 | 0.5079 | 0.5213 | 0.6554 |
| AdaBoost | 0.6372 | 0.5437 | 0.5178 | 0.5238 |
| XGBoost | 0.6316 | 0.5278 | 0.4924 | 0.5437 |
| CatBoost | 0.61111 | 0.5000 | 0.48714 | 0.5143|
| RandomForest | 0.4516 | 0.4603 | 0.4605 | 0.5160 |

**Modelo Ganador:** **LogisticRegression**
- F1: 70.44%
- Accuracy: 54.37%
- Seleccionado por mejor F1 en validación

---

## Evaluación en Test (2025)

### Resultados del Modelo Ganador (Reentrenado en Train+Val)

**Métricas de clasificación:**

| Métrica | Baseline | LogisticRegression | Diferencia |
|---------|----------|-------------------|------------|
| **Accuracy** | 51.50% | 51.50% | 0.00% |
| **F1-Score** | 67.99% | 67.81% | -0.18% |
| **Precision** | 51.50% | 51.52% | +0.02% |
| **Recall** | 100.00% | 99.17% | -0.83% |
| **ROC-AUC** | 0.500 | **0.451** | **-0.049** ❌ |

### Matriz de Confusión

```
                 Predicho DOWN  Predicho UP
Real DOWN              1            112
Real UP                1            119
```

**Interpretación CRÍTICA:**
- **El modelo predice UP en 231 de 233 casos (99.1%)**
- Solo predice DOWN 2 veces en todo el período
- **Tasa de falsos positivos: 99.12%**
- **Es prácticamente idéntico al baseline**

#### Eficiencia del Mercado
- Los precios ya incorporan la información disponible
- Usar solo precios pasados no da ventaja informacional
- Predecir dirección diaria es intrínsecamente difícil

#### Horizonte Temporal
- Predicción DIARIA tiene mucho ruido aleatorio
- Movimientos intradiarios dominados por:
  - Noticias inesperadas
  - Sentimiento del mercado
  - Órdenes institucionales

#### Limitaciones de las Features
- Solo usamos precios históricos y volumen
- **NO usamos:**
  - Sentimiento de noticias/redes sociales
  - Datos fundamentales (earnings, P/E ratio)
  - Datos macroeconómicos
  - Flujo de órdenes
  - Volatilidad implícita (VIX)

**Conclusión:** Nuestro resultado está **dentro del rango esperado** para predicción diaria de dirección con solo features técnicas.

---

##  Simulación de Inversión (USD 10,000 en 2025)

### Configuración de la Simulación

**Capital inicial:** USD 10,000  
**Período:** 2025-01-02 a 2025-12-05 (233 días bursátiles)  

**Estrategia:** 
- Si predice UP → Comprar en OPEN, vender en CLOSE
- Si predice DOWN → Permanecer en efectivo

**Supuestos:**
- Sin costos de transacción
- Sin slippage
- No se permite apalancamiento

### Resultados de la Simulación

**Estrategia ML:**
- Capital final: **USD 13,293.24**
- Retorno total: **+32.93%**
- Retorno anualizado: **+39.12%**
- Trades ejecutados: **231 de 233 días (99.14%)**
- Win rate: **51.52%**
- Sharpe Ratio: **1.15**
- Max Drawdown: **-11.18%**
- Volatilidad anual: **28.55%**

### LA PARADOJA: ¿Cómo es rentable con métricas tan malas?

**El resultado parece contradictorio:**
- Accuracy: 51.50% (casi aleatorio) 
- ROC-AUC: 0.451 (peor que aleatorio) 
- **PERO: Retorno +32.93%** ✅

#### Explicación:

**1. CONTEXTO DE MERCADO ALCISTA**
- **2025 fue un año alcista para AAPL**
- El precio subió significativamente
- Buy & Hold también tuvo retorno positivo
- **El modelo captura la tendencia, no predice la dirección**

**2. EL MODELO ES CASI BUY & HOLD**
- Predice UP el 99.1% del tiempo
- Solo evita 2 días de 233
- **Es prácticamente idéntico a mantener la acción**

**3. FUNCIONÓ POR EL MERCADO, NO POR EL MODELO**
- En un año alcista, predecir siempre UP funciona
- El modelo NO tiene capacidad predictiva real
- En un mercado bajista, perdería igual que Buy & Hold

## Guardado del Modelo Entrenado

Una vez seleccionado el **modelo ganador** (Logistic Regression) y finalizado el proceso de entrenamiento y validación, el modelo se **serializa junto con todo el preprocesamiento necesario** para garantizar predicciones reproducibles en producción.

### ¿Qué se guarda?

El artefacto del modelo incluye:

- **Modelo entrenado** (`best_model`)
- **Pipeline de preprocesamiento** (`preprocessor`)
- **Listado de features numéricas**
- **Listado de features categóricas**
- **Orden final de las columnas de entrada**

Esto asegura que la API pueda recibir datos crudos y aplicar exactamente las mismas transformaciones utilizadas durante el entrenamiento (sin riesgo de inconsistencias).

### Código de guardado del artefacto

```python
artifact = {
    "model": best_model,
    "preprocessor": preprocessor,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_cols": feature_cols
}

joblib.dump(artifact, "best_model_artifact.pkl")

print("best_model_artifact.pkl guardado correctamente con los features finales")

```
---

##  Conclusiones Críticas del Proyecto

###  El Modelo NO es Útil en Producción

**Razones:**

1. **No supera al baseline**
   - Accuracy = baseline (51.50%)
   - ROC-AUC peor que aleatorio (0.451)
   
2. **No agrega valor sobre Buy & Hold**
   - Predice UP 99.1% del tiempo
   - No identifica cuándo NO operar
   
3. **No generaliza**
   - Funcionó solo porque 2025 fue alcista
   - En mercado bajista, fallaría igual

4. **Sin capacidad predictiva real**
   - El retorno se debe al mercado, no al modelo
   - Es un "classifier" que siempre dice la misma clase


**Lecciones aprendidas:**

1. **Predecir mercados es DIFÍCIL**
   - Eficiencia del mercado es real
   - Features técnicas básicas no son suficientes
   - Horizonte diario es extremadamente ruidoso

2. **Métricas ML ≠ Rentabilidad**
   - Un modelo con F1=67% puede ser rentable
   - PERO no significa que tenga valor predictivo
   - Contexto de mercado es crítico

3. **Importancia del Split Temporal**
   - Split aleatorio habría dado resultados engañosos
   - Split temporal revela la verdad: el modelo no funciona

4. **Valor del Análisis Crítico**
   - Entender QUÉ NO funciona es tan valioso como qué sí
   - Base para futuras mejoras
   - Conocimiento real sobre limitaciones de ML en finanzas

### ¿Qué haría falta para mejorar?

**Features más sofisticadas:**
- Sentimiento de noticias (NLP)
- Datos fundamentales (earnings, P/E, EPS)
- Indicadores macroeconómicos
- Volatilidad implícita (opciones)
- Flujo de órdenes institucionales

**Horizonte temporal diferente:**
- Predicción semanal (menos ruido)
- Swing trading (3-5 días)
- Detección de regímenes de mercado

**Problema diferente:**
- Predicción de magnitud (regresión) vs dirección
- Clasificación multi-clase (fuerte subida/leve/flat/bajada)
- Detección de anomalías

**Modelos más complejos:**
- Deep Learning (LSTM, Transformers)
- Reinforcement Learning
- Ensemble con stacking

---

## API REST del Modelo

### Despliegue

```bash
# Levantar la API
docker-compose up -d model-api

# O localmente
cd stock_api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints Disponibles

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "LogisticRegression"
}
```

#### 2. Información del Modelo

```bash
curl http://localhost:8000/model/info
```

#### 3. Predicción Individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "open_prev_day": 150.0,
    "high_prev_day": 152.0,
    "low_prev_day": 149.0,
    "close_prev_day": 151.0,
    "volume_prev_day": 50000000,
    "ret_prev_day": 0.0067,
    "volatility_prev_5": 0.02,
    "volume_avg_7": 48000000,
    "price_avg_7": 150.5,
    "daily_range_prev": 3.0,
    "momentum_3": 0.015,
    "rsi_proxy": 55.0,
    "day_of_week": 2,
    "month": 12
  }'
```
### Resultado de la API

```json
{
  "predicted_class": 1,
  "predicted_direction": "UP",
  "prob_up": 0.869527180533921,
  "prob_down": 0.13047281946607903,
  "confidence": 0.869527180533921
}
```


## Estructura del Proyecto

```
proyecto_trading_final/
│
├── docker-compose.yml
├── .env.example
├── README.md
│
├── notebooks/
│   ├── 01_ingesta_prices_raw.ipynb
│   └── ml_trading_classifier.ipynb
│
├── feature_builder/
│   ├── Dockerfile
│   ├── build_features.py
│   └── requirements.txt
│
├── stock_api/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── best_model_artifact.pkl
│
└── evidences/
    ├── ingesta/
    ├── feature_builder/
    ├── machine_learning/
    └── stock_api/
```
---

## Checklist de Cumplimiento (Rúbrica)

### Proyecto #6 

- [x] **Pipeline RAW**
  - [x] raw.prices_daily completo (1,989 registros)
  - [x] Metadatos completos
  - [x] Mínimo 3 años (8 años)

- [x] **Tabla analytics.daily_features**
  - [x] Estructura correcta (14 features)
  - [x] Features derivadas coherentes
  - [x] Sin data leakage
  - [x] Metadatos completos

- [x] **feature-builder CLI**
  - [x] Modos full y by-date-range
  - [x] Idempotencia
  - [x] Logs claros

- [x] **Reproducibilidad**
  - [x] .env.example
  - [x] README completo
  - [x] Comandos claros

### Proyecto Final 

- [x] **Pipeline ML**
  - [x] EDA completo
  - [x] Features justificadas, sin leakage
  - [x] Split temporal correcto

- [x] **Modelos**
  - [x] 8 modelos (≥7 requeridos)
  - [x] Tuning con RandomizedSearch
  - [x] Comparación justa

- [x] **Evaluación**
  - [x] Métricas en train/val/test
  - [x] Baseline implementado
  - [x] Reentrenamiento en train+val 
  - [x] Modelo ganador justificado
  - [x] **Simulación USD 10,000**
  - [x] **Análisis crítico de resultados**

- [x] **API REST**
  - [x] FastAPI funcional
  - [x] Endpoints /predict
  - [x] Manejo de errores
  - [x] Documentación Swagger

- [x] **Reproducibilidad**
  - [x] .env.example actualizado
  - [x] README con ejemplos
  - [x] Seeds (random_state=42)


