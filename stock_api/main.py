import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "best_model_artifact.pkl")
ARTIFACT_PATH = Path(MODEL_PATH)

try:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"No se encontró {ARTIFACT_PATH}")

    artifact = joblib.load(ARTIFACT_PATH)

    model = artifact["model"]  
    numeric_features = artifact.get("numeric_features", [])
    categorical_features = artifact.get("categorical_features", [])
    feature_cols = artifact["feature_cols"]

    logger.info(f"Modelo cargado: {type(model).__name__}")
    logger.info(f"Total features: {len(feature_cols)}")

except Exception as e:
    logger.error(f"Error cargando el modelo: {e}")
    raise

class DailyFeatures(BaseModel):

    open_prev_day: float
    high_prev_day: float
    low_prev_day: float
    close_prev_day: float
    volume_prev_day: float

    ret_prev_day: float
    volatility_prev_5: float
    volume_avg_7: float
    price_avg_7: float
    daily_range_prev: float
    momentum_3: float
    rsi_proxy: float

    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)

    @field_validator("open_prev_day", "high_prev_day", "low_prev_day", "close_prev_day")
    @classmethod
    def validate_prices(cls, v):
        if v <= 0:
            raise ValueError("Los precios deben ser mayores a 0")
        return v

    @field_validator("volume_prev_day", "volume_avg_7")
    @classmethod
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError("El volumen no puede ser negativo")
        return v

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class BatchPredictionRequest(BaseModel):
    features_list: List[DailyFeatures]


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_direction: str
    prob_up: float
    prob_down: float
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_predictions: int


class ModelInfo(BaseModel):
    model_type: str
    num_features: int
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]

app = FastAPI(
    title="Stock Direction Prediction API",
    description="API para predecir la dirección del precio de acciones",
    version="2.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__
    }


@app.get("/model/info", response_model=ModelInfo)
def get_model_info():
    return ModelInfo(
        model_type=type(model).__name__,
        num_features=len(feature_cols),
        feature_names=feature_cols,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(features: DailyFeatures):
    try:
        data_dict = features.model_dump()
        df_input = pd.DataFrame([data_dict], columns=feature_cols)

        y_pred = model.predict(df_input)[0]

        try:
            proba = model.predict_proba(df_input)[0]
            prob_down = float(proba[0])
            prob_up = float(proba[1])
        except AttributeError:
            prob_up = float(y_pred)
            prob_down = 1.0 - prob_up

        confidence = max(prob_up, prob_down)
        direction = "UP" if y_pred == 1 else "DOWN"

        return PredictionResponse(
            predicted_class=int(y_pred),
            predicted_direction=direction,
            prob_up=prob_up,
            prob_down=prob_down,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f" Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):

    try:
        predictions = [predict_single(f) for f in request.features_list]

        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions)
        )

    except Exception as e:
        logger.error(f"Error batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/example")
def get_example_features():
    return DailyFeatures.Config.json_schema_extra["example"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)