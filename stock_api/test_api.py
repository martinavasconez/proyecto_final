#!/usr/bin/env python3
"""
Script para probar la API de predicción de acciones
"""
import requests
import json
from typing import Dict, Any

# URL base de la API
BASE_URL = "http://localhost:8000"


def test_health():
    """Test del endpoint de health"""
    print("=" * 50)
    print("TEST: Health Check")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test del endpoint de información del modelo"""
    print("=" * 50)
    print("TEST: Model Info")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_example_features():
    """Test del endpoint de ejemplo de features"""
    print("=" * 50)
    print("TEST: Example Features")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/features/example")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.json()

def test_single_prediction(features: Dict[str, Any]):
    """Test de predicción individual"""
    print("=" * 50)
    print("TEST: Single Prediction")
    print("=" * 50)
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=features
    )
    
    print(f"Status Code: {response.status_code}")
    
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except:
        print(f"Raw Response: {response.text}")
        return None

def test_batch_prediction():
    """Test de predicción en batch"""
    print("=" * 50)
    print("TEST: Batch Prediction")
    print("=" * 50)
    
    features_list = [
        {
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
    ]
    
    payload = {"features_list": features_list}
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Raw Response: {response.text}")

    print()



def test_custom_prediction():
    """Test con datos personalizados"""
    print("=" * 50)
    print("TEST: Custom Prediction - Bullish Signal")
    print("=" * 50)
    
    # Ejemplo de señal alcista (bullish)
    bullish_features = {
        "open_prev_day": 145.0,
        "high_prev_day": 148.0,
        "low_prev_day": 144.5,
        "close_prev_day": 147.5,
        "volume_prev_day": 60000000,  # Alto volumen
        "ret_prev_day": 0.017,  # Retorno positivo
        "volatility_prev_5": 0.015,  # Baja volatilidad
        "volume_avg_7": 50000000,
        "price_avg_7": 145.0,
        "daily_range_prev": 3.5,
        "momentum_3": 0.025,  # Momentum positivo
        "rsi_proxy": 65.0,  # RSI elevado pero no sobrecomprado
        "day_of_week": 1,  # Martes
        "month": 12
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=bullish_features
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    print("=" * 50)
    print("TEST: Custom Prediction - Bearish Signal")
    print("=" * 50)
    
    # Ejemplo de señal bajista (bearish)
    bearish_features = {
        "open_prev_day": 150.0,
        "high_prev_day": 150.5,
        "low_prev_day": 147.0,
        "close_prev_day": 147.5,
        "volume_prev_day": 65000000,  # Alto volumen
        "ret_prev_day": -0.017,  # Retorno negativo
        "volatility_prev_5": 0.03,  # Alta volatilidad
        "volume_avg_7": 50000000,
        "price_avg_7": 150.0,
        "daily_range_prev": 3.5,
        "momentum_3": -0.02,  # Momentum negativo
        "rsi_proxy": 35.0,  # RSI bajo
        "day_of_week": 4,  # Viernes
        "month": 12
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=bearish_features
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def run_all_tests():
    """Ejecutar todos los tests"""
    try:
        # Test básicos
        test_health()
        test_model_info()
        
        # Test de features
        example_features = test_example_features()
        
        # Test de predicciones
        test_single_prediction(example_features)
        test_batch_prediction()
        test_custom_prediction()
        
        print("=" * 50)
        print("TODOS LOS TESTS COMPLETADOS")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: No se puede conectar a la API")
        print("Asegúrate de que la API esté corriendo en http://localhost:8000")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()