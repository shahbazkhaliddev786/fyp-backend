import os
import torch
import requests
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models.train_crypto_model import CryptoLSTM

router = APIRouter()

FMP_API_KEY = os.getenv("FMP_API_KEY")
model_path = 'crypto_model.pth'

def fetch_last_3_days(symbol: str):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=3&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from FMP API")
    data = response.json()
    print(f"Fetched data for {symbol}: {data['historical']}")
    return pd.DataFrame(data["historical"])

def preprocess_input(df, symbol, scalers, symbol_columns):
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['price_range'] = df['high'] - df['low']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['log_volume'] = np.log1p(df['volume'])

        print("Preprocessing input: columns after transformation", df.columns)

        symbol_dummies = pd.get_dummies(df['symbol'], prefix='crypto')
        for col in symbol_columns:
            if col not in symbol_dummies:
                symbol_dummies[col] = 0
        symbol_dummies = symbol_dummies[symbol_columns]

        price = scalers['price'].transform(df[['open', 'high', 'low', 'close', 'typical_price', 'price_range']])
        volume = scalers['volume'].transform(df[['log_volume']])
        
        if 'time' in scalers:
            time = scalers['time'].transform(df[['day_of_week', 'day_of_month', 'month']])
        else:
            # fallback if time scaler not found (use zeros)
            time = np.zeros((df.shape[0], 3))

        X = np.concatenate([price, volume, time, symbol_dummies.values], axis=1).astype(np.float32)
        print(f"Preprocessed input shape: {X.shape}")
        return X[np.newaxis, ...]

    except Exception as e:
        print(f"Error in preprocess_input: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during preprocessing")

def load_model_and_scalers(model_path):
    try:
        print(f"Loading model from {model_path}")

        torch.serialization.add_safe_globals([
            np.core.multiarray._reconstruct,
            np.dtype,
            np.ndarray
        ])

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        model = CryptoLSTM(checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        price_scaler = MinMaxScaler(feature_range=(-1, 1))
        price_scaler.min_ = checkpoint['scalers']['price']['min']
        price_scaler.scale_ = checkpoint['scalers']['price']['scale']

        close_scaler = MinMaxScaler(feature_range=(-1, 1))
        close_scaler.min_ = checkpoint['scalers']['close']['min']
        close_scaler.scale_ = checkpoint['scalers']['close']['scale']

        volume_scaler = StandardScaler()
        volume_scaler.mean_ = checkpoint['scalers']['volume']['mean']
        volume_scaler.scale_ = checkpoint['scalers']['volume']['scale']

        # Time scaler might be missing
        time_scaler = None
        if 'time' in checkpoint['scalers']:
            time_scaler = MinMaxScaler()
            time_scaler.min_ = checkpoint['scalers']['time']['min']
            time_scaler.scale_ = checkpoint['scalers']['time']['scale']

        scalers = {
            'price': price_scaler,
            'close': close_scaler,
            'volume': volume_scaler,
        }
        if time_scaler:
            scalers['time'] = time_scaler

        print("Model and scalers loaded successfully")
        return model, scalers, checkpoint['symbol_columns']

    except Exception as e:
        print(f"Error loading model or scalers: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model or scalers")

@router.get("/predict-crypto")
def predict_crypto(symbol: str = Query(..., description="Symbol for prediction, e.g., BTCUSD")):
    try:
        df = fetch_last_3_days(symbol)
        model, scalers, symbol_columns = load_model_and_scalers(model_path)

        # suggestion for investment
        df = df[::-1].reset_index(drop=True)
        last_actual_close_value =  df.iloc[-1]['close']

        X = preprocess_input(df, symbol, scalers, symbol_columns)
        with torch.no_grad():
            y_pred = model(torch.tensor(X)).numpy().flatten()[0]
        predicted_close = scalers['close'].inverse_transform([[y_pred]])[0][0]
        recommendation = ''

        if predicted_close > last_actual_close_value:
            recommendation = "You should invest in this coin"
        else:
            recommendation = "You should not invest in this coin"
        return {
            "symbol": symbol,
            "predicted_close": round(predicted_close, 4),
            "last_actual_close": round(last_actual_close_value, 4),
            "recommendation": recommendation
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


