from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add this to allow NumPy deserialization
torch.serialization.add_safe_globals([torch])

# Load model & preprocessing
checkpoint = torch.load('lstm_stock_model.pth', map_location=torch.device('cpu'), weights_only=False)
scaler_min = checkpoint['scaler_min']
scaler_scale = checkpoint['scaler_scale']
numerical_columns = checkpoint['numerical_columns']  # expected column names
ticker_columns = checkpoint['ticker_columns']
input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']

logger.info(f"Model loaded with numerical columns: {numerical_columns}")
logger.info(f"Model expects ticker columns: {ticker_columns[:5]}...")  # Print first 5 tickers

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = LSTMModel(input_size, hidden_size, num_layers)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

router = APIRouter()

class StockRequest(BaseModel):
    ticker: str

FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    logger.warning("FMP_API_KEY not set, using 'demo' as default")
    FMP_API_KEY = "demo"

def fetch_last_3_days_from_fmp(ticker: str):
    """Fetch stock data from Financial Modeling Prep API"""
    logger.info(f"Fetching data for ticker: {ticker}")
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker.upper()}?timeseries=5&apikey={FMP_API_KEY}"
    response = requests.get(url)
    logger.info(f"API response status code: {response.status_code}")
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"FMP API request failed with status {response.status_code}")

    data = response.json()
    logger.debug(f"API response keys: {list(data.keys())}")
    
    historical = data.get("historical", [])
    if len(historical) < 3:
        raise HTTPException(status_code=400, detail=f"Not enough historical data for {ticker}, need at least 3 days")
    
    logger.debug(f"First historical entry: {historical[0]}")
    
    df = pd.DataFrame(historical[:3][::-1])  # take latest 3 in order (oldest to newest)
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    
    return df

@router.post("/predict-stock")
def predict(req: StockRequest):
    ticker = req.ticker.upper()

    try:
        # Fetch data from API
        df = fetch_last_3_days_from_fmp(ticker)
        logger.info(f"Retrieved data for {ticker}, shape: {df.shape}")
        logger.debug(f"Column names: {df.columns.tolist()}")

        # CRITICAL FIX: Map API column names to model's expected lowercase column names
        # Model expects: open, high, low, close, volume, DayOfWeek
        
        # Create a map of API column names to model column names
        column_mapping = {}
        
        # The model expects lowercase column names:
        # 'open', 'high', 'low', 'close', 'volume', 'DayOfWeek'
        
        # Map columns case-insensitively
        lowercase_columns = {col.lower(): col for col in df.columns}
        logger.debug(f"Lowercase column mapping: {lowercase_columns}")
        
        if 'open' in lowercase_columns:
            column_mapping[lowercase_columns['open']] = 'open'
        
        if 'high' in lowercase_columns:
            column_mapping[lowercase_columns['high']] = 'high'
            
        if 'low' in lowercase_columns:
            column_mapping[lowercase_columns['low']] = 'low'
            
        if 'close' in lowercase_columns:
            column_mapping[lowercase_columns['close']] = 'close'
            
        if 'volume' in lowercase_columns:
            column_mapping[lowercase_columns['volume']] = 'volume'
        
        logger.info(f"Column mapping: {column_mapping}")
        
        # Rename columns to match model's expected names
        df = df.rename(columns=column_mapping)
        
        # Add day of week column (expected by model)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['DayOfWeek'] = df['date'].dt.dayofweek
        
        # Verify all required columns are present
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'DayOfWeek']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise HTTPException(status_code=400, detail=f"Missing columns in data: {', '.join(missing_columns)}")
        
        # Extract features in the correct order
        features = df[expected_columns].copy()
        logger.debug(f"Features shape: {features.shape}")
        
        # Scale numerical features
        features[numerical_columns] = (features[numerical_columns] - scaler_min) / scaler_scale
        
        # One-hot encode ticker
        ticker_vector = np.zeros(len(ticker_columns), dtype=np.float32)
        ticker_key = f"ticker_{ticker}"
        
        if ticker_key not in ticker_columns:
            logger.error(f"Ticker '{ticker}' not found in training set")
            raise HTTPException(status_code=400, detail=f"Ticker '{ticker}' was not in training set")
        
        ticker_idx = ticker_columns.index(ticker_key)
        ticker_vector[ticker_idx] = 1.0
        
        # Repeat ticker vector for each day in the sequence
        ticker_repeated = np.tile(ticker_vector, (3, 1))
        
        # Combine feature data with ticker encoding
        feature_values = features.values.astype(np.float32)
        full_input = np.hstack((feature_values, ticker_repeated))
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)  # shape: (1, 3, input_size)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        logger.info(f"Prediction for {ticker}: {prediction}")
        
        # Get the last actual close value (most recent day)
        last_actual_close = df.iloc[-1]['close']

        # Generate recommendation
        recommendation = ''
        if prediction > last_actual_close:
            recommendation = "You should invest in this stock"
        else:
             recommendation = "You should not invest in this stock"

        return {
            "symbol": ticker,
            "predicted_close": round(prediction, 2),
            "last_actual_close": round(float(last_actual_close), 2),
            "recommendation": recommendation
        }
        
    except Exception as e:
        logger.exception(f"Error predicting stock price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    