import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.utils import clip_grad_norm_

# Train Loss improved from 0.162978 → 0.000588 (99.6% reduction)
# Val Loss improved from 0.005221 → 0.000299 (94.3% reduction)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_crypto_data(data_path):
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV must contain: date, symbol, open, high, low, close, volume")
    
    # Convert date and extract temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    
    # Crypto-specific features
    df['price_range'] = df['high'] - df['low']
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['log_volume'] = np.log1p(df['volume'])  # Better for volume scaling
    
    return df

def prepare_features(df):
    # One-hot encode cryptocurrency symbols
    symbol_dummies = pd.get_dummies(df['symbol'], prefix='crypto')
    
    # Separate features for different scaling
    price_features = df[['open', 'high', 'low', 'close', 'typical_price', 'price_range']]
    close_price = df[['close']]  # Separate target
    volume_features = df[['log_volume']]
    time_features = df[['day_of_week', 'day_of_month', 'month']]
    
    # Scale features appropriately
    price_scaler = MinMaxScaler(feature_range=(-1, 1))  # For input features
    close_scaler = MinMaxScaler(feature_range=(-1, 1))  # Separate scaler for target
    volume_scaler = StandardScaler()
    time_scaler = MinMaxScaler()
    
    price_scaled = price_scaler.fit_transform(price_features)
    y = close_scaler.fit_transform(close_price).flatten().astype(np.float32)
    volume_scaled = volume_scaler.fit_transform(volume_features)
    time_scaled = time_scaler.fit_transform(time_features)
    
    # Combine all features
    X = np.concatenate([
        price_scaled,
        volume_scaled,
        time_scaled,
        symbol_dummies.values
    ], axis=1).astype(np.float32)
    
    return X, y, {'price': price_scaler, 'close': close_scaler, 'volume': volume_scaler}, list(symbol_dummies.columns)

def create_sequences(X, y, symbol_columns, time_steps=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        if symbol_columns:  # Only check if we have symbol columns
            current_symbol = X[i, -len(symbol_columns):].argmax()
            next_symbol = X[i+time_steps, -len(symbol_columns):].argmax()
            if current_symbol != next_symbol:
                continue
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

class CryptoDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context)

def get_scaler_params(scaler):
    """Helper function to get parameters for different scaler types"""
    if isinstance(scaler, MinMaxScaler):
        return {'min': scaler.min_, 'scale': scaler.scale_}
    elif isinstance(scaler, StandardScaler):
        return {'mean': scaler.mean_, 'scale': scaler.scale_}
    else:
        return {}

def train_crypto_model(data_path, epochs=30, batch_size=64):
    # Load and prepare data
    df = load_crypto_data(data_path)
    X, y, scalers, symbol_columns = prepare_features(df)
    X_seq, y_seq = create_sequences(X, y, symbol_columns)
    
    # Create datasets
    dataset = CryptoDataset(X_seq, y_seq)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize model
    input_size = X.shape[1]
    model = CryptoLSTM(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    early_stop_patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                epoch_val_loss += criterion(outputs, y_val).item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'scalers': {k: get_scaler_params(v) for k, v in scalers.items()},
                'symbol_columns': symbol_columns,
                'input_size': input_size
            }, 'crypto_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

if __name__ == "__main__":
    model = train_crypto_model("data/multi_crypto_data.csv", epochs=50)
    print("Training complete. Best model saved as 'crypto_model.pth'")