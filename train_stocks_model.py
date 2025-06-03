import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# By just adding name of stock as feature:
# Training loss improved from 0.052 → 0.00178 (96.6% reduction)
# Validation loss improved from 0.025 → 0.00139 (94.4% reduction)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load and prepare data
data_path = "data/multi_stocks_dataset.csv"
stock_data = pd.read_csv(data_path)

# Validate required columns
required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Ticker']
if not all(col in stock_data.columns for col in required_columns):
    raise ValueError("CSV file must contain columns: date, open, high, low, close, volume, Ticker")

# Feature engineering
stock_data['DayOfWeek'] = pd.to_datetime(stock_data['date']).dt.dayofweek

# One-hot encode tickers
ticker_dummies = pd.get_dummies(stock_data['Ticker'], prefix='ticker')

# Separate numerical and categorical features
numerical_features = stock_data[['open', 'high', 'low', 'close', 'volume', 'DayOfWeek']]
categorical_features = ticker_dummies

# Normalize only numerical features
scaler = MinMaxScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Combine features ensuring all are float32
X_numerical = numerical_features_scaled.astype(np.float32)
X_categorical = categorical_features.values.astype(np.float32)
X_combined = np.concatenate([X_numerical, X_categorical], axis=1)

# Create sequences
def create_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 3])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

time_steps = 3
X, y = create_sequences(X_combined, time_steps)

# Convert to PyTorch tensors
X = torch.from_numpy(X)  # dtype is preserved from numpy
y = torch.from_numpy(y).view(-1, 1)

# Dataset class
class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create and split dataset
dataset = StockDataset(X, y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# LSTM Model definition
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
        out = out[:, -1, :]  # Take the last time step
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Initialize model with correct input size
input_size = X_combined.shape[1]  # Total number of features
hidden_size = 50
num_layers = 1
model = LSTMModel(input_size, hidden_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    # Store losses
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.6f}, '
          f'Val Loss: {avg_val_loss:.6f}')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save model and scaler
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_min': scaler.min_,
    'scaler_scale': scaler.scale_,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'ticker_columns': list(ticker_dummies.columns),
    'numerical_columns': list(numerical_features.columns)
}, 'lstm_stock_model.pth')

print("Model and scaler saved successfully as 'lstm_stock_model.pth'")