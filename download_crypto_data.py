import requests
import pandas as pd
import time

api_key = "wS9wpderveZUndNsetBdYqp9Mp5oDqgH"

# List of crypto symbols (FMP uses SYMBOLUSD format)
cryptos = [
    "BTCUSD", "ETHUSD", "BNBUSD", "XRPUSD", "ADAUSD",
    "SOLUSD", "DOGEUSD", "DOTUSD", "LTCUSD", "LINKUSD"
]

def fetch_crypto(symbol):
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?apikey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if not data:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        return df.sort_values(by="date")
    else:
        print(f"Failed to fetch {symbol}: {response.status_code} - {response.text}")
        return pd.DataFrame()

# Combine data from all coins
all_data = pd.DataFrame()

for symbol in cryptos:
    print(f"Fetching data for {symbol}...")
    df = fetch_crypto(symbol)
    if not df.empty:
        all_data = pd.concat([all_data, df], ignore_index=True)
    time.sleep(2)  # Respect rate limits

# Save to CSV
output_file = "data/multi_crypto_data.csv"
all_data.to_csv(output_file, index=False)
print(f"\n✅ Data saved to '{output_file}' with shape: {all_data.shape}")
