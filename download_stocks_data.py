import requests
import pandas as pd

api_key = "wS9wpderveZUndNsetBdYqp9Mp5oDqgH"

# Add your stock tickers here
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "JPM", "DIS"]

# Date range
start_date = "2024-05-01"
end_date = "2025-05-08"

# List to hold individual stock DataFrames
dfs = []

# Loop through each ticker and fetch its historical data
for symbol in tickers:
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    
    response = requests.get(url)
    data = response.json()

    if "historical" in data:
        df = pd.DataFrame(data["historical"])
        df["Ticker"] = symbol
        dfs.append(df)
    else:
        print(f"Failed to fetch data for {symbol}")

# Combine all into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: sort by date and ticker
combined_df = combined_df.sort_values(by=["date", "Ticker"], ascending=[False, True])

# Save to CSV
combined_df.to_csv("data/multi_stocks_dataset.csv", index=False)
print("✅ Dataset saved as 'multi_stocks_dataset.csv'")
