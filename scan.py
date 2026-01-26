import yfinance as yf
import pandas as pd
import numpy as np
import itertools

tickers = ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ADBE", "CRM"]
start = "2022-01-01"
window = 60

prices = yf.download(tickers, start=start, auto_adjust=True)["Close"]
returns = np.log(prices / prices.shift(1)).dropna()

pairs = []

for a, b in itertools.combinations(returns.columns, 2):
    corr = returns[a].rolling(window).corr(returns[b])
    pairs.append({
        "pair": f"{a}/{b}",
        "mean_corr": corr.mean(),
        "std_corr": corr.std()
    })

df = pd.DataFrame(pairs).sort_values("mean_corr", ascending=False)

print("\nTOP SKORELOWANE PARY:\n")
print(df.head(5))
