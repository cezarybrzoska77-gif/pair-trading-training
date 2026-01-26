# -------------------------------
# Pair Trading: ADBE / CRM
# -------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- KROK 1: POBRANIE DANYCH ---
ticker_x = "ADBE"
ticker_y = "CRM"
start_date = "2022-01-01"

# Pobranie danych z Yahoo Finance
data_raw = yf.download([ticker_x, ticker_y], start=start_date, group_by="ticker")

# Tworzymy czysty DataFrame tylko z Adjusted Close
data = pd.DataFrame({ticker: data_raw[ticker]["Adj Close"] for ticker in [ticker_x, ticker_y]})

print("Pierwsze 5 obserwacji:")
print(data.head())

# --- KROK 2: CZYSZCZENIE DANYCH ---
data = data.dropna()
print("\nLiczba obserwacji:", len(data))

# --- KROK 3: HEDGE RATIO ---
X = sm.add_constant(data[ticker_y])
model = sm.OLS(data[ticker_x], X).fit()
hedge_ratio = model.params[1]
print("\nHedge ratio (ADBE vs CRM):", round(hedge_ratio, 4))

# --- KROK 4: SPREAD ---
spread = data[ticker_x] - hedge_ratio * data[ticker_y]
print("\nSpread – pierwsze wartości:")
print(spread.head())

# --- KROK 5: Z-SCORE ---
spread_mean = spread.mean()
spread_std = spread.std()
z_score = (spread - spread_mean) / spread_std
print("\nAktualny Z-score:", round(z_score.iloc[-1], 2))

# --- KROK 6: HALF-LIFE ---
spread_lag = spread.shift(1)
spread_ret = spread - spread_lag
spread_lag = sm.add_constant(spread_lag.dropna())
spread_ret = spread_ret.dropna()
hl_model = sm.OLS(spread_ret, spread_lag).fit()
half_life = -np.log(2) / hl_model.params[1]
print("\nHalf-life (dni):", round(half_life, 1))

