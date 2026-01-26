# -------------------------------
# Pair Trading: ADBE / CRM
# -------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time

# --- KROK 1: POBRANIE DANYCH Z RETRY ---
ticker_x = "ADBE"
ticker_y = "CRM"
start_date = "2022-01-01"

max_retries = 5
for attempt in range(max_retries):
    try:
        # Pobranie tylko Adjusted Close
        data = yf.download([ticker_x, ticker_y], start=start_date)["Adj Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        # Sprawdzenie czy są wartości
        if data.dropna().empty:
            raise ValueError("Pobrany DataFrame jest pusty")
        break  # dane pobrane poprawnie
    except Exception as e:
        print(f"Błąd pobierania danych (próba {attempt+1}/{max_retries}): {e}")
        time.sleep(3)
else:
    raise RuntimeError("Nie udało się pobrać danych po kilku próbach")

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
