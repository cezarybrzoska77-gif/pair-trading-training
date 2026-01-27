import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations

# =============================
# PARAMETRY
# =============================
ETF = "XLK"
START_DATE = "2022-01-01"
CORR_THRESHOLD = 0.80
COINT_PVALUE = 0.05

# =============================
# SKŁAD XLK (ręcznie – stabilnie)
# =============================
tickers = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
    "CRM", "ADBE", "AMD", "INTC", "QCOM"
]

print("Pobieranie danych...")
prices = yf.download(tickers, start=START_DATE)["Adj Close"]
prices = prices.dropna()

print(f"Liczba obserwacji: {len(prices)}")

# =============================
# KORELACJA
# =============================
corr_matrix = prices.corr()

results = []

print("Skanowanie par...")
for x, y in combinations(tickers, 2):
    corr = corr_matrix.loc[x, y]

    if corr < CORR_THRESHOLD:
        continue

    score, pvalue, _ = coint(prices[x], prices[y])

    if pvalue < COINT_PVALUE:
        results.append({
            "pair": f"{x}/{y}",
            "correlation": round(corr, 3),
            "pvalue": round(pvalue, 4)
        })

df = pd.DataFrame(results)

if df.empty:
    print("❌ Brak par spełniających kryteria")
else:
    df = df.sort_values("correlation", ascending=False)
    print("\n✅ TOP PARY XLK:")
    print(df.head(10))

    df.to_csv("XLK_scan_results.csv", index=False)
    print("\n📁 Zapisano: XLK_scan_results.csv")
