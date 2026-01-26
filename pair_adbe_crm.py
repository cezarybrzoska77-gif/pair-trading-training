
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_adbe_crm.py
Skrypt do par tradingu ADBE–CRM:
- Pobiera dane (Adj Close),
- Liczy hedge ratio (OLS z wyrazem wolnym),
- Buduje spread,
- Liczy Z-score,
- Estymuje half-life procesu OU (Δs_t = α + β s_{t-1} + ε) z poprawnym indeksowaniem parametru 'lag'.

Wymagane biblioteki: numpy, pandas, statsmodels, yfinance (instalacja on-the-fly jeśli brak).
"""

import sys
import math
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# --- Importy i ewentualna instalacja brakujących paczek ---
def _safe_imports():
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
    except ImportError as e:
        print(f"Brakująca biblioteka: {e}. Zainstaluj: numpy, pandas, statsmodels")
        raise

    # yfinance może nie być zainstalowane w niektórych środowiskach (np. GitHub Actions slim)
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
        import yfinance as yf

    return np, pd, sm, yf

np, pd, sm, yf = _safe_imports()


# --- Funkcje pomocnicze ---
def fetch_prices(tickers, start=None, end=None):
    """
    Pobiera Adj Close dla listy tickerów do ramki z kolumnami = tickery i wspólnym indeksem dat.
    """
    if start is None:
        # domyślnie: ostatnie 4 lata (ok. 1000 sesji)
        start = (datetime.utcnow() - timedelta(days=365*4+30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,   # wycisza pasek postępu yfinance
        auto_adjust=False # bierzemy Adj Close jawnie
    )

    if "Adj Close" in data.columns:
        data = data["Adj Close"].copy()
    else:
        # Jeżeli yfinance zwróci płaską ramkę (rzadkie przypadki)
        data = data.copy()

    # Gdy jest jeden ticker, yfinance zwraca Series—zamieńmy na DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Upewnij się, że kolumny to dokładnie nazwy tickerów
    data = data.loc[:, tickers]
    data = data.dropna(how="any")

    return data


def compute_hedge_ratio(y, x):
    """
    OLS: y ~ const + x
    Zwraca (model, hedge_ratio), gdzie hedge_ratio to współczynnik przy x.
    """
    X = pd.DataFrame({"x": x})
    X = sm.add_constant(X)  # kolumny: const, x
    model = sm.OLS(y, X).fit()
    hedge_ratio = model.params["x"]
    return model, hedge_ratio


def compute_spread(y, x, hedge_ratio):
    """
    Spread = y - hedge_ratio * x
    """
    spread = y - hedge_ratio * x
    spread.name = "spread"
    return spread


def compute_zscore(series):
    """
    Z-score: (x - mean) / std
    Zwraca: (z_series, current_z)
    """
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        z = pd.Series(index=series.index, data=np.nan)
        current = np.nan
    else:
        z = (series - mu) / sigma
        current = float(z.iloc[-1])
    return z, current


def estimate_half_life(spread):
    """
    Half-life dla procesu OU:
      Δs_t = α + β * s_{t-1} + ε
    λ ≈ -β, half-life = ln(2) / λ = -ln(2)/β  (dla β < 0)
    Zwraca (half_life_float, model)
    """
    df = pd.DataFrame({
        "lag": spread.shift(1),
        "ds": spread.diff()
    }).dropna()

    # Zabezpieczenie: wystarczająco obserwacji?
    if len(df) < 10:
        return math.inf, None

    X = sm.add_constant(df["lag"])  # kolumny: const, lag
    y = df["ds"]
    model = sm.OLS(y, X).fit()

    # Używamy NAZWY parametru, nie indeksu liczbowego!
    beta = model.params["lag"]

    # Zabezpieczenia numeryczne i interpretacyjne
    if beta >= 0 or np.isclose(beta, 0.0):
        # Brak średniopowrotności – half-life nie ma sensownej dodatniej wartości
        return math.inf, model

    half_life = -np.log(2.0) / beta  # beta < 0 => wartość dodatnia
    return float(half_life), model


def pretty_number(x, digits=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "∞"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(x)


# --- Główna logika ---
def main():
    # Parametry
    tickers = ["ADBE", "CRM"]
    start = "2022-01-01"  # możesz zmienić zakres dat
    end = None            # do dzisiaj

    # 1) Pobierz dane
    prices = fetch_prices(tickers, start=start, end=end)
    prices.columns = tickers  # upewnij się, że nazwy kolumn to ADBE, CRM

    # 2) Log diagnostyczny
    print("Pierwsze 5 obserwacji:")
    print(prices.head())
    print(f"\nLiczba obserwacji: {len(prices)}\n")

    # 3) Hedge ratio: ADBE ~ const + CRM
    adbe = prices["ADBE"]
    crm = prices["CRM"]
    hr_model, hedge_ratio = compute_hedge_ratio(y=adbe, x=crm)
    print(f"Hedge ratio (ADBE vs CRM): {pretty_number(hedge_ratio, 4)}\n")

    # 4) Spread
    spread = compute_spread(adbe, crm, hedge_ratio)
    print("Spread – pierwsze wartości:")
    print(spread.head())
    print("")

    # 5) Z-score (aktualny)
    z, current_z = compute_zscore(spread)
    if np.isnan(current_z):
        print("Aktualny Z-score: nieokreślony (brak odchylenia standardowego)\n")
    else:
        print(f"Aktualny Z-score: {pretty_number(current_z, 2)}\n")

    # 6) Half-life OU
    half_life, hl_model = estimate_half_life(spread)
    if math.isinf(half_life):
        print("Half-life: ∞ (beta >= 0 lub zbyt mało danych — brak średniopowrotności lub niestabilna estymacja)\n")
    else:
        print(f"Half-life (dni): {pretty_number(half_life, 2)}\n")

    # (opcjonalnie) Podsumowanie modelu half-life w debugowaniu:
    # if hl_model is not None:
    #     print(hl_model.summary())

    # 7) Prosta logika sygnałów (opcjonalnie, do wglądu)
    #    Wejście: |Z| > 2, Wyjście: |Z| < 0.5
    entry_z = 2.0
    exit_z = 0.5
    signal = None
    if not np.isnan(current_z):
        if current_z > entry_z:
            signal = "Short spread (sprzedaj ADBE, kup CRM)"
        elif current_z < -entry_z:
            signal = "Long spread (kup ADBE, sprzedaj CRM)"
        elif abs(current_z) < exit_z:
            signal = "Wyjście z pozycji (mean reversion osiągnięte)"
        else:
            signal = "Utrzymaj / brak nowego sygnału"

        print(f"Sygnał (reguły Z=±{entry_z}, wyjście {exit_z}): {signal}")

    # 8) Informacyjnie: data ostatniej obserwacji
    print(f"Ostatnia data w danych: {prices.index[-1].date()}")


if __name__ == "__main__":
    main()
