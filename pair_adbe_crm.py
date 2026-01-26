
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_adbe_crm.py
Analiza pary ADBE–CRM:
- Pobiera Adj Close
- Liczy hedge ratio (OLS: ADBE ~ const + CRM)
- Buduje spread
- Liczy Z-score (wraz z ostatnią wartością)
- Estymuje half-life procesu OU: Δs_t = α + β * s_{t-1} + ε, half-life = -ln(2)/β (dla β < 0)

Wymagania: numpy, pandas, statsmodels, yfinance
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
        print("Brakująca biblioteka. Zainstaluj wymagane paczki:")
        print("  pip install numpy pandas statsmodels yfinance")
        raise

    # yfinance: spróbuj doinstalować automatycznie, jeśli brak
    try:
        import yfinance as yf
    except ImportError:
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
            import yfinance as yf
        except Exception as e:
            print("Nie udało się zainstalować yfinance automatycznie. Zainstaluj ręcznie:")
            print("  pip install yfinance")
            raise
    return np, pd, sm, yf

np, pd, sm, yf = _safe_imports()


# --- Funkcje pomocnicze ---
def fetch_prices(tickers, start=None, end=None):
    """
    Pobiera Adj Close dla listy tickerów i zwraca DataFrame o kolumnach = tickery oraz wspólnym indeksie dat.
    """
    if start is None:
        # ~4 lata wstecz (ok. 1000 sesji)
        start = (datetime.utcnow() - timedelta(days=365*4+30)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,    # wycisza progress bar
        auto_adjust=False
    )

    if data is None or len(data) == 0:
        raise RuntimeError("Brak danych z yfinance. Sprawdź tickery i połączenie.")

    # yfinance zwykle zwraca MultiIndex w kolumnach; wybierz "Adj Close"
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in set(lvl0):
            data = data["Adj Close"].copy()
        elif "Close" in set(lvl0):
            data = data["Close"].copy()
        else:
            # jeśli struktura nietypowa, spróbuj wydobyć po pierwszym poziomie
            raise RuntimeError("Nie znaleziono poziomu 'Adj Close' w danych yfinance.")
    else:
        # Płaskie kolumny — przyjmij, że to już są ceny zamknięcia skorygowane
        data = data.copy()

    # Jeśli dla jednego tickera dostalibyśmy Series, rzutuj na DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Uporządkuj kolumny według podanych tickerów i usuń braki z obu serii
    data = data.reindex(columns=tickers)
    data = data.dropna(how="any")

    if data.empty:
        raise RuntimeError("Po oczyszczeniu z NaN brak wspólnych obserwacji dla tickerów.")

    return data


def compute_hedge_ratio(y, x):
    """
    OLS: y ~ const + x
    Zwraca (model, hedge_ratio), gdzie hedge_ratio to współczynnik przy 'x'.
    """
    X = pd.DataFrame({"x": x})
    X = sm.add_constant(X)  # kolumny: const, x
    model = sm.OLS(y, X).fit()
    hedge_ratio = float(model.params["x"])
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
    Zwraca (z_series, current_z)
    """
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        z = pd.Series(index=series.index, data=np.nan, name="zscore")
        current = np.nan
    else:
        z = ((series - mu) / sigma).rename("zscore")
        current = float(z.iloc[-1])
    return z, current


def estimate_half_life(spread):
    """
    Half-life dla procesu OU:
      Δs_t = α + β * s_{t-1} + ε
    λ ≈ -β  =>  half-life = ln(2) / λ = -ln(2)/β  (dla β < 0)
    Zwraca (half_life_float, model)
    """
    df = pd.DataFrame({
        "lag": spread.shift(1),
        "ds": spread.diff()
    }).dropna()

    if len(df) < 10:
        return math.inf, None  # zbyt mało danych na sensowną estymację

    X = sm.add_constant(df["lag"])  # kolumny: const, lag
    y = df["ds"]
    model = sm.OLS(y, X).fit()

    # Kluczowa poprawka: odczyt parametru po NAZWIE, nie po indeksie liczbowym
    beta = float(model.params["lag"])

    if beta >= 0 or np.isclose(beta, 0.0):
        # Brak średniopowrotności (OU wymaga β < 0)
        return math.inf, model

    half_life = -math.log(2.0) / beta  # beta < 0 => wynik dodatni
    return float(half_life), model


def pretty_number(x, digits=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "∞"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(x)


# --- Główna logika ---
def main():
    # Parametry wejściowe
    tickers = ["ADBE", "CRM"]
    start = "2022-01-01"  # można zmienić na dłuższy okres
    end = None            # do dzisiaj

    # 1) Pobierz dane
    prices = fetch_prices(tickers, start=start, end=end)
    prices.columns = tickers  # upewnij się, że nazwy kolumn to dokładnie ADBE, CRM

    # 2) Podgląd danych
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
        print("Half-life: ∞ (β >= 0, zbyt mało danych lub brak średniopowrotności)\n")
    else:
        print(f"Half-life (dni): {pretty_number(half_life, 2)}\n")

    # (opcjonalnie) Jeśli debugujesz:
    # if hl_model is not None:
    #     print(hl_model.summary())

    # 7) Prosta logika sygnałów (opcjonalna)
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

    print(f"Ostatnia data w danych: {prices.index[-1].date()}")


if __name__ == "__main__":
    main()
