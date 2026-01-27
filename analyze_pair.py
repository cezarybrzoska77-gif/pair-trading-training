#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_pair.py
---------------
Analiza pary (domyślnie KLAC–LRCX):

- Pobranie danych (odporne na różne układy kolumn yfinance)
- Hedge ratio (beta), alpha
- Spread = A - (alpha + beta * B)
- Z-score (20/40/60)
- Volatility filter: VolRatio = sigma/mean z 60 dni
- ADF stationarity (p-value)
- Sygnały long/short
- Zapis CSV

Uruchomienie lokalnie:
    python analyze_pair.py --start-date 2018-01-01 --a KLAC --b LRCX --out-csv results/klac_lrcx_analysis.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ------------- Pomocnicze --------------

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def extract_prices(df: pd.DataFrame,
                   field_preferred: str = "Adj Close",
                   field_fallback: str = "Close",
                   tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Wyciąga macierz cen (daty x tickery) niezależnie od układu kolumn zwróconych przez yfinance.
    Obsługuje MultiIndex [field, ticker], [ticker, field] oraz SingleIndex.
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych z yfinance (DataFrame pusty).")

    cols = df.columns

    # MultiIndex: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred]
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback]
        if out.empty:
            raise KeyError(f"Nie ma kolumn '{field_preferred}' ani '{field_fallback}'.")
        return out

    # MultiIndex: [ticker, field]
    if isinstance(cols, pd.MultiIndex) and (
        field_preferred in cols.get_level_values(-1) or field_fallback in cols.get_level_values(-1)
    ):
        try:
            out = df.xs(field_preferred, axis=1, level=-1)
        except KeyError:
            if field_fallback in cols.get_level_values(-1):
                out = df.xs(field_fallback, axis=1, level=-1)
            else:
                raise KeyError(f"Nie ma kolumn '{field_preferred}' ani '{field_fallback}'.")
        return out

    # SingleIndex: spróbuj bezpośrednio
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    raise KeyError(f"Nie znaleziono '{field_preferred}' ani '{field_fallback}'. Kolumny={list(map(str, cols))}")


def download_prices(tickers: list[str], start_date: str, auto_adjust: bool = False) -> pd.DataFrame:
    """
    Pobiera dane przez yfinance i zwraca macierz cen (Adj Close preferowane, Close jako fallback).
    """
    info(f"Pobieranie danych: {tickers}")
    df = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=auto_adjust,     # jeśli True -> 'Close' już skorygowany
        progress=False,
        group_by="column"
    )
    if df is None or df.empty:
        raise RuntimeError("Brak danych z yfinance. Sprawdź tickery/datę/sieć.")

    if auto_adjust:
        prices = extract_prices(df, field_preferred="Close", field_fallback="Close", tickers=tickers)
    else:
        prices = extract_prices(df, field_preferred="Adj Close", field_fallback="Close", tickers=tickers)

    # proste czyszczenie
    prices = prices.dropna(how="all")
    min_non_na = max(30, int(0.4 * len(prices)))
    prices = prices.dropna(axis=1, thresh=min_non_na)
    prices = prices.sort_index(axis=1)

    if prices.empty:
        raise RuntimeError("Po czyszczeniu brak danych cenowych.")
    info(f"Dane OK: kształt = {prices.shape}")
    return prices


def hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """
    OLS: y ~ const + beta*x  → zwraca (alpha, beta, spread = y - (alpha + beta*x))
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 30:
        raise RuntimeError("Za mało danych do regresji (min 30 obserwacji).")
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    alpha = float(model.params.get("const", 0.0))
    beta = float(model.params.get("x", np.nan))
    spread = df["y"] - (alpha + beta * df["x"])
    spread.name = "spread"
    return alpha, beta, spread


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    return z


def adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 40:
        return np.nan
    try:
        res = adfuller(s)
        return float(res[1])
    except Exception:
        return np.nan


# ------------- Główna analiza --------------

def analyze_pair(a: str = "KLAC",
                 b: str = "LRCX",
                 start_date: str = "2018-01-01",
                 out_csv: str = "results/klac_lrcx_analysis.csv",
                 auto_adjust: bool = False) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # 1) Dane cenowe
    prices = download_prices([a, b], start_date=start_date, auto_adjust=auto_adjust)
    if a not in prices.columns or b not in prices.columns:
        raise RuntimeError(f"Brak kolumn dla {a} lub {b} po ekstrakcji.")

    A = prices[a].astype(float)
    B = prices[b].astype(float)

    # 2) Hedge ratio + spread
    alpha, beta, spread = hedge_ratio(A, B)
    info(f"Hedge ratio beta ({a} ~ {b}) = {beta:.4f}, alpha = {alpha:.4f}")

    # 3) Z-score
    z20 = zscore(spread, 20)
    z40 = zscore(spread, 40)
    z60 = zscore(spread, 60)

    # 4) Volatility: sigma/mean (60 dni) – zabezpieczenie przed dzieleniem przez 0
    sigma60 = spread.rolling(60).std()
    mean60 = spread.rolling(60).mean().abs().replace(0, np.nan)
    vol_ratio60 = sigma60 / mean60

    # 5) ADF p-value (na całym dostępnym spreadzie)
    p_adf = adf_pvalue(spread)
    info(f"ADF p-value (całość) = {p_adf:.4f}" if np.isfinite(p_adf) else "ADF p-value: brak (za mało danych)")

    # 6) Sygnały (prosto i konserwatywnie)
    long_signal = (z20 < -2.0) & (z40 < -1.5)
    short_signal = (z20 >  2.0) & (z40 >  1.5)

    # 7) Progi jakości (na dziś – ostatnia wartość)
    vol_ok_now = bool(np.isfinite(vol_ratio60.iloc[-1]) and vol_ratio60.iloc[-1] <= 4.0)
    adf_ok = bool(np.isfinite(p_adf) and p_adf <= 0.05)
    info(f"Volatility test (σ/mean ≤ 4.0, 60 dni) → {vol_ok_now}")
    info(f"Stationarity test (ADF p ≤ 0.05)     → {adf_ok}")

    # 8) Zapis szczegółów do CSV
    out = pd.DataFrame({
        a: A,
        b: B,
        "alpha": alpha,
        "beta": beta,
        "Spread": spread,
        "Z20": z20,
        "Z40": z40,
        "Z60": z60,
        "VolRatio60": vol_ratio60,
        "LongSignal": long_signal,
        "ShortSignal": short_signal
    })
    out.to_csv(out_csv)
    info(f"Zapisano analizę → {out_csv}")

    # 9) Krótka stopka podsumowania w logu
    last = out.dropna().iloc[-1]
    info(
        f"LAST | Spread={last['Spread']:.2f}  Z20={last['Z20']:.2f}  Z40={last['Z40']:.2f}  "
        f"VolR60={last['VolRatio60']:.2f}  long={bool(last['LongSignal'])}  short={bool(last['ShortSignal'])}"
    )
    info("Gotowe.")


# ------------- CLI --------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza pary: Z-score, Volatility, ADF.")
    parser.add_argument("--a", default="KLAC", help="Ticker A (domyślnie KLAC)")
    parser.add_argument("--b", default="LRCX", help="Ticker B (domyślnie LRCX)")
    parser.add_argument("--start-date", default="2018-01-01", help="Data początkowa (YYYY-MM-DD)")
    parser.add_argument("--out-csv", default="results/klac_lrcx_analysis.csv", help="Ścieżka do CSV z wynikami")
    parser.add_argument("--auto-adjust", action="store_true",
                        help="Jeśli ustawione, użyj yfinance auto_adjust=True (Close jest skorygowany).")
    args = parser.parse_args()

    analyze_pair(a=args.a, b=args.b, start_date=args.start_date, out_csv=args.out_csv, auto_adjust=args.auto_adjust)